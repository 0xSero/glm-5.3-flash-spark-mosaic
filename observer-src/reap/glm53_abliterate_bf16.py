"""Stream rank-one refusal-direction ablation over pinned GLM-5.3 BF16 shards.

The source tree is read-only.  Each source shard is digest-checked, transformed
independently, and written to a separate resumable output tree.  Only hidden-
state writers are changed: token embeddings, attention output projections, and
dense/shared/routed MLP down projections.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from reap.glm53_refusal_directions import _atomic_json, _sha256

EMBEDDING_KEY = "model.language_model.embed_tokens.weight"
OUTPUT_PATTERN = re.compile(
    r"model\.language_model\.layers\.\d+\.self_attn\.o_proj\.weight"
)
DOWN_PATTERN = re.compile(
    r"model\.language_model\.layers\.\d+\.mlp\."
    r"(?:down_proj|shared_experts\.down_proj|experts\.\d+\.down_proj)\.weight"
)


def target_kind(name: str) -> str | None:
    if name == EMBEDDING_KEY:
        return "embedding"
    if OUTPUT_PATTERN.fullmatch(name) or DOWN_PATTERN.fullmatch(name):
        return "output"
    return None


def expected_target_counts(config: dict[str, Any]) -> dict[str, int]:
    text = config["text_config"]
    primary_layers = int(text["num_hidden_layers"])
    nextn_layers = int(text.get("num_nextn_predict_layers", 0))
    layer_types = list(text["mlp_layer_types"])
    if len(layer_types) != primary_layers or not layer_types:
        raise RuntimeError("MLP layer-type count does not match primary decoder layers")
    # GLM-5.3 appends each MTP predictor to the indexed decoder-layer sequence
    # using the final primary layer's MLP type; the config list covers only the
    # primary stack.
    layer_types.extend([layer_types[-1]] * nextn_layers)
    expected_layers = primary_layers + nextn_layers
    routed = int(text["n_routed_experts"])
    shared = int(text["n_shared_experts"])
    down = sum(
        routed + shared if layer_type == "sparse" else 1 for layer_type in layer_types
    )
    return {"embedding": 1, "output": expected_layers + down}


def project_tensor(
    tensor: torch.Tensor,
    direction: torch.Tensor,
    kind: str,
    *,
    device: torch.device,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, dict[str, float]]:
    if tensor.ndim != 2:
        raise RuntimeError(f"target tensor must be rank two, got {tuple(tensor.shape)}")
    hidden_size = direction.numel()
    if kind == "embedding" and tensor.shape[1] != hidden_size:
        raise RuntimeError("embedding hidden dimension does not match direction")
    if kind == "output" and tensor.shape[0] != hidden_size:
        raise RuntimeError("writer output dimension does not match direction")
    direction = direction.to(device=device, dtype=torch.float32)
    direction = direction / torch.linalg.vector_norm(direction)
    output = torch.empty_like(tensor)
    original_sq = 0.0
    delta_sq = 0.0
    residual_sq = 0.0

    axis_size = tensor.shape[0] if kind == "embedding" else tensor.shape[1]
    for start in range(0, axis_size, chunk_size):
        stop = min(start + chunk_size, axis_size)
        if kind == "embedding":
            original = tensor[start:stop].to(device=device, dtype=torch.float32)
            coefficients = original @ direction
            projected = original - coefficients[:, None] * direction[None, :]
            cast = projected.to(dtype=tensor.dtype)
            output[start:stop] = cast.cpu()
            recast = cast.float()
            residual = recast @ direction
        else:
            original = tensor[:, start:stop].to(device=device, dtype=torch.float32)
            coefficients = direction @ original
            projected = original - direction[:, None] * coefficients[None, :]
            cast = projected.to(dtype=tensor.dtype)
            output[:, start:stop] = cast.cpu()
            recast = cast.float()
            residual = direction @ recast
        original_sq += float(original.square().sum())
        delta_sq += float((recast - original).square().sum())
        residual_sq += float(residual.square().sum())

    if original_sq <= 0 or delta_sq <= 0:
        raise RuntimeError("projection did not change a nonzero target tensor")
    return output, {
        "relative_delta": (delta_sq / original_sq) ** 0.5,
        "relative_projection_residual": (residual_sq / original_sq) ** 0.5,
    }


def _validate_selection(selection_path: Path, direction_path: Path) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text())
    if selection.get("state") != "COMPLETE":
        raise RuntimeError("refusal-direction selection is not complete")
    if selection.get("direction_sha256") != _sha256(direction_path):
        raise RuntimeError("selected direction digest mismatch")
    return selection


def _write_shard(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()}, temporary
    )
    temporary.chmod(0o644)
    temporary.replace(path)


def abliterated_model_card(
    *, source_repo_id: str, source_revision: str, selected_candidate: dict[str, Any]
) -> str:
    layer = selected_candidate.get("layer", "recorded in abliteration-manifest.json")
    return f"""---
base_model: {source_repo_id}
library_name: transformers
license: mit
tags:
  - glm
  - mixture-of-experts
  - abliterated
---

# GLM-5.3 Flash BF16 — abliterated

This full-precision checkpoint is derived from `{source_repo_id}` at immutable
revision `{source_revision}`. A refusal direction selected jointly from sealed
Q4 and 3.0bpw observations (candidate layer `{layer}`) was projected from the
token embedding and hidden-state writer matrices. The transformation is
recorded in `abliteration-manifest.json` and independently checked by the
separate verification report before downstream quantization or publication.

This card makes no standalone refusal-behavior, generation-quality, serving,
or benchmark claim. Those results are published only when their named reports
have completed. No calibration prompts or prompt-token rows are included.
"""


def _overwrite_safetensor_tensors(
    path: Path, replacements: dict[str, torch.Tensor]
) -> None:
    """Overwrite tensor payload ranges without changing a safetensors header."""
    with path.open("rb") as handle:
        encoded_length = handle.read(8)
        if len(encoded_length) != 8:
            raise RuntimeError(f"invalid safetensors header prefix: {path}")
        header_length = struct.unpack("<Q", encoded_length)[0]
        header = json.loads(handle.read(header_length))
    data_start = 8 + header_length
    descriptor = os.open(path, os.O_RDWR)
    try:
        for name in sorted(replacements):
            metadata = header.get(name)
            if not isinstance(metadata, dict):
                raise RuntimeError(f"replacement tensor is absent from header: {name}")
            start, stop = metadata["data_offsets"]
            tensor = replacements[name].detach().cpu().contiguous()
            raw = tensor.view(torch.uint8).numpy().reshape(-1)
            if raw.size != stop - start:
                raise RuntimeError(f"replacement byte count differs for {name}")
            payload = memoryview(raw)
            written = 0
            while written < len(payload):
                chunk_stop = min(written + (16 << 20), len(payload))
                count = os.pwrite(
                    descriptor,
                    payload[written:chunk_stop],
                    data_start + start + written,
                )
                if count <= 0:
                    raise RuntimeError(f"short safetensors payload write for {name}")
                written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o644)
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name, expected in replacements.items():
            if not torch.equal(handle.get_tensor(name), expected.cpu()):
                raise RuntimeError(
                    f"sparse safetensors rewrite did not round-trip: {name}"
                )


def _reflink_clone(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    result = subprocess.run(
        [
            "cp",
            "--reflink=always",
            "--preserve=mode,timestamps",
            str(source),
            str(destination),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"reflink clone failed for {source.name}: {result.stderr.strip()}"
        )


def abliterate(
    *,
    source: Path,
    output: Path,
    source_manifest_path: Path,
    direction_path: Path,
    selection_path: Path,
    model_revision: str,
    device: torch.device,
    chunk_size: int = 4096,
    maximum_residual: float = 5e-4,
    write_mode: str = "safetensors",
    stop_after_shards: int | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if (
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise RuntimeError("source and output trees must be disjoint")
    if write_mode not in {"safetensors", "reflink"}:
        raise ValueError("write mode must be safetensors or reflink")
    if stop_after_shards is not None and stop_after_shards < 1:
        raise ValueError("stop-after-shards must be a positive integer")
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("revision") != model_revision:
        raise RuntimeError("source-manifest revision does not match requested revision")
    selection = _validate_selection(selection_path, direction_path)
    direction_tensors = load_file(direction_path)
    direction = direction_tensors.get("direction")
    if direction is None or direction.ndim != 1:
        raise RuntimeError("selected direction tensor is missing or malformed")

    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    config = json.loads((source / "config.json").read_text())
    expected_counts = expected_target_counts(config)
    target_names = {name for name in weight_map if target_kind(name) is not None}
    observed_counts = {
        "embedding": sum(target_kind(name) == "embedding" for name in target_names),
        "output": sum(target_kind(name) == "output" for name in target_names),
    }
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"target inventory mismatch: expected {expected_counts}, got {observed_counts}"
        )

    source_rows = {row["name"]: row for row in source_manifest["shards"]}
    shard_names = sorted(set(weight_map.values()))
    if set(shard_names) != set(source_rows):
        raise RuntimeError(
            "source manifest and tensor index have different shard closure"
        )
    manifest_path = output / "abliteration-manifest.json"
    if output.exists() and not manifest_path.is_file() and any(output.iterdir()):
        raise RuntimeError(
            "nonempty output tree has no resumable abliteration manifest"
        )
    output.mkdir(parents=True, exist_ok=True)
    base_manifest = {
        "schema": "glm53-streamed-bf16-abliteration-v1",
        "state": "RUNNING",
        "model_revision": model_revision,
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_index_sha256": _sha256(index_path),
        "selection_sha256": _sha256(selection_path),
        "direction_sha256": _sha256(direction_path),
        "selected_candidate": selection.get("selected"),
        "target_counts": observed_counts,
        "maximum_residual": maximum_residual,
        "write_mode": write_mode,
        "completed_shards": [],
    }
    manifest = base_manifest
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        for key in (
            "model_revision",
            "source_manifest_sha256",
            "source_index_sha256",
            "selection_sha256",
            "direction_sha256",
            "target_counts",
            "maximum_residual",
            "write_mode",
        ):
            if prior.get(key) != base_manifest[key]:
                raise RuntimeError(f"resume manifest differs at {key}")
        manifest = prior
    completed = {row["name"]: row for row in manifest["completed_shards"]}

    shard_targets: dict[str, list[str]] = {name: [] for name in shard_names}
    for name in sorted(target_names):
        shard_targets[weight_map[name]].append(name)
    transformed_total = sum(map(len, shard_targets.values()))
    if transformed_total != sum(observed_counts.values()):
        raise RuntimeError("target-to-shard inventory is inconsistent")

    for shard_number, shard_name in enumerate(shard_names, 1):
        if stop_after_shards is not None and len(completed) >= stop_after_shards:
            manifest["state"] = "RUNNING"
            manifest["pilot_stop_after_shards"] = stop_after_shards
            _atomic_json(manifest_path, manifest)
            return manifest
        destination = output / shard_name
        prior = completed.get(shard_name)
        if (
            prior
            and destination.is_file()
            and _sha256(destination) == prior["output_sha256"]
        ):
            print(json.dumps({"event": "shard_resume", "name": shard_name}), flush=True)
            continue
        source_path = source / shard_name
        source_sha = _sha256(source_path)
        if source_sha != source_rows[shard_name]["sha256"]:
            raise RuntimeError(f"source shard digest mismatch: {shard_name}")
        expected_keys = {
            name for name, mapped in weight_map.items() if mapped == shard_name
        }
        with safe_open(source_path, framework="pt", device="cpu") as handle:
            source_keys = set(handle.keys())
        if source_keys != expected_keys:
            raise RuntimeError(f"source shard key closure mismatch: {shard_name}")
        metrics = []
        if write_mode == "reflink":
            replacements = {}
            with safe_open(source_path, framework="pt", device="cpu") as handle:
                for name in shard_targets[shard_name]:
                    kind = target_kind(name)
                    assert kind is not None
                    projected, values = project_tensor(
                        handle.get_tensor(name),
                        direction,
                        kind,
                        device=device,
                        chunk_size=chunk_size,
                    )
                    if values["relative_projection_residual"] > maximum_residual:
                        raise RuntimeError(f"projection residual exceeds limit: {name}")
                    replacements[name] = projected
                    metrics.append({"name": name, "kind": kind, **values})
            _reflink_clone(source_path, destination)
            _overwrite_safetensor_tensors(destination, replacements)
            del replacements
        else:
            tensors = load_file(source_path)
            for name in shard_targets[shard_name]:
                kind = target_kind(name)
                assert kind is not None
                projected, values = project_tensor(
                    tensors[name], direction, kind, device=device, chunk_size=chunk_size
                )
                if values["relative_projection_residual"] > maximum_residual:
                    raise RuntimeError(f"projection residual exceeds limit: {name}")
                tensors[name] = projected
                metrics.append({"name": name, "kind": kind, **values})
            _write_shard(destination, tensors)
            del tensors
        row = {
            "name": shard_name,
            "source_sha256": source_sha,
            "output_sha256": _sha256(destination),
            "bytes": destination.stat().st_size,
            "transformed_tensors": len(metrics),
            "maximum_relative_projection_residual": max(
                (item["relative_projection_residual"] for item in metrics), default=0.0
            ),
            "minimum_relative_delta": min(
                (item["relative_delta"] for item in metrics), default=0.0
            ),
        }
        completed[shard_name] = row
        manifest["completed_shards"] = [
            completed[name] for name in shard_names if name in completed
        ]
        _atomic_json(manifest_path, manifest)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "shard_complete",
                    "completed": shard_number,
                    "total": len(shard_names),
                    "name": shard_name,
                    "transformed_tensors": len(metrics),
                }
            ),
            flush=True,
        )
        if (
            stop_after_shards is not None
            and len(completed) >= stop_after_shards
            and len(completed) < len(shard_names)
        ):
            manifest["state"] = "RUNNING"
            manifest["pilot_stop_after_shards"] = stop_after_shards
            _atomic_json(manifest_path, manifest)
            return manifest

    for path in source.iterdir():
        if path.is_file() and path.name not in shard_names:
            shutil.copy2(path, output / path.name)
    (output / "README.md").write_text(
        abliterated_model_card(
            source_repo_id=source_manifest.get("repo_id", "zai-org/GLM-5.3-Flash-BF16"),
            source_revision=model_revision,
            selected_candidate=selection.get("selected") or {},
        ),
        encoding="utf-8",
    )
    manifest["state"] = "COMPLETE"
    manifest["shards"] = len(shard_names)
    manifest["transformed_tensors"] = transformed_total
    manifest["untargeted_tensors"] = len(weight_map) - transformed_total
    manifest["output_index_sha256"] = _sha256(output / index_path.name)
    manifest["files"] = [
        {
            "path": row["name"],
            "bytes": row["bytes"],
            "sha256": row["output_sha256"],
        }
        for row in manifest["completed_shards"]
    ]
    manifest.pop("pilot_stop_after_shards", None)
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--maximum-residual", type=float, default=5e-4)
    parser.add_argument(
        "--write-mode", choices=("safetensors", "reflink"), default="safetensors"
    )
    parser.add_argument("--stop-after-shards", type=int)
    args = parser.parse_args()
    manifest = abliterate(
        source=args.source,
        output=args.output,
        source_manifest_path=args.source_manifest,
        direction_path=args.direction,
        selection_path=args.selection,
        model_revision=args.model_revision,
        device=torch.device(args.device),
        chunk_size=args.chunk_size,
        maximum_residual=args.maximum_residual,
        write_mode=args.write_mode,
        stop_after_shards=args.stop_after_shards,
    )
    print(
        json.dumps(
            {
                "state": manifest["state"],
                "completed_shards": len(manifest["completed_shards"]),
                "shards": manifest.get("shards"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
