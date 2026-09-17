"""Extract the source-precision portion of a verified ablated GLM-5.3 tree."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from reap.glm53_refusal_directions import _atomic_json, _sha256
from reap.glm53_virtual_abliteration import VirtualAbliteration


ROUTED = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\."
    r"(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
EXPECTED_QUANTIZED_TENSORS = 36_288
EXPECTED_RETAINED_TENSORS = 2_482
EXPECTED_RETAINED_TENSOR_BYTES = 33_835_039_608


def is_quantized(name: str) -> bool:
    match = ROUTED.fullmatch(name)
    return bool(
        match and 3 <= int(match.group(1)) <= 44 and 0 <= int(match.group(2)) < 288
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(memoryview(contiguous.numpy())).hexdigest()


def _write_shard(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file({name: value.contiguous() for name, value in tensors.items()}, temporary, metadata=metadata)
    temporary.chmod(0o644)
    temporary.replace(path)


def _valid_prior(path: Path, sidecar: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file() or not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text())
        for key, value in expected.items():
            if payload.get(key) != value:
                return None
        if payload.get("bytes") != path.stat().st_size or payload.get("sha256") != _sha256(path):
            return None
        with safe_open(path, framework="pt", device="cpu") as handle:
            if sorted(handle.keys()) != payload.get("keys"):
                return None
        return payload
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def extract_retained(
    *,
    source: Path,
    output: Path,
    abliteration_manifest_path: Path,
    source_manifest_path: Path | None = None,
    virtual_descriptor_path: Path | None = None,
    expected_quantized_tensors: int = EXPECTED_QUANTIZED_TENSORS,
    expected_retained_tensors: int = EXPECTED_RETAINED_TENSORS,
    expected_retained_tensor_bytes: int = EXPECTED_RETAINED_TENSOR_BYTES,
    stop_after_shards: int | None = None,
    only_source_shard: str | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise RuntimeError("source and retained output trees must be disjoint")
    if stop_after_shards is not None and stop_after_shards < 1:
        raise ValueError("stop-after-shards must be a positive integer")
    abliteration = json.loads(abliteration_manifest_path.read_text())
    if abliteration.get("state") != "COMPLETE":
        raise RuntimeError("abliteration manifest is not complete")
    index_path = source / "model.safetensors.index.json"
    if abliteration.get("output_index_sha256") != _sha256(index_path):
        raise RuntimeError("abliteration manifest does not bind the source index")
    virtual = None
    virtual_descriptor_sha256 = None
    if virtual_descriptor_path is not None:
        if source_manifest_path is None:
            raise RuntimeError("virtual retained extraction requires a source manifest")
        virtual = VirtualAbliteration(
            descriptor_path=virtual_descriptor_path,
            source_root=source,
            device=torch.device("cpu"),
        )
        virtual_descriptor_sha256 = _sha256(virtual_descriptor_path)
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    quantized = sorted(name for name in weight_map if is_quantized(name))
    retained = sorted(name for name in weight_map if not is_quantized(name))
    if len(quantized) != expected_quantized_tensors:
        raise RuntimeError(f"quantized tensor scope mismatch: {len(quantized)}")
    if len(retained) != expected_retained_tensors:
        raise RuntimeError(f"retained tensor scope mismatch: {len(retained)}")
    source_evidence = (
        json.loads(source_manifest_path.read_text())
        if source_manifest_path is not None
        else abliteration
    )
    completed_source = {
        row["name"]: row for row in source_evidence.get("shards", source_evidence.get("completed_shards", []))
    }
    shard_names = sorted(set(weight_map.values()))
    if set(completed_source) != set(shard_names):
        raise RuntimeError("abliteration manifest and source index have different shard closure")
    by_shard: dict[str, list[str]] = {}
    for name in retained:
        by_shard.setdefault(weight_map[name], []).append(name)
    if only_source_shard is not None and only_source_shard not in by_shard:
        raise ValueError(
            "requested source shard has no retained tensors: "
            f"{only_source_shard}"
        )

    manifest_path = output / "retained-manifest.json"
    if output.exists() and not manifest_path.is_file() and any(output.iterdir()):
        raise RuntimeError("nonempty retained output has no resumable manifest")
    output.mkdir(parents=True, exist_ok=True)
    derivation = {
        "schema": "glm53-abliterated-retained-v1",
        "state": "RUNNING",
        "abliteration_manifest_sha256": _sha256(abliteration_manifest_path),
        "source_index_sha256": _sha256(index_path),
        "base_model_revision": abliteration["model_revision"],
        "quantized_tensor_count": len(quantized),
        "retained_tensor_count": len(retained),
        "virtual_abliteration_descriptor_sha256": virtual_descriptor_sha256,
        "source_manifest_sha256": (
            _sha256(source_manifest_path) if source_manifest_path is not None else None
        ),
        "completed_shards": [],
    }
    manifest = derivation
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        for key in (
            "schema",
            "abliteration_manifest_sha256",
            "source_index_sha256",
            "base_model_revision",
            "quantized_tensor_count",
            "retained_tensor_count",
            "virtual_abliteration_descriptor_sha256",
            "source_manifest_sha256",
        ):
            if prior.get(key) != derivation[key]:
                raise RuntimeError(f"retained resume manifest differs at {key}")
        manifest = prior
    completed = {row["source_shard"]: row for row in manifest["completed_shards"]}
    total_tensor_bytes = sum(int(row["tensor_bytes"]) for row in completed.values())
    retained_weight_map: dict[str, str] = {}
    for row in completed.values():
        for name in row["keys"]:
            retained_weight_map[name] = f"retained/{row['output_name']}"

    source_shards = (
        [only_source_shard]
        if only_source_shard is not None
        else sorted(by_shard)
    )
    for source_shard in source_shards:
        if stop_after_shards is not None and len(completed) >= stop_after_shards:
            manifest["state"] = "RUNNING"
            manifest["pilot_stop_after_shards"] = stop_after_shards
            _atomic_json(manifest_path, manifest)
            return manifest
        keys = sorted(by_shard[source_shard])
        source_path = source / source_shard
        source_row = completed_source[source_shard]
        source_sha = _sha256(source_path)
        if (
            source_sha != source_row.get("output_sha256", source_row.get("sha256"))
            or source_path.stat().st_size != source_row["bytes"]
        ):
            raise RuntimeError(f"abliterated source shard mismatch: {source_shard}")
        number = source_shard.removeprefix("model-").removesuffix("-of-00120.safetensors")
        output_name = f"retained-{number}-of-00120.safetensors"
        output_path = output / "shards" / output_name
        sidecar = output_path.with_name(output_path.name + ".json")
        expected = {
            "state": "COMPLETE",
            "source_shard": source_shard,
            "source_shard_sha256": source_sha,
            "abliteration_manifest_sha256": derivation[
                "abliteration_manifest_sha256"
            ],
            "virtual_abliteration_descriptor_sha256": virtual_descriptor_sha256,
            "keys": keys,
        }
        prior = _valid_prior(output_path, sidecar, expected)
        if prior is not None:
            completed[source_shard] = prior
            continue
        tensors: dict[str, torch.Tensor] = {}
        tensor_digests: dict[str, str] = {}
        tensor_bytes = 0
        with safe_open(source_path, framework="pt", device="cpu") as handle:
            if not set(keys).issubset(set(handle.keys())):
                raise RuntimeError(f"retained keys are missing from {source_shard}")
            for name in keys:
                tensor = handle.get_tensor(name).contiguous()
                if virtual is not None:
                    tensor = virtual.get(name, tensor)
                tensors[name] = tensor
                tensor_digests[name] = tensor_sha256(tensor)
                tensor_bytes += tensor.numel() * tensor.element_size()
        _write_shard(
            output_path,
            tensors,
            {
                "format": "pt",
                "abliteration_manifest_sha256": derivation[
                    "abliteration_manifest_sha256"
                ],
                "virtual_abliteration_descriptor_sha256": (
                    virtual_descriptor_sha256 or "none"
                ),
            },
        )
        with safe_open(output_path, framework="pt", device="cpu") as handle:
            if sorted(handle.keys()) != keys:
                raise RuntimeError(f"retained key mismatch after write: {output_name}")
            for name in keys:
                if tensor_sha256(handle.get_tensor(name)) != tensor_digests[name]:
                    raise RuntimeError(f"retained tensor changed after write: {name}")
        row = {
            **expected,
            "output_name": output_name,
            "tensor_count": len(keys),
            "tensor_bytes": tensor_bytes,
            "tensor_sha256": tensor_digests,
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        }
        _atomic_json(sidecar, row)
        completed[source_shard] = row
        total_tensor_bytes = sum(int(item["tensor_bytes"]) for item in completed.values())
        manifest["completed_shards"] = [
            completed[name] for name in sorted(completed)
        ]
        _atomic_json(manifest_path, manifest)
        del tensors
        gc.collect()

    if len(completed) < len(by_shard):
        manifest["state"] = "RUNNING"
        manifest["completed_shards"] = [
            completed[name] for name in sorted(completed)
        ]
        _atomic_json(manifest_path, manifest)
        return manifest

    total_tensor_bytes = sum(int(row["tensor_bytes"]) for row in completed.values())
    if total_tensor_bytes != expected_retained_tensor_bytes:
        raise RuntimeError(
            f"retained tensor byte mismatch: {total_tensor_bytes} != "
            f"{expected_retained_tensor_bytes}"
        )
    if sum(int(row["tensor_count"]) for row in completed.values()) != len(retained):
        raise RuntimeError("retained tensor count does not close")
    retained_weight_map = {
        name: f"retained/{row['output_name']}"
        for row in completed.values()
        for name in row["keys"]
    }
    manifest.update(
        {
            "state": "COMPLETE",
            "retained_tensor_bytes": total_tensor_bytes,
            "retained_shard_count": len(completed),
            "weight_map": dict(sorted(retained_weight_map.items())),
        }
    )
    manifest.pop("pilot_stop_after_shards", None)
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--abliteration-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--virtual-abliteration-descriptor", type=Path)
    parser.add_argument("--stop-after-shards", type=int)
    parser.add_argument("--only-source-shard")
    args = parser.parse_args()
    manifest = extract_retained(
        source=args.source,
        output=args.output,
        abliteration_manifest_path=args.abliteration_manifest,
        source_manifest_path=args.source_manifest,
        virtual_descriptor_path=args.virtual_abliteration_descriptor,
        stop_after_shards=args.stop_after_shards,
        only_source_shard=args.only_source_shard,
    )
    print(
        json.dumps(
            {
                "state": manifest["state"],
                "completed_shards": len(manifest["completed_shards"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
