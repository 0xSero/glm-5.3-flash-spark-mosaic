"""Independently verify a streamed GLM-5.3 BF16 abliteration tree."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from reap.glm53_abliterate_bf16 import target_kind
from reap.glm53_refusal_directions import _atomic_json, _sha256


def projection_metrics(
    source: torch.Tensor,
    output: torch.Tensor,
    direction: torch.Tensor,
    kind: str,
    *,
    device: torch.device,
    chunk_size: int = 4096,
) -> dict[str, float]:
    if source.shape != output.shape or source.dtype != output.dtype:
        raise RuntimeError("source/output target shape or dtype changed")
    direction = direction.to(device=device, dtype=torch.float32)
    direction /= torch.linalg.vector_norm(direction)
    source_sq = 0.0
    delta_sq = 0.0
    residual_sq = 0.0
    axis_size = output.shape[0] if kind == "embedding" else output.shape[1]
    for start in range(0, axis_size, chunk_size):
        stop = min(start + chunk_size, axis_size)
        if kind == "embedding":
            original = source[start:stop].to(device=device, dtype=torch.float32)
            changed = output[start:stop].to(device=device, dtype=torch.float32)
            residual = changed @ direction
        else:
            original = source[:, start:stop].to(device=device, dtype=torch.float32)
            changed = output[:, start:stop].to(device=device, dtype=torch.float32)
            residual = direction @ changed
        source_sq += float(original.square().sum())
        delta_sq += float((changed - original).square().sum())
        residual_sq += float(residual.square().sum())
    return {
        "relative_delta": (delta_sq / source_sq) ** 0.5,
        "relative_projection_residual": (residual_sq / source_sq) ** 0.5,
    }


def verify(
    *,
    source: Path,
    output: Path,
    direction_path: Path,
    report_path: Path,
    device: torch.device,
    chunk_size: int = 4096,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    manifest_path = output / "abliteration-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("state") != "COMPLETE":
        raise RuntimeError("abliteration manifest is not complete")
    if manifest.get("direction_sha256") != _sha256(direction_path):
        raise RuntimeError("verification direction digest mismatch")
    direction = load_file(direction_path).get("direction")
    if direction is None or direction.ndim != 1:
        raise RuntimeError("verification direction is missing or malformed")

    source_index_path = source / "model.safetensors.index.json"
    output_index_path = output / "model.safetensors.index.json"
    if source_index_path.read_bytes() != output_index_path.read_bytes():
        raise RuntimeError("output tensor index differs from source")
    index = json.loads(source_index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    completed = {row["name"]: row for row in manifest["completed_shards"]}
    shard_names = sorted(set(weight_map.values()))
    if set(completed) != set(shard_names):
        raise RuntimeError("abliteration manifest does not close over indexed shards")

    transformed = 0
    untouched = 0
    maximum_residual = 0.0
    minimum_delta = float("inf")
    for shard_number, shard_name in enumerate(shard_names, 1):
        source_path = source / shard_name
        output_path = output / shard_name
        row = completed[shard_name]
        if _sha256(source_path) != row["source_sha256"]:
            raise RuntimeError(f"source shard changed: {shard_name}")
        if _sha256(output_path) != row["output_sha256"]:
            raise RuntimeError(f"output shard digest mismatch: {shard_name}")
        source_tensors = load_file(source_path)
        output_tensors = load_file(output_path)
        expected_keys = {name for name, mapped in weight_map.items() if mapped == shard_name}
        if set(source_tensors) != expected_keys or set(output_tensors) != expected_keys:
            raise RuntimeError(f"tensor key closure mismatch: {shard_name}")
        shard_transformed = 0
        for name in sorted(expected_keys):
            original = source_tensors[name]
            changed = output_tensors[name]
            kind = target_kind(name)
            if kind is None:
                if not torch.equal(original, changed):
                    raise RuntimeError(f"untargeted tensor changed: {name}")
                untouched += 1
                continue
            if torch.equal(original, changed):
                raise RuntimeError(f"targeted tensor did not change: {name}")
            metrics = projection_metrics(
                original,
                changed,
                direction,
                kind,
                device=device,
                chunk_size=chunk_size,
            )
            maximum_residual = max(
                maximum_residual, metrics["relative_projection_residual"]
            )
            minimum_delta = min(minimum_delta, metrics["relative_delta"])
            transformed += 1
            shard_transformed += 1
        if shard_transformed != row["transformed_tensors"]:
            raise RuntimeError(f"transformed tensor count mismatch: {shard_name}")
        print(
            json.dumps(
                {
                    "event": "verification_shard_complete",
                    "completed": shard_number,
                    "total": len(shard_names),
                    "name": shard_name,
                }
            ),
            flush=True,
        )
        del original, changed, source_tensors, output_tensors
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if transformed != manifest["transformed_tensors"]:
        raise RuntimeError("global transformed tensor count mismatch")
    if untouched != manifest["untargeted_tensors"]:
        raise RuntimeError("global untouched tensor count mismatch")
    if maximum_residual > manifest["maximum_residual"]:
        raise RuntimeError("independent projection residual exceeds manifest limit")
    report = {
        "schema": "glm53-bf16-abliteration-verification-v1",
        "state": "COMPLETE",
        "manifest_sha256": _sha256(manifest_path),
        "direction_sha256": _sha256(direction_path),
        "source_index_sha256": _sha256(source_index_path),
        "output_index_sha256": _sha256(output_index_path),
        "shards": len(shard_names),
        "transformed_tensors": transformed,
        "untargeted_tensors": untouched,
        "maximum_relative_projection_residual": maximum_residual,
        "minimum_relative_delta": minimum_delta,
    }
    _atomic_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    report = verify(
        source=args.source,
        output=args.output,
        direction_path=args.direction,
        report_path=args.report,
        device=torch.device(args.device),
        chunk_size=args.chunk_size,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
