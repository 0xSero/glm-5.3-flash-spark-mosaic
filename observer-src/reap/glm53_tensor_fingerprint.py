"""Canonical tensor fingerprints for layout-independent GLM-5.3 comparison."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import torch
from safetensors import safe_open

from reap.glm53_refusal_directions import _atomic_json, _sha256


def _digest_field(digest: hashlib._Hash, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def canonical_tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    """Hash tensor identity and logical bytes, independent of file layout."""
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    _digest_field(digest, name.encode("utf-8"))
    _digest_field(digest, str(contiguous.dtype).encode("ascii"))
    _digest_field(
        digest,
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"),
    )
    _digest_field(digest, memoryview(contiguous.view(torch.uint8).numpy()))
    return digest.hexdigest()


def _tree_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        _digest_field(digest, row["name"].encode("utf-8"))
        _digest_field(digest, bytes.fromhex(row["canonical_tensor_sha256"]))
    return digest.hexdigest()


def fingerprint_tree(
    *,
    output: Path,
    abliteration_verification_path: Path,
    report_path: Path,
    stop_after_shards: int | None = None,
) -> dict[str, Any]:
    output = output.resolve()
    report_path = report_path.resolve()
    if report_path == output or output.is_relative_to(report_path):
        raise RuntimeError("fingerprint report path must not contain the model tree")
    if stop_after_shards is not None and stop_after_shards < 1:
        raise ValueError("stop-after-shards must be a positive integer")

    manifest_path = output / "abliteration-manifest.json"
    index_path = output / "model.safetensors.index.json"
    manifest = json.loads(manifest_path.read_text())
    verification = json.loads(abliteration_verification_path.read_text())
    if manifest.get("state") != "COMPLETE":
        raise RuntimeError("abliteration manifest is not complete")
    if verification.get("state") != "COMPLETE":
        raise RuntimeError("abliteration verification is not complete")
    if verification.get("manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("abliteration verification does not bind the manifest")
    if verification.get("output_index_sha256") != _sha256(index_path):
        raise RuntimeError("abliteration verification does not bind the output index")

    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    completed_source = {
        row["name"]: row for row in manifest.get("completed_shards", [])
    }
    if len(shard_names) != 120 or set(completed_source) != set(shard_names):
        raise RuntimeError(
            "abliteration manifest and index do not close over 120 shards"
        )
    keys_by_shard: dict[str, list[str]] = {name: [] for name in shard_names}
    for name, shard in weight_map.items():
        keys_by_shard[shard].append(name)
    for names in keys_by_shard.values():
        names.sort()

    base = {
        "schema": "glm53-canonical-tensor-tree-fingerprint-v1",
        "state": "RUNNING",
        "abliteration_manifest_sha256": _sha256(manifest_path),
        "abliteration_verification_sha256": _sha256(abliteration_verification_path),
        "output_index_sha256": _sha256(index_path),
        "completed_shards": [],
    }
    report = base
    if report_path.is_file():
        prior = json.loads(report_path.read_text())
        for key in (
            "schema",
            "abliteration_manifest_sha256",
            "abliteration_verification_sha256",
            "output_index_sha256",
        ):
            if prior.get(key) != base[key]:
                raise RuntimeError(f"fingerprint resume report differs at {key}")
        report = prior
    completed = {row["name"]: row for row in report["completed_shards"]}

    for shard_number, shard_name in enumerate(shard_names, 1):
        if stop_after_shards is not None and len(completed) >= stop_after_shards:
            report["state"] = "RUNNING"
            report["pilot_stop_after_shards"] = stop_after_shards
            _atomic_json(report_path, report)
            return report
        source_row = completed_source[shard_name]
        expected = {
            "name": shard_name,
            "source_file_sha256": source_row["output_sha256"],
            "source_file_bytes": source_row["bytes"],
            "tensor_count": len(keys_by_shard[shard_name]),
        }
        prior = completed.get(shard_name)
        if prior is not None and all(
            prior.get(key) == value for key, value in expected.items()
        ):
            continue

        path = output / shard_name
        if not path.is_file() or path.stat().st_size != source_row["bytes"]:
            raise RuntimeError(f"abliterated shard size mismatch: {shard_name}")
        shard_digest = hashlib.sha256()
        tensor_bytes = 0
        with safe_open(path, framework="pt", device="cpu") as handle:
            if sorted(handle.keys()) != keys_by_shard[shard_name]:
                raise RuntimeError(f"tensor key closure mismatch: {shard_name}")
            for name in keys_by_shard[shard_name]:
                tensor = handle.get_tensor(name).contiguous()
                tensor_digest = canonical_tensor_sha256(name, tensor)
                _digest_field(shard_digest, name.encode("utf-8"))
                _digest_field(shard_digest, bytes.fromhex(tensor_digest))
                tensor_bytes += tensor.numel() * tensor.element_size()
                del tensor
        row = {
            **expected,
            "tensor_bytes": tensor_bytes,
            "canonical_tensor_sha256": shard_digest.hexdigest(),
        }
        completed[shard_name] = row
        report["completed_shards"] = [completed[name] for name in sorted(completed)]
        _atomic_json(report_path, report)
        print(
            json.dumps(
                {
                    "event": "fingerprint_shard_complete",
                    "completed": shard_number,
                    "total": len(shard_names),
                    "name": shard_name,
                }
            ),
            flush=True,
        )
        gc.collect()

    rows = [completed[name] for name in shard_names]
    tensor_count = sum(row["tensor_count"] for row in rows)
    tensor_bytes = sum(row["tensor_bytes"] for row in rows)
    if tensor_count != len(weight_map):
        raise RuntimeError("canonical fingerprint tensor count does not close")
    if tensor_bytes != index["metadata"]["total_size"]:
        raise RuntimeError("canonical fingerprint tensor bytes do not close")
    report.update(
        {
            "state": "COMPLETE",
            "shards": len(rows),
            "tensors": tensor_count,
            "tensor_bytes": tensor_bytes,
            "canonical_tree_sha256": _tree_digest(rows),
        }
    )
    report.pop("pilot_stop_after_shards", None)
    _atomic_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--abliteration-verification", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stop-after-shards", type=int)
    args = parser.parse_args()
    report = fingerprint_tree(
        output=args.output,
        abliteration_verification_path=args.abliteration_verification,
        report_path=args.report,
        stop_after_shards=args.stop_after_shards,
    )
    print(
        json.dumps(
            {
                "state": report["state"],
                "completed_shards": len(report["completed_shards"]),
                "canonical_tree_sha256": report.get("canonical_tree_sha256"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
