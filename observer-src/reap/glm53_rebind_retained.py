"""Rebind retained exact tensors to an equivalent immutable Hub revision.

This is intentionally a metadata-only operation. It is valid only when the old
and new virtual-abliteration descriptors are identical except for the immutable
Hub revision, and every retained shard still matches its sealed byte digest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open

from reap.glm53_refusal_directions import _atomic_json, _sha256


def _descriptor_semantics(payload: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(payload)
    comparable.pop("hub_revision", None)
    return comparable


def rebind_retained(
    *,
    retained_root: Path,
    old_descriptor_path: Path,
    new_descriptor_path: Path,
) -> dict[str, Any]:
    retained_root = retained_root.resolve()
    manifest_path = retained_root / "retained-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    old_descriptor = json.loads(old_descriptor_path.read_text())
    new_descriptor = json.loads(new_descriptor_path.read_text())
    old_sha = _sha256(old_descriptor_path)
    new_sha = _sha256(new_descriptor_path)
    old_revision = old_descriptor.get("hub_revision")
    new_revision = new_descriptor.get("hub_revision")
    if not (
        manifest.get("state") == "COMPLETE"
        and old_descriptor.get("state") == new_descriptor.get("state") == "COMPLETE"
        and old_descriptor.get("mode")
        == new_descriptor.get("mode")
        == "immutable_hub_exact_overlay"
        and isinstance(old_revision, str)
        and isinstance(new_revision, str)
        and old_revision != new_revision
        and old_sha != new_sha
        and _descriptor_semantics(old_descriptor)
        == _descriptor_semantics(new_descriptor)
        and manifest.get("virtual_abliteration_descriptor_sha256")
        in {old_sha, new_sha}
    ):
        raise RuntimeError("retained descriptor rebind is outside its seal")

    manifest_before_sha = manifest.get("descriptor_rebind", {}).get(
        "retained_manifest_before_sha256", _sha256(manifest_path)
    )
    evidence = {
        "schema": "glm53-retained-descriptor-rebind-v1",
        "state": "COMPLETE",
        "old_descriptor_sha256": old_sha,
        "new_descriptor_sha256": new_sha,
        "old_hub_revision": old_revision,
        "new_hub_revision": new_revision,
        "hub_repo_id": new_descriptor["hub_repo_id"],
        "hub_shard_closure_sha256": new_descriptor[
            "hub_shard_closure_sha256"
        ],
        "canonical_tree_sha256": new_descriptor["canonical_tree_sha256"],
        "retained_manifest_before_sha256": manifest_before_sha,
        "tensor_files_rewritten": False,
    }
    completed = []
    for original_row in manifest.get("completed_shards", []):
        row = dict(original_row)
        shard_path = retained_root / "shards" / row["output_name"]
        sidecar_path = shard_path.with_name(shard_path.name + ".json")
        sidecar = json.loads(sidecar_path.read_text())
        if not (
            shard_path.stat().st_size == row["bytes"] == sidecar.get("bytes")
            and _sha256(shard_path) == row["sha256"] == sidecar.get("sha256")
            and row["keys"] == sidecar.get("keys")
            and sidecar.get("virtual_abliteration_descriptor_sha256")
            in {old_sha, new_sha}
        ):
            raise RuntimeError(f"retained shard differs during rebind: {shard_path.name}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if not (
                sorted(handle.keys()) == row["keys"]
                and metadata.get("virtual_abliteration_descriptor_sha256") == old_sha
            ):
                raise RuntimeError(
                    f"retained materialization descriptor differs: {shard_path.name}"
                )
        row["virtual_abliteration_descriptor_sha256"] = new_sha
        row["retained_tensor_materialization_descriptor_sha256"] = old_sha
        row["descriptor_rebind"] = evidence
        _atomic_json(sidecar_path, row)
        completed.append(row)

    if len(completed) != manifest.get("retained_shard_count"):
        raise RuntimeError("retained shard closure differs during rebind")
    manifest["completed_shards"] = completed
    manifest["virtual_abliteration_descriptor_sha256"] = new_sha
    manifest["retained_tensor_materialization_descriptor_sha256"] = old_sha
    manifest["descriptor_rebind"] = evidence
    _atomic_json(manifest_path, manifest)
    return {
        **evidence,
        "retained_manifest_sha256": _sha256(manifest_path),
        "retained_shards": len(completed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained-root", type=Path, required=True)
    parser.add_argument("--old-descriptor", type=Path, required=True)
    parser.add_argument("--new-descriptor", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            rebind_retained(
                retained_root=args.retained_root,
                old_descriptor_path=args.old_descriptor,
                new_descriptor_path=args.new_descriptor,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
