"""CPU-only synchronization of sealed production receipts from a local mirror.

Dry-run is the default and performs no network calls or writes. The explicitly
supplied qualification is a caller-trusted input, not a signed attestation.
Historical upload markers are never fresh remote-acceptance evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from types import SimpleNamespace

from reap import glm53_observation_pipeline as pipeline
from reap.glm53_observation_suite import digest, runtime_identity, validate_plan, merge_shards
from reap.glm53_observation_worker import file_sha256, preflight_sidecar, read_json


def safe_path(path: Path) -> Path:
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("path traversal is not permitted")
    path = path.absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError("symlink paths are not permitted")
    return path


def validate_destination(repo, prefix, *, required):
    if not repo and not prefix and not required:
        return
    if (not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo)
            or not isinstance(prefix, str) or not prefix or prefix.startswith("/")
            or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) or part in (".", "..") for part in prefix.split("/"))):
        raise ValueError("explicit safe repository and relative sidecar prefix are required")


def qualified_runtime(plan, qualification):
    """Reconstruct the canonical qualification, including all observed counts."""
    if (not isinstance(qualification, dict) or qualification.get("schema") != "glm53-pipeline-qualification-v1"
            or qualification.get("state") != "QUALIFICATION_COMPLETE" or qualification.get("run_id") != plan["run_id"]):
        raise ValueError("qualification schema/state/run_id mismatch")
    body = {key: value for key, value in qualification.items() if key != "sha256"}
    if qualification.get("sha256") != digest(body):
        raise ValueError("qualification digest mismatch")
    runtime = runtime_identity(qualification.get("runtime_identity"))
    sid, samples = qualification.get("source_shard_id"), qualification.get("sequence_count")
    if type(sid) is not int or not 0 <= sid < len(plan["shards"]):
        raise ValueError("invalid qualification source shard")
    if type(samples) is not int or not 1 <= samples <= 64:
        raise ValueError("invalid qualification sample count")
    shard = plan["shards"][sid]
    observations = qualification.get("observations", {})
    stages = [{"stage_id": rank, "owned_layers": list(layers), "shard_id": sid,
               "start": shard["start"], "end": shard["start"] + samples, "runtime": runtime,
               "observations": {str(layer): observations[str(layer)] for layer in layers if layer >= 3}}
              for rank, layers in enumerate(pipeline.STAGE_LAYERS)]
    expected = pipeline.make_qualification(plan, sid, samples, stages, runtime)
    if qualification != expected:
        raise ValueError("qualification observations or full contract do not match")
    return runtime


def _marker(path, binding):
    safe_path(path)
    if not path.exists():
        return False
    marker = read_json(path)
    if (set(marker) != {*binding, "commit"} or any(marker.get(key) != value for key, value in binding.items())
            or not isinstance(marker.get("commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", marker["commit"])):
        raise ValueError("historical upload marker binding mismatch")
    return True


def synchronize(*, plan_path, run_root, qualification_path, repo=None, prefix=None, upload=False,
                preflight=preflight_sidecar, upload_receipt=None, finalize=None):
    validate_destination(repo, prefix, required=upload)
    plan_path, root, qualification_path = map(safe_path, (plan_path, run_root, qualification_path))
    if not root.is_dir():
        raise ValueError("run-root must be an existing mirrored run directory")
    plan = read_json(plan_path)
    validate_plan(plan)
    if plan["shard_size"] != 64 or len(plan["shards"]) != 375:
        raise ValueError("sidecar requires the exact 375 x 64-row production plan")
    qualification = read_json(qualification_path)
    runtime = qualified_runtime(plan, qualification)
    if (root / "qualification-result.json").exists() or (root / "qualification-result.json").is_symlink():
        raise ValueError("smoke qualification output cannot serve as production run-root")
    records, historical = [], []
    for directory in sorted(root.iterdir()):
        if not directory.name.startswith("shard-"):
            continue
        safe_path(directory)
        if not re.fullmatch(r"shard-[0-9]{5}", directory.name) or not directory.is_dir():
            raise ValueError("invalid production shard directory")
        sid = int(directory.name[6:])
        if sid >= 375:
            raise ValueError("production shard outside exact plan")
        path = safe_path(directory / "receipt.json")
        marker_path = safe_path(directory / "upload_verified.json")
        if not path.is_file():
            # Empty/in-progress directories are permitted but never uploaded.
            if marker_path.exists():
                raise ValueError("orphan upload marker without sealed receipt")
            continue
        receipt = read_json(path)
        pipeline.validate_receipt(plan, sid, receipt, runtime)
        sha = file_sha256(path)
        if marker_path.exists():
            marker = read_json(marker_path)
            selected_repo = repo or marker.get("repo_id")
            selected_remote = (f"{prefix}/{plan['run_id']}/shards/{sid:05d}.json" if prefix else marker.get("path_in_repo"))
            if not isinstance(selected_remote, str) or not selected_remote.endswith(f"/{plan['run_id']}/shards/{sid:05d}.json"):
                raise ValueError("historical marker has unsafe artifact destination")
            historical_prefix = selected_remote.removesuffix(f"/{plan['run_id']}/shards/{sid:05d}.json")
            validate_destination(selected_repo, historical_prefix, required=True)
            _marker(marker_path, {"receipt_sha256": sha, "repo_id": selected_repo, "path_in_repo": selected_remote})
            historical.append(sid)
        records.append((sid, path, sha))
    # Prevalidate any existing merged artifacts before the first network call.
    merged_path = safe_path(root / "merged-observations.json")
    merged_marker = safe_path(root / "merged-upload-verified.json")
    if merged_path.exists() or merged_marker.exists():
        if len(records) != 375 or not merged_path.is_file():
            raise ValueError("merged artifact/marker requires a complete mirrored lane")
        expected = merge_shards(plan, (read_json(path) for _, path, _ in records))
        if read_json(merged_path) != expected:
            raise ValueError("existing merged observations differ from validated receipts")
        if merged_marker.exists():
            marker = read_json(merged_marker)
            selected_repo = repo or marker.get("repo_id")
            selected_remote = f"{prefix}/{plan['run_id']}/merged-observations.json" if prefix else marker.get("path_in_repo")
            suffix = f"/{plan['run_id']}/merged-observations.json"
            if not isinstance(selected_remote, str) or not selected_remote.endswith(suffix):
                raise ValueError("historical merged destination mismatch")
            validate_destination(selected_repo, selected_remote.removesuffix(suffix), required=True)
            _marker(merged_marker, {"artifact_sha256": file_sha256(merged_path),
                                    "repo_id": selected_repo, "path_in_repo": selected_remote})
    result = {"schema": "glm53-production-sidecar-sync-v1", "state": "DRY_RUN",
              "run_id": plan["run_id"], "qualified_runtime_sha256": digest(runtime),
              "qualification_sha256": qualification["sha256"], "validated_shards": [sid for sid, _, _ in records],
              "validated_samples": len(records) * 64, "total_shards": 375,
              "historical_markers": historical, "newly_uploaded_shards": [],
              "full_lane_pass": False, "full_suite_pass": False,
              "remote_acceptance": "NOT_CURRENTLY_VERIFIED", "network_used": False,
              "trust": "caller-supplied qualification; historical local markers are not fresh remote evidence"}
    if not upload:
        return result
    if not records:
        raise ValueError("no sealed production receipts to upload")
    preflight(repo)  # Existing private destination and write permission; no creation.
    result["network_used"] = True
    args = SimpleNamespace(sidecar_repo=repo, sidecar_prefix=prefix, no_upload=False, limit_shards=None)
    uploader = upload_receipt or pipeline._upload_receipt
    for sid, path, sha in records:
        if file_sha256(path) != sha:
            raise ValueError("sealed mirror changed after prevalidation")
        uploader(path, args, plan, sid)
        if sid not in historical:
            result["newly_uploaded_shards"].append(sid)
    for _, path, sha in records:
        if file_sha256(path) != sha:
            raise ValueError("sealed mirror changed during synchronization")
    if len(records) == 375:
        finalized = (finalize or pipeline.finalize_lane)(plan, root, [sid for sid, _, _ in records], args)
        result["merged_artifact_recorded"] = finalized.get("state") == "FULL_LANE_COMPLETE"
    result["state"] = "SIDECAR_SYNC_RECORDED_REMOTE_ACCEPTANCE_PENDING"
    # Full acceptance requires the separate campaign --verify-remote audit,
    # including fresh readback of any previously recorded upload commits.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--repo")
    parser.add_argument("--prefix")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--upload", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = synchronize(plan_path=args.plan, run_root=args.run_root,
                         qualification_path=args.qualification, repo=args.repo, prefix=args.prefix,
                         upload=args.upload)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
