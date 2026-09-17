"""Sequential bounded-shard launcher. GPU dependencies load in the subprocess.

Token manifest paths are relative to the manifest directory; the plan binds its
complete bytes with corpus_identity.sha256. Only aggregate receipts are uploaded.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable

from reap.glm53_observation_suite import LAYERS, SEQUENCE_LENGTH, SAMPLES, SUM_FIELDS, make_shard_receipt, runtime_identity, validate_plan

OBSERVATION_FIELDS = {*SUM_FIELDS, "expert_frequency", "total_tokens", "max_activations", "activation_norm_max"}


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def token_identity_field(manifest: dict, key: str) -> Any:
    nested = manifest.get("identity", {}).get(key)
    outer = manifest.get(key)
    if outer is not None and nested is not None and outer != nested:
        raise ValueError(f"token manifest identity disagreement: {key}")
    result = outer if outer is not None else nested
    if result is None:
        raise ValueError(f"token manifest lacks identity: {key}")
    return result


def verify_inputs(plan: dict, token_manifest: Path, model_root: Path) -> dict[int, dict]:
    validate_plan(plan)
    if file_sha256(token_manifest) != plan["corpus_identity"]["sha256"]:
        raise ValueError("token manifest sha256 mismatch")
    if file_sha256(model_root / "model.safetensors.index.json") != plan["model_identity"]["sha256"]:
        raise ValueError("model index sha256 mismatch")
    manifest = read_json(token_manifest)
    if (manifest.get("state") != "COMPLETE" or manifest.get("sequence_count") != SAMPLES
            or manifest.get("sequence_length") != SEQUENCE_LENGTH
            or manifest.get("tokens") != SAMPLES * SEQUENCE_LENGTH):
        raise ValueError("token manifest is not the full complete corpus")
    if file_sha256(model_root / "tokenizer.json") != token_identity_field(manifest, "tokenizer_sha256"):
        raise ValueError("tokenizer sha256 mismatch")
    shards = {}
    for shard in manifest.get("shards", []):
        sid = shard.get("shard_id")
        if type(sid) is not int or sid in shards or not 0 <= sid < len(plan["shards"]):
            raise ValueError("duplicate or invalid token shard")
        expected = plan["shards"][sid]
        if any(shard.get(key) != expected[key] for key in ("start", "end")):
            raise ValueError("token shard range mismatch")
        relative = Path(shard["path"])
        path = (token_manifest.parent / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(token_manifest.parent.resolve()) or not path.is_file():
            raise ValueError("token shard escapes manifest root or is missing")
        shards[sid] = {**shard, "resolved_path": path}
    if set(shards) != set(range(len(plan["shards"]))):
        raise ValueError("incomplete token shard coverage")
    return shards


def observer_command(*, python: str, plan: dict, shard: dict, token_path: Path,
                     model_root: Path, output: Path, corpus: str, group_size: int) -> list[str]:
    if corpus not in ("ours", "wikipedia") or group_size < 1:
        raise ValueError("invalid corpus or group_size")
    return [python, "-m", "reap.glm53_exl3_observe", "--model-root", str(model_root),
            "--corpus", corpus, "--output", str(output), "--sequence-count",
            str(shard["end"] - shard["start"]), "--sequence-length", str(SEQUENCE_LENGTH),
            "--group-size", str(group_size), "--model-revision", plan["model_identity"]["revision"],
            "--dataset-revision", plan["corpus_identity"]["revision"],
            "--prepared-tokens", str(token_path)]


def upload_verified(path: Path, repo: str, remote_path: str) -> str:
    # Lazy import: planning, tests and dry-runs need neither HF nor torch.
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    if api.repo_info(repo, repo_type="dataset").private is not True:
        raise ValueError("observation sidecar destination must already be private")
    commit = api.upload_file(path_or_fileobj=path, path_in_repo=remote_path,
                             repo_id=repo, repo_type="dataset",
                             commit_message=f"Seal full REAP shard {path.stem}")
    downloaded = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=remote_path,
                                     revision=commit.oid, force_download=True))
    if file_sha256(path) != file_sha256(downloaded):
        raise ValueError("uploaded receipt verification failed")
    return commit.oid


def preflight_sidecar(repo: str) -> None:
    """Fail before GPU work if the configured token cannot write this private sink.

    auth_check is Hub's read-only permission check; unlike creating an upload
    commit, it does not mutate the repository. Unsupported clients fail closed.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    if api.repo_info(repo, repo_type="dataset").private is not True:
        raise ValueError("observation sidecar destination must already be private")
    api.auth_check(repo_id=repo, repo_type="dataset", write=True)


def clean_replay(output: Path) -> None:
    # Only exact generated slots; no globs, parent cleanup or model/token deletion.
    for name in ("state-a", "state-b"):
        path = output / name
        if path.is_symlink():
            raise ValueError("refusing symlink replay cleanup")
        if path.exists():
            if not path.is_dir() or path.resolve().parent != output.resolve():
                raise ValueError("unexpected replay path")
            shutil.rmtree(path)


def run_worker(*, plan: dict, token_manifest: Path, model_root: Path, output_root: Path,
               worker_id: int, corpus: str, sidecar_repo: str, sidecar_prefix: str,
               python: str = sys.executable, group_size: int = 1,
               delete_uploaded_scratch: bool = False, dry_run: bool = False,
               limit_shards: int | None = None,
               run: Callable = subprocess.run, upload: Callable = upload_verified,
               preflight: Callable = preflight_sidecar) -> dict:
    if type(worker_id) is not int or not 0 <= worker_id < plan.get("workers", 0):
        raise ValueError("worker id is outside plan")
    if limit_shards is not None and limit_shards < 1:
        raise ValueError("limit_shards must be positive")
    if not sidecar_repo or not sidecar_prefix.strip("/") or ".." in Path(sidecar_prefix).parts:
        raise ValueError("explicit sidecar destination required")
    shards = verify_inputs(plan, token_manifest, model_root)
    token_metadata = read_json(token_manifest)
    if token_identity_field(token_metadata, "corpus") != corpus:
        raise ValueError("corpus label differs from token manifest")
    if not dry_run:
        preflight(sidecar_repo)
    root = output_root.resolve() / plan["run_id"]
    if root.is_symlink():
        raise ValueError("refusing symlink run output")
    root.mkdir(parents=True, exist_ok=True)
    completed, commands = [], []
    # One worker id owns its ranges; prohibit simultaneous launches/resume races.
    with (root / f"worker-{worker_id}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assigned = [s for s in plan["shards"] if s["worker"] == worker_id]
        for shard in assigned:
            sid = shard["shard_id"]
            source = shards[sid]
            if file_sha256(source["resolved_path"]) != source["sha256"]:
                raise ValueError(f"token shard {sid} sha256 mismatch")
            output = root / f"shard-{sid:05d}"
            if output.is_symlink():
                raise ValueError("refusing symlink shard output")
            command = observer_command(python=python, plan=plan, shard=shard,
                                       token_path=source["resolved_path"], model_root=model_root,
                                       output=output, corpus=corpus, group_size=group_size)
            commands.append(command)
            if dry_run:
                if limit_shards and len(commands) >= limit_shards:
                    break
                continue
            output.mkdir(parents=True, exist_ok=True)
            receipt_path = output / "receipt.json"
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                runtime = runtime_identity(read_json(output / "run_identity.json"))
                if receipt != make_shard_receipt(plan, sid, receipt.get("observations", {}), runtime=runtime):
                    raise ValueError("existing receipt does not match plan")
                if any(set(payload) - OBSERVATION_FIELDS for payload in receipt["observations"].values()):
                    raise ValueError("receipt contains fields outside aggregate allowlist")
            else:
                run(command, check=True)
                manifest = read_json(output / "manifest.json")
                expected = {"state": "COMPLETE", "sequence_count": shard["end"] - shard["start"],
                            "sequence_length": SEQUENCE_LENGTH, "tokens": (shard["end"] - shard["start"]) * SEQUENCE_LENGTH,
                            "model_revision": plan["model_identity"]["revision"], "corpus": corpus,
                            "layers": list(LAYERS)}
                if any(manifest.get(key) != value for key, value in expected.items()):
                    raise ValueError("observer manifest does not seal expected shard")
                observations = {}
                for layer in LAYERS:
                    raw = read_json(output / "observations" / f"layer-{layer:02d}.json")
                    observations[str(layer)] = {key: value for key, value in raw.items() if key in OBSERVATION_FIELDS}
                runtime = runtime_identity(read_json(output / "run_identity.json"))
                receipt = make_shard_receipt(plan, sid, observations, runtime=runtime)
                atomic_json(receipt_path, receipt)
            remote_path = f"{sidecar_prefix.strip('/')}/{plan['run_id']}/shards/{sid:05d}.json"
            uploaded_path = output / "upload_verified.json"
            binding = {"receipt_sha256": file_sha256(receipt_path), "repo_id": sidecar_repo,
                       "path_in_repo": remote_path}
            if uploaded_path.exists():
                marker = read_json(uploaded_path)
                if any(marker.get(key) != value for key, value in binding.items()) or not marker.get("commit"):
                    raise ValueError("upload marker does not match receipt/destination")
            else:
                commit = upload(receipt_path, sidecar_repo, remote_path)
                if not isinstance(commit, str) or not commit:
                    raise ValueError("upload returned no verified commit")
                atomic_json(uploaded_path, {**binding, "commit": commit})
            if delete_uploaded_scratch:
                clean_replay(output)
            completed.append(sid)
            atomic_json(root / f"worker-{worker_id}.json", {"state": "RUNNING", "run_id": plan["run_id"],
                        "worker_id": worker_id, "verified_shards": completed})
            if limit_shards and len(completed) >= limit_shards:
                break
    result = {"state": "DRY_RUN" if dry_run else ("WORKER_COMPLETE" if len(completed) == len(assigned) else "PARTIAL"),
              "run_id": plan["run_id"], "worker_id": worker_id, "verified_shards": completed}
    if dry_run:
        result["commands"] = commands
    else:
        atomic_json(root / f"worker-{worker_id}.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "token-manifest", "model-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--corpus", choices=("ours", "wikipedia"), required=True)
    parser.add_argument("--sidecar-repo", required=True)
    parser.add_argument("--sidecar-prefix", required=True)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--delete-uploaded-scratch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-shards", type=int)
    args = vars(parser.parse_args())
    args["plan"] = read_json(args["plan"])
    print(json.dumps(run_worker(**args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
