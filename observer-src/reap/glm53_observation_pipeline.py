"""Two resident stages; only sealed aggregate receipts reach disk or the Hub.

Launch with torchrun (two nodes, one process each). GPU libraries are lazy so
protocol validation is CPU-testable. Plan worker assignments are not pipeline
stage assignments: both stages jointly execute every requested corpus shard.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import fcntl
import importlib
import json
import os
from pathlib import Path
import re
import time

from reap.glm53_observation_suite import (
    LAYERS, SEQUENCE_LENGTH, _validated_observation, digest, make_shard_receipt,
    merge_shards, runtime_identity, validate_plan,
)
from reap.glm53_observation_worker import (
    OBSERVATION_FIELDS, atomic_json, file_sha256, preflight_sidecar, read_json,
    token_identity_field, upload_verified, verify_inputs,
)

STAGE_LAYERS = (tuple(range(23)), tuple(range(23, 45)))
STAGE_METADATA = {"model.safetensors.index.json", "config.json", "tokenizer.json",
                  "tokenizer_config.json", "generation_config.json", "processor_config.json",
                  "chat_template.jinja"}


def verify_stage_source(plan, model_root, stage_id):
    """Verify precisely the owned files, not an impossible full local checkpoint."""
    if type(stage_id) is not int or stage_id not in (0, 1):
        raise ValueError("invalid source stage id")
    receipt = read_json(model_root / "observation-stage-receipt.json")
    body = {key: value for key, value in receipt.items() if key != "sha256"}
    if receipt.get("sha256") != digest(body):
        raise ValueError("stage source receipt digest mismatch")
    model = plan["model_identity"]
    expected = {"state": "RESIDENT_STAGE_COMPLETE", "repo": model["name"],
                "revision": model["revision"], "stage_id": stage_id,
                "owned_layers": list(STAGE_LAYERS[stage_id]), "stop_after_layer": None,
                "index_sha256": model["sha256"]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("stage source receipt does not match pinned plan and ownership")
    index_path = model_root / "model.safetensors.index.json"
    if file_sha256(index_path) != model["sha256"]:
        raise ValueError("stage source index mismatch")
    mapping = read_json(index_path)["weight_map"]
    selected_weights = set()
    for key, filename in mapping.items():
        match = re.match(r"^model\.language_model\.layers\.(\d+)\.", key)
        if ((key == "model.language_model.embed_tokens.weight" and stage_id == 0)
                or (match and int(match[1]) in STAGE_LAYERS[stage_id])):
            selected_weights.add(filename)
    entries = receipt.get("selected_files")
    if not isinstance(entries, list) or receipt.get("staged_files") != len(entries):
        raise ValueError("invalid selected file receipt")
    files = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"file", "bytes", "sha256"}:
            raise ValueError("invalid selected file metadata")
        name = entry["file"]
        if not isinstance(name, str) or name in files:
            raise ValueError("duplicate or invalid selected file")
        relative = Path(name)
        path = model_root / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(model_root.resolve()) or not path.is_file()):
            raise ValueError("unsafe or missing selected stage file")
        if (type(entry["bytes"]) is not int or entry["bytes"] < 0
                or not isinstance(entry["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
            raise ValueError("invalid selected file size or sha256")
        files[name] = entry
    required_metadata = {"model.safetensors.index.json", "config.json", "tokenizer.json"}
    if (set(files) - STAGE_METADATA != selected_weights
            or not required_metadata <= set(files)
            or receipt.get("weight_files") != len(set(mapping.values()))):
        raise ValueError("selected stage files do not exactly cover owned weights")
    # Read and hash every selected weight before importing/loading GPU modules.
    for name, entry in files.items():
        path = model_root / name
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"selected stage file checksum mismatch: {name}")
    return {"stage_id": stage_id, "receipt_sha256": receipt["sha256"],
            "selected_files_sha256": digest(entries), "config_sha256": files["config.json"]["sha256"]}


def runtime_code_identity(paths, transformers_version, stage_sources, *, batch_experts=False):
    if type(batch_experts) is not bool or (batch_experts and "batched_helper" not in paths):
        raise ValueError("batched expert mode requires an explicit flag and helper hash")
    if len(stage_sources) != 2 or [item.get("stage_id") for item in stage_sources] != [0, 1]:
        raise ValueError("source verification must cover both ordered stages")
    if stage_sources[0]["config_sha256"] != stage_sources[1]["config_sha256"]:
        raise ValueError("stage model configurations differ")
    return {"code_sha256": {name: file_sha256(path) for name, path in paths.items()},
            "transformers_version": transformers_version, "stage_sources": stage_sources,
            "expert_execution": "full-sequence" if batch_experts else "per-attention-chunk"}


def validate_token_geometry(shape, dtype, samples):
    if list(shape) != [samples, SEQUENCE_LENGTH] or str(dtype) not in ("torch.int64", "I64"):
        raise ValueError("token shard must contain exact full-length int64 samples")


def validate_peer_identities(identities):
    if len(identities) != 2 or identities[0] != identities[1]:
        raise ValueError("pipeline peers disagree on plan, tokens, model, or runtime")
    runtime_identity(identities[0]["runtime"])


def _merge_stage_range(expected, stages, runtime):
    """Validate disjoint ownership without changing any full-suite contract."""
    if len(stages) != 2:
        raise ValueError("exactly two stage payloads required")
    merged, seen = {}, set()
    for stage in stages:
        stage_id = stage.get("stage_id")
        if type(stage_id) is not int or stage_id not in (0, 1) or stage_id in seen:
            raise ValueError("duplicate or invalid stage id")
        seen.add(stage_id)
        if any(stage.get(key) != expected[key] for key in ("shard_id", "start", "end")):
            raise ValueError("stage sample range mismatch")
        if stage.get("runtime") != runtime:
            raise ValueError("stage runtime mismatch")
        if stage.get("owned_layers") != list(STAGE_LAYERS[stage_id]):
            raise ValueError("stage layer ownership mismatch")
        observations = stage.get("observations", {})
        wanted = {str(layer) for layer in STAGE_LAYERS[stage_id] if layer in LAYERS}
        if set(observations) != wanted or set(merged) & set(observations):
            raise ValueError("missing, overlapping, or foreign routed layers")
        if any(set(payload) - OBSERVATION_FIELDS for payload in observations.values()):
            raise ValueError("non-aggregate fields in stage observations")
        merged.update(observations)
    return merged


def merge_stage_observations(plan, shard_id, stages, runtime):
    """Disjoint stage ownership plus existing full receipt validation."""
    merged = _merge_stage_range(plan["shards"][shard_id], stages, runtime)
    # Checks all 42 layers, sample token counts, route counts, finiteness, maxima.
    make_shard_receipt(plan, shard_id, merged, runtime=runtime)
    return merged


def select_smoke_shard(plan, committed, samples, no_upload):
    if not no_upload:
        raise ValueError("--smoke-samples requires --no-upload")
    selected = next((s for s in plan["shards"] if s["shard_id"] not in committed), plan["shards"][0])
    if type(samples) is not int or not 1 <= samples <= selected["end"] - selected["start"]:
        raise ValueError("smoke samples must fit the selected first shard")
    return selected


def make_qualification(plan, shard_id, samples, stages, runtime):
    validate_plan(plan)
    runtime_identity(runtime)
    source = plan["shards"][shard_id]
    if type(samples) is not int or not 1 <= samples <= source["end"] - source["start"]:
        raise ValueError("invalid qualification sample count")
    observed = {**source, "end": source["start"] + samples}
    observations = _merge_stage_range(observed, stages, runtime)
    for payload in observations.values():
        _validated_observation(payload, samples * SEQUENCE_LENGTH)
    result = {"schema": "glm53-pipeline-qualification-v1", "state": "QUALIFICATION_COMPLETE",
              "run_id": plan["run_id"], "source_shard_id": shard_id,
              "observed_range": {"start": observed["start"], "end": observed["end"]},
              "sequence_count": samples, "sequence_length": SEQUENCE_LENGTH,
              "tokens": samples * SEQUENCE_LENGTH, "observations": observations,
              "runtime_identity": runtime, "full_lane_pass": False, "full_suite_pass": False,
              "activation_disk_writes": False, "uploaded": False}
    result["sha256"] = digest(result)
    return result


def write_qualification(root, result):
    path = root / "qualification-result.json"
    if path.is_symlink():
        raise ValueError("refusing symlink qualification output")
    atomic_json(path, result)


def validate_receipt(plan, shard_id, receipt, runtime):
    expected = make_shard_receipt(plan, shard_id, receipt.get("observations", {}), runtime=runtime)
    if receipt != expected:
        raise ValueError("receipt digest, range, plan, or runtime mismatch")
    if any(set(value) - OBSERVATION_FIELDS for value in receipt["observations"].values()):
        raise ValueError("receipt contains non-aggregate fields")


def _export_stage(stage, shard, rank, runtime):
    observations = {}
    for layer, accumulator in stage.observations.items():
        raw = accumulator.as_dict() if hasattr(accumulator, "as_dict") else accumulator
        observations[str(layer)] = {key: value for key, value in raw.items() if key in OBSERVATION_FIELDS}
    return {"stage_id": rank, "owned_layers": list(stage.owned_layers),
            **{key: shard[key] for key in ("shard_id", "start", "end")},
            "runtime": runtime, "observations": observations}


def _runtime(group_size, chunk_size, model_root, stage_sources, *, batch_experts=False):
    import transformers
    from reap import glm53_exl3_observe as observer
    from reap.glm53_exl3_observe import _KDA_IDENTITY, _expert_identity
    identity = _expert_identity()
    if identity["backend"] != "native" or not _KDA_IDENTITY["module"].startswith("fla.ops.kda"):
        raise ValueError("resident pipeline requires native EXL3 and the selected FLA kernel")
    stage_path = Path(__file__).with_name("glm53_observation_stage.py")
    model_module = importlib.import_module(observer.Glm5NextTextModel.__module__)
    paths = {"pipeline": Path(__file__), "stage": stage_path, "observer": Path(observer.__file__),
             "config": model_root / "config.json", "transformers_model": Path(model_module.__file__)}
    if batch_experts:
        paths["batched_helper"] = Path(__file__).with_name("glm53_observation_batched.py")
    components = runtime_code_identity(paths, transformers.__version__, stage_sources, batch_experts=batch_experts)
    code = digest(components)
    return runtime_identity({"expert_implementation": identity, "kda_implementation": _KDA_IDENTITY,
                             "prefill_chunk_size": chunk_size,
                             "prefill_policy": f"resident-two-stage-layer-chunks-b{group_size}-experts-{'full' if batch_experts else 'chunked'}-{code}"})


def _upload_receipt(path, args, plan, sid):
    remote = f"{args.sidecar_prefix.strip('/')}/{plan['run_id']}/shards/{sid:05d}.json"
    binding = {"receipt_sha256": file_sha256(path), "repo_id": args.sidecar_repo,
               "path_in_repo": remote}
    marker_path = path.parent / "upload_verified.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        if any(marker.get(k) != v for k, v in binding.items()) or not marker.get("commit"):
            raise ValueError("upload marker does not bind current receipt and destination")
    else:
        commit = upload_verified(path, args.sidecar_repo, remote)
        if not isinstance(commit, str) or not commit:
            raise ValueError("upload returned no verified commit")
        atomic_json(marker_path, {**binding, "commit": commit})


def _receipt_path(root, sid):
    directory = root / f"shard-{sid:05d}"
    path = directory / "receipt.json"
    if directory.is_symlink() or path.is_symlink() or (directory / "upload_verified.json").is_symlink():
        raise ValueError("refusing symlink aggregate output")
    return path


def finalize_lane(plan, root, committed, args, *, upload=upload_verified):
    """One lane is complete only after its merged artifact is verified remotely."""
    result = {"state": "QUALIFICATION_COMPLETE", "run_id": plan["run_id"],
              "completed_shards": sorted(committed), "total_shards": len(plan["shards"]),
              "full_lane_pass": False, "full_suite_pass": False, "activation_disk_writes": False}
    wanted = {shard["shard_id"] for shard in plan["shards"]}
    if (len(committed) != len(wanted) or set(committed) != wanted
            or args.no_upload or args.limit_shards is not None):
        return result
    full = merge_shards(plan, [read_json(_receipt_path(root, sid)) for sid in sorted(committed)])
    path = root / "merged-observations.json"
    marker_path = root / "merged-upload-verified.json"
    if path.is_symlink() or marker_path.is_symlink():
        raise ValueError("refusing symlink merged aggregate output")
    atomic_json(path, full)
    remote = f"{args.sidecar_prefix.strip('/')}/{plan['run_id']}/merged-observations.json"
    binding = {"artifact_sha256": file_sha256(path), "repo_id": args.sidecar_repo,
               "path_in_repo": remote}
    if marker_path.exists():
        marker = read_json(marker_path)
        if any(marker.get(key) != value for key, value in binding.items()) or not marker.get("commit"):
            raise ValueError("merged upload marker does not bind artifact and destination")
    else:
        # upload_verified uploads to a private dataset and independently downloads
        # the exact commit, checking its SHA before returning a commit identifier.
        commit = upload(path, args.sidecar_repo, remote)
        if not isinstance(commit, str) or not commit:
            raise ValueError("merged upload returned no verified commit")
        atomic_json(marker_path, {**binding, "commit": commit})
    result.update(state="FULL_LANE_COMPLETE", full_lane_pass=True)
    return result


def run_pipeline(args):
    if int(os.environ.get("WORLD_SIZE", "0")) != 2 or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 1:
        raise ValueError("use torchrun with exactly two nodes and one process per node")
    rank = int(os.environ.get("RANK", "-1"))
    if rank not in (0, 1):
        raise ValueError("invalid pipeline rank")
    if (args.group_size < 1 or args.prefill_chunk_size < 64 or args.prefill_chunk_size % 64
            or args.timeout_seconds <= 0 or (args.limit_shards is not None and args.limit_shards < 1)):
        raise ValueError("invalid group, prefill chunk, or qualification shard limit")
    if not args.no_upload and (not args.sidecar_repo or not args.sidecar_prefix.strip("/")
                              or ".." in Path(args.sidecar_prefix).parts):
        raise ValueError("explicit private sidecar destination required")
    plan = read_json(args.plan)
    shards = verify_inputs(plan, args.token_manifest, args.model_root)
    if token_identity_field(read_json(args.token_manifest), "corpus") != args.corpus:
        raise ValueError("corpus label differs from sealed token manifest")
    if args.smoke_samples is not None:
        select_smoke_shard(plan, [], args.smoke_samples, args.no_upload)
    stage_source = verify_stage_source(plan, args.model_root, rank)
    # Credentials belong only on rank zero. Fail before model/GPU initialization.
    preflight_error = None
    if rank == 0 and not args.no_upload:
        try:
            preflight_sidecar(args.sidecar_repo)
        except Exception as exc:
            # Both ranks receive a small failure status before touching CUDA.
            preflight_error = type(exc).__name__
    if os.environ.get("GLM53_OBSERVER_EXPERTS", "native") != "native":
        raise ValueError("pipeline cannot use portable expert backend")
    if os.environ.get("GLM53_OBSERVER_KDA", "fla") != "fla":
        raise ValueError("pipeline requires FLA")
    os.environ["GLM53_OBSERVER_EXPERTS"] = "native"
    os.environ["GLM53_OBSERVER_KDA"] = "fla"

    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file
    from reap.glm53_observation_stage import load_stage

    timeout = timedelta(seconds=args.timeout_seconds)
    # Gloo handles small identity/statistics objects; only NCCL carries activations.
    dist.init_process_group("gloo", timeout=timeout)
    lock = None
    try:
        failures = [preflight_error]
        dist.broadcast_object_list(failures, src=0)
        if failures[0] is not None:
            raise RuntimeError(f"private sidecar preflight failed: {failures[0]}")
        stage_sources = [None, None]
        dist.all_gather_object(stage_sources, stage_source)
        runtime = _runtime(args.group_size, args.prefill_chunk_size, args.model_root, stage_sources,
                           batch_experts=args.batch_experts)
        identity = {"plan": digest(plan), "token_manifest": file_sha256(args.token_manifest),
                    "model_index": file_sha256(args.model_root / "model.safetensors.index.json"),
                    "runtime": runtime, "no_upload": args.no_upload,
                    "limit_shards": args.limit_shards, "smoke_samples": args.smoke_samples}
        peers = [None, None]
        dist.all_gather_object(peers, identity)
        validate_peer_identities(peers)
        root = args.output_root.resolve() / plan["run_id"]
        if root.is_symlink():
            raise ValueError("refusing symlink output root")
        committed = []
        if rank == 0:
            root.mkdir(parents=True, exist_ok=True)
            lock = (root / "pipeline.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for shard in plan["shards"]:
                sid = shard["shard_id"]
                path = _receipt_path(root, sid)
                if path.is_file():
                    validate_receipt(plan, sid, read_json(path), runtime)
                    if not args.no_upload:
                        _upload_receipt(path, args, plan, sid)
                    committed.append(sid)
        state = [committed]
        dist.broadcast_object_list(state, src=0)
        committed = state[0]
        remaining = [shard for shard in plan["shards"] if shard["shard_id"] not in committed]
        if args.smoke_samples is not None:
            remaining = [select_smoke_shard(plan, committed, args.smoke_samples, args.no_upload)]
        elif args.limit_shards is not None:
            remaining = remaining[:args.limit_shards]
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        data_group = dist.new_group(ranks=[0, 1], backend="nccl", timeout=timeout)
        stage = load_stage(args.model_root, rank, prefill_chunk_size=args.prefill_chunk_size,
                           batch_experts=args.batch_experts) if remaining else None
        if stage is not None and tuple(stage.owned_layers) != STAGE_LAYERS[rank]:
            raise ValueError("loader returned unexpected stage ownership")
        if stage is not None and stage.batch_experts is not args.batch_experts:
            raise ValueError("loader expert batching differs from sealed runtime mode")
        if stage is not None:
            print(json.dumps({"event": "resident_stage_ready", "rank": rank,
                              "trunk_bytes": stage.trunk_bytes, "cache_bytes": stage.cache_bytes,
                              "batch_experts": stage.batch_experts}), flush=True)
        for shard in remaining:
            sid = shard["shard_id"]
            source = shards[sid]
            if file_sha256(source["resolved_path"]) != source["sha256"]:
                raise ValueError("individual token shard sha256 mismatch")
            ids_cpu = load_file(str(source["resolved_path"]), device="cpu")["input_ids"]
            validate_token_geometry(ids_cpu.shape, ids_cpu.dtype, shard["end"] - shard["start"])
            observed = shard
            if args.smoke_samples is not None:
                ids_cpu = ids_cpu[:args.smoke_samples]
                observed = {**shard, "end": shard["start"] + args.smoke_samples}
            stage.reset_observations()
            with torch.inference_mode():
                for start in range(0, len(ids_cpu), args.group_size):
                    group_started = time.monotonic()
                    expected_ids = ids_cpu[start:start + args.group_size].to(device)
                    batch = len(expected_ids)
                    if rank == 0:
                        hidden = stage.forward(expected_ids)
                        if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (batch, SEQUENCE_LENGTH, 4, 4096):
                            raise ValueError("stage zero output violates hidden transport contract")
                        dist.send(expected_ids.contiguous(), dst=1, group=data_group)
                        dist.send(hidden.contiguous(), dst=1, group=data_group)
                    else:
                        ids = torch.empty_like(expected_ids)
                        hidden = torch.empty((batch, SEQUENCE_LENGTH, 4, 4096), dtype=torch.bfloat16, device=device)
                        dist.recv(ids, src=0, group=data_group)
                        if not torch.equal(ids, expected_ids):
                            raise ValueError("received sample ids differ from sealed shard")
                        dist.recv(hidden, src=0, group=data_group)
                        result = stage.forward(ids, hidden=hidden)
                        del result, ids
                    del hidden, expected_ids
                    print(json.dumps({"event": "pipeline_group_complete", "rank": rank,
                                      "shard_id": sid, "start": shard["start"] + start,
                                      "end": shard["start"] + start + batch,
                                      "wall_seconds_including_transport": time.monotonic() - group_started}), flush=True)
            gathered = [None, None]
            dist.all_gather_object(gathered, _export_stage(stage, observed, rank, runtime))
            if args.smoke_samples is not None:
                result = make_qualification(plan, sid, args.smoke_samples, gathered, runtime)
                if rank == 0:
                    write_qualification(root, result)
                    print(json.dumps({key: value for key, value in result.items() if key != "observations"}), flush=True)
                dist.barrier()
                # No receipt, upload marker, committed range or production status
                # is written or advanced by a partial-shard qualification.
                return result
            observations = merge_stage_observations(plan, sid, gathered, runtime)
            if rank == 0:
                receipt = make_shard_receipt(plan, sid, observations, runtime=runtime)
                path = _receipt_path(root, sid)
                # Atomic receipt publication is the commit. Restart replays only
                # shards without a validated receipt, and never adds old stats.
                atomic_json(path, receipt)
                if not args.no_upload:
                    _upload_receipt(path, args, plan, sid)
                print(json.dumps({"event": "pipeline_shard_sealed", "shard_id": sid,
                                  "samples": shard["end"] - shard["start"],
                                  "uploaded": not args.no_upload}), flush=True)
            dist.barrier()
            committed.append(sid)
        result = {"state": "QUALIFICATION_COMPLETE", "run_id": plan["run_id"],
                  "completed_shards": sorted(committed), "total_shards": len(plan["shards"]),
                  "full_lane_pass": False, "full_suite_pass": False, "activation_disk_writes": False}
        if rank == 0:
            result = finalize_lane(plan, root, committed, args)
            atomic_json(root / "pipeline-status.json", result)
            print(json.dumps(result), flush=True)
        return result
    finally:
        if lock is not None:
            lock.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--corpus", choices=("ours", "wikipedia"), required=True)
    parser.add_argument("--sidecar-repo", default="")
    parser.add_argument("--sidecar-prefix", default="")
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--batch-experts", action="store_true", help="opt in to full-sample expert batches; attention stays chunked")
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--smoke-samples", type=int, help="local-only first-shard sample count; requires --no-upload")
    parser.add_argument("--no-upload", action="store_true", help="explicit local qualification only")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    run_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()
