"""F216 COMPLETION DERIVATIVE of glm53_record_observation_pipeline.py (derived 2026-09-15).

Original: "Local-only original-record observations, distinct from the fixed 24K suite.
One record per forward, no packing or padding. Each layer's cache lives only
for that record/layer; this does not retain all 45 layer caches simultaneously."

DERIVATION (receipted): this file is byte-identical to the original except
(1) this docstring, (2) run(): the phase list is the CONTIGUOUS TAIL of missing shards
(shard_id >= --completion-start-shard, default 332 = the cuda/science records absent from the
sealed partial) instead of phase_plan(shards, committed, ...) over the full manifest, and
(3) main(): one added --completion-start-shard argument. ALL observation math, validation,
contract/run_id computation (deterministic from the SAME original manifest + model identity),
receipt format, collectives, resume scan, and pause handling are UNCHANGED. The run-root
resume scan is retained: a crash mid-completion resumes at the next uncommitted tail shard.
Deviations are recorded in the runtime prefill_policy digest via this file's own sha256.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path
import re
import time

from reap import glm53_observation_pipeline as base
from reap.glm53_observation_sidecar import safe_path
from reap.glm53_observation_suite import LAYERS, _validated_observation, digest, runtime_identity
from reap.glm53_observation_worker import OBSERVATION_FIELDS, atomic_json, file_sha256, read_json
from reap.glm53_prepare_record_tokens import record_identity, SOURCE_SHA256, TOKENIZER_SHA256


def validate_manifest(manifest):
    if (manifest.get("schema") != "glm53-record-token-manifest-v1" or manifest.get("state") != "COMPLETE"
            or manifest.get("sequence_length") != 16384):
        raise ValueError("not a complete original-record token manifest")
    if (manifest.get("corpus") != "ours"
            or manifest.get("identity") != record_identity(SOURCE_SHA256, TOKENIZER_SHA256)):
        raise ValueError("missing pinned record source/tokenizer identity")
    cursor, tokens = 0, 0
    shards = manifest.get("shards", [])
    if not shards:
        raise ValueError("empty record manifest")
    for sid, shard in enumerate(shards):
        lengths = shard.get("lengths", [])
        if (not isinstance(lengths, list) or not 1 <= len(lengths) <= 64
                or any(type(n) is not int or not 1 <= n <= 16384 for n in lengths)
                or (sid < len(shards) - 1 and len(lengths) != 64)
                or shard.get("shard_id") != sid or shard.get("start") != cursor
                or shard.get("end") != cursor + len(lengths)
                or shard.get("sequence_count") != len(lengths) or shard.get("tokens") != sum(lengths)):
            raise ValueError("record shard ranges/counts/lengths mismatch")
        name = shard.get("path")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", name)
                or not re.fullmatch(r"[0-9a-f]{64}", shard.get("sha256", ""))):
            raise ValueError("unsafe record shard path or checksum")
        cursor += len(lengths)
        tokens += sum(lengths)
    if manifest.get("sequence_count") != cursor or manifest.get("tokens") != tokens:
        raise ValueError("manifest total records/tokens mismatch")
    return shards


def load_records(manifest_path, shard):
    path = safe_path(manifest_path.parent / shard["path"])
    if file_sha256(path) != shard["sha256"]:
        raise ValueError("record token shard checksum mismatch")
    value = read_json(path)
    records = value.get("records", [])
    if value.get("schema") != "glm53-record-token-shard-v1" or len(records) != shard["sequence_count"]:
        raise ValueError("record shard schema/count mismatch")
    if (shard.get("source_rows") != [row.get("source_row") for row in records]
            or shard.get("record_ids") != [row.get("id") for row in records]):
        raise ValueError("record IDs/source rows differ from manifest")
    last_row = -1
    for record, length in zip(records, shard["lengths"]):
        ids = record.get("input_ids", [])
        original = record.get("original_tokens")
        source_row = record.get("source_row")
        if (len(ids) != length or any(type(n) is not int or n < 0 for n in ids)
                or type(original) is not int or original < length or min(original, 16384) != length
                or type(record.get("truncated")) is not bool or record["truncated"] != (original > 16384)
                or type(source_row) is not int or source_row <= last_row):
            raise ValueError("invalid original-record token geometry or order")
        last_row = source_row
    return records


def record_contract(model, manifest, manifest_sha):
    if (not isinstance(model, dict) or set(model) != {"name", "revision", "sha256"}
            or any(not isinstance(model[k], str) or not model[k] for k in model)
            or not re.fullmatch(r"[0-9a-f]{64}", model["sha256"])):
        raise ValueError("explicit immutable model identity required")
    validate_manifest(manifest)
    body = {"schema": "glm53-record-observation-contract-v1", "model_identity": model,
            "token_manifest_sha256": manifest_sha, "sequence_count": manifest["sequence_count"],
            "tokens": manifest["tokens"], "policy": "original-records-truncate-16384-no-padding-no-packing"}
    return {**body, "run_id": "records-" + digest(body)[:24]}


def make_receipt(contract, shard, observations, runtime, *, smoke=False):
    runtime_identity(runtime)
    if set(observations) != {str(layer) for layer in LAYERS}:
        raise ValueError("record receipt must cover exactly 42 routed layers")
    for row in observations.values():
        if set(row) - OBSERVATION_FIELDS:
            raise ValueError("non-aggregate observation fields")
        _validated_observation(row, shard["tokens"])
    body = {"schema": "glm53-record-observation-receipt-v1",
            "state": "RECORD_SMOKE_COMPLETE" if smoke else "RECORD_SHARD_COMPLETE",
            "run_id": contract["run_id"], "contract_sha256": digest(contract),
            "shard_id": shard["shard_id"], "start": shard["start"], "end": shard["end"],
            "sequence_count": shard["sequence_count"], "tokens": shard["tokens"],
            "lengths_sha256": digest(shard["lengths"]), "token_shard_sha256": shard["sha256"],
            "runtime_identity": runtime, "observations": observations,
            "uploaded": False, "full_suite_pass": False}
    return {**body, "sha256": digest(body)}


def validate_receipt(contract, shard, receipt, runtime):
    if receipt != make_receipt(contract, shard, receipt.get("observations", {}), runtime):
        raise ValueError("record receipt contract/runtime/count/digest mismatch")


def forward_record(stage, ids, hidden=None, *, cache_factory=None):
    import torch
    if cache_factory is None:
        from transformers.cache_utils import DynamicCache
        cache_factory = DynamicCache
    if ids.dtype != torch.int64 or ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 16384:
        raise ValueError("original records require int64 [1,1..16384] ids")
    if hidden is None:
        if stage.stage_id != 0:
            raise ValueError("stage one requires incoming hidden")
        hidden = stage.model.embed_tokens(ids).unsqueeze(2).expand(-1, -1, 4, -1)
    if tuple(hidden.shape) != (*ids.shape, 4, 4096) or hidden.dtype != torch.bfloat16:
        raise ValueError("record hidden transport geometry/dtype mismatch")
    for layer_index in stage.owned_layers:
        cache = cache_factory(config=stage.model.config)
        parts = []
        for start in range(0, ids.shape[1], stage.prefill_chunk_size):
            end = min(ids.shape[1], start + stage.prefill_chunk_size)
            local_ids = ids[:, start:end]
            result, topk = stage.model.layers[layer_index](
                hidden[:, start:end], attention_mask=torch.ones_like(local_ids, dtype=torch.bool),
                position_ids=torch.arange(start, end, device=ids.device).unsqueeze(0),
                past_key_values=cache, use_cache=True, prev_topk_indices=None, input_ids=local_ids)
            if topk is not None or result.shape != hidden[:, start:end].shape or result.dtype != torch.bfloat16:
                raise ValueError("record layer changed hidden geometry/dtype or shared topk")
            parts.append(result)
        hidden = torch.cat(parts, dim=1)
        del cache, parts
    return hidden


def phase_plan(shards, committed, smoke_records=None, qualify_then_collect=False):
    if smoke_records not in (None, 1) or (smoke_records and qualify_then_collect):
        raise ValueError("standalone smoke and qualify-then-collect are mutually exclusive")
    production = [(shard, False) for shard in shards if shard["shard_id"] not in committed]
    if smoke_records:
        return [(shards[0], True)]
    return ([(shards[0], True)] if qualify_then_collect and production else []) + production


def observation_phases(stage, phases):
    """Never carry pilot or previous-shard statistics into the next phase."""
    for shard, smoke in phases:
        stage.reset_observations()
        yield shard, smoke


def run(args):
    rank = int(os.environ.get("RANK", "-1"))
    if rank not in (0, 1) or os.environ.get("WORLD_SIZE") != "2" or os.environ.get("LOCAL_WORLD_SIZE") != "1":
        raise ValueError("requires two nodes with one rank each")
    if (args.prefill_chunk_size < 64 or args.prefill_chunk_size % 64 or args.timeout_seconds <= 0
            or args.smoke_records not in (None, 1)
            or (args.smoke_records and args.qualify_then_collect)):
        raise ValueError("invalid chunk, timeout or smoke contract")
    args.token_manifest = safe_path(args.token_manifest)
    manifest = read_json(args.token_manifest)
    shards = validate_manifest(manifest)
    contract = record_contract(read_json(safe_path(args.model_identity)), manifest, file_sha256(args.token_manifest))
    # Validate all tokens before GPU initialization; neither peer trusts metadata alone.
    last_row = -1
    for shard in shards:
        rows = load_records(args.token_manifest, shard)
        if rows[0]["source_row"] <= last_row:
            raise ValueError("source rows repeat or reorder across shards")
        last_row = rows[-1]["source_row"]
    if file_sha256(args.model_root / "tokenizer.json") != manifest["identity"]["tokenizer_sha256"]:
        raise ValueError("record tokenizer differs from staged model tokenizer")
    source = base.verify_stage_source(contract, args.model_root, rank)
    for key, wanted in (("GLM53_OBSERVER_EXPERTS", "native"), ("GLM53_OBSERVER_KDA", "fla")):
        if os.environ.get(key, wanted) != wanted:
            raise ValueError("record observer requires qualified native/FLA runtime")
        os.environ[key] = wanted
    import torch
    import torch.distributed as dist
    from reap.glm53_observation_stage import load_stage
    timeout = timedelta(seconds=args.timeout_seconds)
    dist.init_process_group("gloo", timeout=timeout)
    lock = None
    try:
        sources = [None, None]
        dist.all_gather_object(sources, source)
        runtime = base._runtime(1, args.prefill_chunk_size, args.model_root, sources)
        from reap.glm53_observation_kernels import selected_kda_identity
        from transformers.models.glm5_next.modeling_glm5_next import recurrent_kimi_delta_attention
        recurrent = selected_kda_identity(recurrent_kimi_delta_attention, "fla")
        runtime["prefill_policy"] = "original-records-" + digest({"base": runtime, "recurrent": recurrent,
                                                                "runner_sha256": file_sha256(Path(__file__))})
        peers = [None, None]
        dist.all_gather_object(peers, {"contract": contract, "runtime": runtime, "smoke": args.smoke_records,
                                      "qualify_then_collect": args.qualify_then_collect})
        base.validate_peer_identities(peers)
        root = safe_path(args.output_root) / contract["run_id"]
        safe_path(root)
        committed = []
        if rank == 0:
            root.mkdir(parents=True, exist_ok=True)
            safe_path(root / "pipeline.lock")
            lock = (root / "pipeline.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            allowed = {f"shard-{s['shard_id']:05d}" for s in shards}
            if any(p.name.startswith("shard-") and p.name not in allowed for p in root.iterdir()):
                raise ValueError("foreign record receipt range in output root")
            for shard in shards:
                path = safe_path(base._receipt_path(root, shard["shard_id"]))
                if path.exists():
                    validate_receipt(contract, shard, read_json(path), runtime)
                    committed.append(shard["shard_id"])
        state = [committed]
        dist.broadcast_object_list(state, src=0)
        committed = state[0]
        # F216 COMPLETION DERIVATION: contiguous tail of missing shards only.
        start = args.completion_start_shard
        missing = [s for s in shards if s["shard_id"] >= start]
        if not missing:
            raise ValueError(f"no shards at or beyond completion start {start}")
        if args.smoke_records not in (None, 1) or (args.smoke_records and args.qualify_then_collect):
            raise ValueError("standalone smoke and qualify-then-collect are mutually exclusive")
        if args.smoke_records:
            phases = [(missing[0], True)]
        else:
            phases = ([(missing[0], True)] if args.qualify_then_collect else []) + [(s, False) for s in missing]
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        group = dist.new_group(ranks=[0, 1], backend="nccl", timeout=timeout)
        stage = load_stage(args.model_root, rank, prefill_chunk_size=args.prefill_chunk_size) if phases else None
        paused = False
        for shard, smoke in observation_phases(stage, phases):
            records = load_records(args.token_manifest, shard)
            observed = shard
            if smoke:
                records = records[:1]
                observed = {**shard, "end": shard["start"] + 1, "sequence_count": 1,
                            "lengths": shard["lengths"][:1], "tokens": shard["lengths"][0]}
            for offset, record in enumerate(records):
                started = time.monotonic()
                ids = torch.tensor([record["input_ids"]], dtype=torch.int64, device=device)
                with torch.inference_mode():
                    if rank == 0:
                        hidden = forward_record(stage, ids)
                        dist.send(ids.contiguous(), dst=1, group=group)
                        dist.send(hidden.contiguous(), dst=1, group=group)
                    else:
                        incoming = torch.empty_like(ids)
                        dist.recv(incoming, src=0, group=group)
                        if not torch.equal(ids, incoming):
                            raise ValueError("peer record differs from sealed original tokens")
                        hidden = torch.empty((1, ids.shape[1], 4, 4096), dtype=torch.bfloat16, device=device)
                        dist.recv(hidden, src=0, group=group)
                        output = forward_record(stage, ids, hidden)
                        del output, incoming
                del hidden, ids
                print(json.dumps({"event": "record_complete", "rank": rank, "record": shard["start"] + offset,
                                  "phase": "qualification" if smoke else "production",
                                  "tokens": len(record["input_ids"]), "seconds": time.monotonic() - started}), flush=True)
            gathered = [None, None]
            dist.all_gather_object(gathered, base._export_stage(stage, observed, rank, runtime))
            obs = base._merge_stage_range(observed, gathered, runtime)
            receipt = make_receipt(contract, observed, obs, runtime, smoke=smoke)
            if rank == 0:
                path = root / "record-smoke.json" if smoke else base._receipt_path(root, shard["shard_id"])
                safe_path(path)
                atomic_json(path, receipt)
                print(json.dumps({"event": receipt["state"], "shard_id": shard["shard_id"],
                                  "records": observed["sequence_count"], "tokens": observed["tokens"]}), flush=True)
            dist.barrier()
            if smoke:
                if args.smoke_records:
                    return {"state": "RECORD_SMOKE_COMPLETE", "full_suite_pass": False}
                # Validation and atomic smoke publication succeeded on rank zero.
                # Advancing the phase iterator resets every expert accumulator;
                # production then replays the complete shard from its first row.
                continue
            committed.append(shard["shard_id"])
            stop = [bool(args.pause_file and args.pause_file.exists()) if rank == 0 else False]
            dist.broadcast_object_list(stop, src=0)
            if stop[0]:
                paused = True
                break
        result = {"state": "PAUSED_AT_RECORD_SHARD" if paused else "RECORD_LOCAL_COMPLETE",
                  "run_id": contract["run_id"], "completed_shards": committed,
                  "records": sum(shards[s]["sequence_count"] for s in committed),
                  "tokens": sum(shards[s]["tokens"] for s in committed),
                  "uploaded": False, "full_suite_pass": False}
        if rank == 0:
            safe_path(root / "record-status.json")
            atomic_json(root / "record-status.json", result)
        return result
    finally:
        if lock is not None:
            lock.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-identity", "token-manifest", "model-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--smoke-records", type=int, choices=(1,))
    parser.add_argument("--qualify-then-collect", action="store_true",
                        help="qualify one record, reset statistics, then collect with the same resident model")
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument("--completion-start-shard", type=int, default=332)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
