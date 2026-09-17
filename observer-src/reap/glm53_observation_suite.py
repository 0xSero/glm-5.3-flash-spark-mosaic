"""Pure-stdlib full REAP campaign contracts; this module does not launch jobs.

Ranges are zero-based, half-open global corpus row indices. Each shard receipt
contains the raw observation dictionaries under ``observations`` (layer keys
are strings). Token text and token arrays never belong in these receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

SAMPLES = 24_000
SEQUENCE_LENGTH = 16_384
LAYERS = tuple(range(3, 45))
EXPERTS = 288
TOP_K = 8
SUM_FIELDS = ("ean_sum", "weighted_ean_sum", "weighted_ean_model_scaled_sum",
              "weighted_expert_frequency_sum")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def runtime_identity(run: dict[str, Any]) -> dict[str, Any]:
    """Extract a path-free runtime contract from the observer's sealed identity."""
    fields = ("kda_implementation", "expert_implementation", "prefill_policy", "prefill_chunk_size")
    if not isinstance(run, dict) or any(field not in run for field in fields):
        raise ValueError("missing sealed runtime identity")
    result = {field: run[field] for field in fields}
    if (not isinstance(result["prefill_policy"], str) or not result["prefill_policy"]
            or type(result["prefill_chunk_size"]) is not int or result["prefill_chunk_size"] <= 0):
        raise ValueError("invalid runtime prefill policy")
    allowed = {"schema", "backend", "module", "function", "package_version", "package_versions",
               "wrapper_module", "extension_module", "adapter_sha256", "reference_sha256",
               "wrapper_sha256", "extension_sha256", "sha256"}
    for field in fields[:2]:
        value = result[field]
        if not isinstance(value, dict) or not value or set(value) - allowed:
            raise ValueError("invalid runtime implementation fields")
        for key, item in value.items():
            if key == "package_versions":
                if not isinstance(item, dict) or any(not isinstance(k, str) for k in item):
                    raise ValueError("invalid runtime package versions")
                values = [*item.keys(), *item.values()]
            else:
                values = [item]
            if any(v is not None and (not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_.+<>:-]+", v)
                                      or re.search(r"\d+\.\d+\.\d+\.\d+", v)) for v in values):
                raise ValueError("runtime identity contains unsafe metadata")
            if key.endswith("sha256") and (not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item)):
                raise ValueError("invalid runtime sha256")
    if (not all(result["kda_implementation"].get(k) for k in ("module", "function"))
            or "package_version" not in result["kda_implementation"]):
        raise ValueError("missing runtime KDA implementation")
    if not all(result["expert_implementation"].get(k) for k in ("backend", "adapter_sha256")):
        raise ValueError("missing runtime expert implementation")
    expert = result["expert_implementation"]
    if expert["backend"] not in ("portable", "native"):
        raise ValueError("unknown runtime expert backend")
    if expert["backend"] == "native" and not all(expert.get(k) for k in (
            "wrapper_module", "wrapper_sha256", "extension_module", "extension_sha256",
            "reference_sha256", "package_versions")):
        raise ValueError("missing native runtime identity")
    # Validate policy separately as it too is public metadata.
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", result["prefill_policy"]):
        raise ValueError("unsafe runtime prefill policy")
    return json.loads(json.dumps(result, allow_nan=False))


def _identity(value: dict[str, Any], label: str) -> dict[str, Any]:
    # sha256 binds the exact model index/manifest or corpus token manifest.
    if not isinstance(value, dict) or not value.get("name") or not value.get("revision"):
        raise ValueError(f"{label} identity needs name, revision and sha256")
    if not isinstance(value.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
        raise ValueError(f"{label} identity needs a lowercase sha256")
    return json.loads(json.dumps(value, allow_nan=False))


def build_plan(model_identity: dict[str, Any], corpus_identity: dict[str, Any],
               *, shard_size: int = 64, workers: int = 2) -> dict[str, Any]:
    if shard_size not in (64, 128) or type(shard_size) is not int:
        raise ValueError("shard_size must be 64 or 128")
    if workers not in (2, 4) or type(workers) is not int:
        raise ValueError("workers must be 2 or 4")
    contract = {
        "schema": "glm53-full-reap-plan-v1", "sequence_count": SAMPLES,
        "sequence_length": SEQUENCE_LENGTH, "tokens": SAMPLES * SEQUENCE_LENGTH,
        "layers": list(LAYERS), "experts": EXPERTS, "top_k": TOP_K,
        "model_identity": _identity(model_identity, "model"),
        "corpus_identity": _identity(corpus_identity, "corpus"),
        "shard_size": shard_size, "workers": workers,
    }
    contract["run_id"] = digest(contract)
    contract["shards"] = [
        {"shard_id": i, "start": start, "end": min(start + shard_size, SAMPLES),
         "worker": i % workers}
        for i, start in enumerate(range(0, SAMPLES, shard_size))
    ]
    return contract


def validate_plan(plan: dict[str, Any]) -> None:
    expected = build_plan(plan.get("model_identity"), plan.get("corpus_identity"),
                          shard_size=plan.get("shard_size"), workers=plan.get("workers"))
    if plan != expected:
        raise ValueError("plan differs from its exact full-suite contract")


def _vector(payload: dict[str, Any], field: str, *, integer: bool = False) -> list:
    values = payload.get(field)
    if not isinstance(values, list) or len(values) != EXPERTS:
        raise ValueError(f"{field} must have {EXPERTS} entries")
    for value in values:
        if (type(value) not in ((int,) if integer else (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"invalid nonnegative {'integer ' if integer else ''}{field}")
    return values


def _validated_observation(payload: dict[str, Any], tokens: int) -> dict[str, list]:
    if type(payload.get("total_tokens")) is not int or payload["total_tokens"] != tokens:
        raise ValueError("layer token count differs from shard token count")
    vectors = {field: _vector(payload, field) for field in SUM_FIELDS}
    vectors["expert_frequency"] = _vector(payload, "expert_frequency", integer=True)
    if sum(vectors["expert_frequency"]) != tokens * TOP_K:
        raise ValueError("layer route count differs from tokens times top_k")
    if any(count > tokens for count in vectors["expert_frequency"]):
        raise ValueError("expert frequency exceeds shard tokens")
    vectors["max_activations"] = _vector(payload, "max_activations")
    if "activation_norm_max" in payload and _vector(payload, "activation_norm_max") != vectors["max_activations"]:
        raise ValueError("activation maximum aliases disagree")
    for expert, count in enumerate(vectors["expert_frequency"]):
        if not count and any(vectors[field][expert] for field in (*SUM_FIELDS, "max_activations")):
            raise ValueError("unobserved expert has nonzero statistics")
    return vectors


def make_shard_receipt(plan: dict[str, Any], shard_id: int,
                       observations: dict[str, Any], *, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    validate_plan(plan)
    if type(shard_id) is not int or not 0 <= shard_id < len(plan["shards"]):
        raise ValueError("invalid shard_id")
    shard = plan["shards"][shard_id]
    tokens = (shard["end"] - shard["start"]) * SEQUENCE_LENGTH
    if set(observations) != {str(layer) for layer in LAYERS}:
        raise ValueError("shard must contain all 42 routed layers")
    for payload in observations.values():
        _validated_observation(payload, tokens)
    receipt = {"schema": "glm53-full-reap-shard-v1", "state": "COMPLETE",
               "run_id": plan["run_id"], **shard, "tokens": tokens,
               "observations": observations}
    if runtime is not None:
        receipt["runtime_identity"] = runtime_identity(runtime)
    receipt["sha256"] = digest(receipt)
    return receipt


def merge_shards(plan: dict[str, Any], receipts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    validate_plan(plan)
    seen: set[int] = set()
    totals: dict[str, dict[str, list]] = {}
    receipt_hashes = {}
    selected_runtime = None
    for receipt in receipts:
        shard_id = receipt.get("shard_id")
        if type(shard_id) is not int or shard_id in seen:
            raise ValueError("invalid or duplicate shard range")
        runtime = receipt.get("runtime_identity")
        canonical = make_shard_receipt(plan, shard_id, receipt.get("observations", {}), runtime=runtime)
        if receipt != canonical:
            raise ValueError("shard identity, range, state or digest mismatch")
        if seen and runtime != selected_runtime:
            raise ValueError("mixed runtime identities cannot be merged")
        selected_runtime = runtime
        seen.add(shard_id)
        receipt_hashes[str(shard_id)] = receipt["sha256"]
        for layer, payload in receipt["observations"].items():
            vectors = _validated_observation(payload, receipt["tokens"])
            if layer not in totals:
                totals[layer] = {field: [0] * EXPERTS for field in vectors}
            for field, values in vectors.items():
                current = totals[layer][field]
                totals[layer][field] = ([max(a, b) for a, b in zip(current, values)]
                                        if field == "max_activations"
                                        else [a + b for a, b in zip(current, values)])
    if seen != set(range(len(plan["shards"]))):
        raise ValueError("incomplete shard coverage; cannot seal full observations")
    for payload in totals.values():
        counts = [max(1, value) for value in payload["expert_frequency"]]
        for mean, raw in (("activation_norm_mean", "ean_sum"),
                          ("reap_score", "weighted_ean_sum"),
                          ("reap_score_model_scaled", "weighted_ean_model_scaled_sum"),
                          ("router_weight_mean", "weighted_expert_frequency_sum")):
            payload[mean] = [value / count for value, count in zip(payload[raw], counts)]
        payload["activation_norm_max"] = list(payload["max_activations"])
        payload["total_tokens"] = plan["tokens"]
        _validated_observation(payload, plan["tokens"])
    return {"schema": "glm53-full-reap-merged-v1", "state": "COMPLETE",
            "run_id": plan["run_id"], "sequence_count": SAMPLES,
            "sequence_length": SEQUENCE_LENGTH, "tokens": plan["tokens"],
            "layers": list(LAYERS), "model_identity": plan["model_identity"],
            "corpus_identity": plan["corpus_identity"],
            **({"runtime_identity": selected_runtime} if selected_runtime is not None else {}),
            "shard_receipt_sha256": receipt_hashes, "observations": totals}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--model-identity", type=Path, required=True)
    plan_parser.add_argument("--corpus-identity", type=Path, required=True)
    plan_parser.add_argument("--shard-size", type=int, default=64)
    plan_parser.add_argument("--workers", type=int, default=2)
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--plan", type=Path, required=True)
    merge_parser.add_argument("receipts", type=Path, nargs="+")
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    if args.command == "plan":
        result = build_plan(read(args.model_identity), read(args.corpus_identity),
                            shard_size=args.shard_size, workers=args.workers)
    else:
        result = merge_shards(read(args.plan), (read(path) for path in args.receipts))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
