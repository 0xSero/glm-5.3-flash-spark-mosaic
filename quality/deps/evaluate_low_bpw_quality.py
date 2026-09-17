#!/usr/bin/env python3
"""Fail-closed BF16-vs-selective-EXL3 quality evaluation for GLM-5.3 Flash.

The evaluator streams 32 sealed Wikitext-2 rows through one decoder layer at
a time on the configured conversion GPUs. It never materializes the whole model.
The sealed BF16 reference is reused; the variant pass reconstructs only routed
expert slices from the packed EXL3 tensors and compares exact vocabulary
chunks against the BF16 baseline.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as dt
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import traceback

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from release_route_capture_v2 import load_layer
from glm53_exl3_tp4 import _load_linear, _packed_prefix


CAMPAIGN = Path(os.environ.get("GLM53_EXL3_ROOT", "/opt/glm53-exl3"))
ARTIFACT = Path(os.environ.get("GLM53_EXL3_ARTIFACT", "/models/GLM-5.3-Flash-EXL3-Q4"))
CONFIG_ROOT = Path(os.environ.get("GLM53_CONFIG_ROOT", str(ARTIFACT)))
VARIANT_LABEL = os.environ.get("GLM53_EXL3_VARIANT_LABEL", "q4")
EVAL = Path(os.environ.get("GLM53_EXL3_QUALITY_ROOT", str(CAMPAIGN / f"quality-eval-{VARIANT_LABEL}")))
FIXTURE_EVAL = CAMPAIGN / "quality-eval-v1"
REFERENCE_OUTPUT = CAMPAIGN / "quality-reference-v2"
TOKENS = FIXTURE_EVAL / "token_rows.safetensors"
BF16_NORMALIZED = REFERENCE_OUTPUT / "bf16-normalized"
ROWS = 32
COLUMNS = 2048
LAYERS = 45
ROUTED_FIRST = 3
EXPERTS = 288
TP = 4
SLICE = 512
HIDDEN = 4096
VOCAB_CHUNK = 4096
GPU_IDS = tuple(int(value) for value in os.environ.get("GLM53_EXL3_GPU_IDS", "0,1").split(","))
if not GPU_IDS or len(set(GPU_IDS)) != len(GPU_IDS):
    raise RuntimeError(f"invalid GLM53_EXL3_GPU_IDS: {GPU_IDS}")
EXPECTED_EVAL_MANIFEST = "46255621cd2da8e05546edc3498cfbee518da092af2f44ae243f79da97c0f00b"
ROUTED_SOURCE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


class EvaluationSourceTensors:
    """Load exact retained BF16 tensors and disposable routed placeholders.

    Every routed expert placeholder is overwritten from the packed artifact by
    ``overwrite_variant_experts`` before the layer executes.  This avoids
    retaining a second 643 GB BF16 source copy on the validation host while the
    sealed BF16 normalized reference remains the comparison baseline.
    """

    def __init__(self) -> None:
        index = json.loads((ARTIFACT / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]

    def get(self, key: str) -> torch.Tensor:
        match = ROUTED_SOURCE.fullmatch(key)
        if match:
            projection = match.group(3)
            shape = (4096, 2048) if projection == "down_proj" else (2048, 4096)
            return torch.zeros(shape, dtype=torch.bfloat16)
        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"retained source tensor not indexed: {key}")
        if not shard.startswith("retained/"):
            raise RuntimeError(f"non-retained source resolution refused: {key} -> {shard}")
        with safe_open(ARTIFACT / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def state_path(slot: Path, row: int) -> Path:
    return slot / f"row-{row:03d}.safetensors"


def sidecar_path(slot: Path, row: int) -> Path:
    return slot / f"row-{row:03d}.json"


def write_state(slot: Path, row: int, layer: int, hidden: torch.Tensor, topk: torch.Tensor | None) -> None:
    slot.mkdir(parents=True, exist_ok=True)
    tensors = {"hidden_streams": hidden.detach().to("cpu", torch.bfloat16).contiguous()}
    if topk is not None:
        tensors["prev_topk_indices"] = topk.detach().to("cpu", torch.int32).contiguous()
    target = state_path(slot, row)
    temporary = target.with_name(target.name + ".tmp")
    save_file(tensors, temporary)
    temporary.replace(target)
    atomic_json(sidecar_path(slot, row), {
        "row": row, "layer": layer, "bytes": target.stat().st_size,
        "sha256": sha256(target), "completed_at": now(),
    })


def verified_state(slot: Path, row: int, layer: int) -> bool:
    target, sidecar = state_path(slot, row), sidecar_path(slot, row)
    if not target.is_file() or not sidecar.is_file():
        return False
    try:
        meta = json.loads(sidecar.read_text())
        if meta != {**meta, "row": row, "layer": layer}:
            return False
        if meta["bytes"] != target.stat().st_size or meta["sha256"] != sha256(target):
            return False
        with safe_open(target, framework="pt", device="cpu") as handle:
            return tuple(handle.get_slice("hidden_streams").get_shape()) == (1, COLUMNS, 4, HIDDEN)
    except Exception:
        return False


def initialize(mode_root: Path) -> tuple[Path, int]:
    progress = mode_root / "progress.json"
    if progress.is_file():
        state = json.loads(progress.read_text())
        slot = mode_root / state["current_slot"]
        layer = int(state["layer"])
        if all(verified_state(slot, row, layer) for row in range(ROWS)):
            return slot, layer
        raise RuntimeError(f"progress points to incomplete state: {progress}")
    slot = mode_root / "slot-a"
    if slot.exists():
        shutil.rmtree(slot)
    slot.mkdir(parents=True)
    token_rows = load_file(TOKENS)["input_ids"]
    source = EvaluationSourceTensors()
    embedding = source.get("model.language_model.embed_tokens.weight")
    for row in range(ROWS):
        hidden = embedding[token_rows[row]].unsqueeze(0).unsqueeze(2).expand(-1, -1, 4, -1).contiguous()
        write_state(slot, row, -1, hidden, None)
    atomic_json(progress, {"state": "RUNNING", "layer": -1, "current_slot": "slot-a", "updated_at": now()})
    return slot, -1


def overwrite_variant_experts(layer: torch.nn.Module, layer_idx: int, device: torch.device) -> None:
    gate_up = layer.mlp.experts.gate_up_proj
    down = layer.mlp.experts.down_proj
    intermediate = layer.mlp.experts.intermediate_dim
    if intermediate != TP * SLICE or layer.mlp.experts.num_experts != EXPERTS:
        raise RuntimeError("routed expert geometry differs from sealed contract")
    sidecars = sorted((ARTIFACT / "layers").glob(f"layer-{layer_idx:02d}-part-*.json"))
    if not sidecars:
        raise RuntimeError(f"no packed variant parts for layer {layer_idx}")
    observed_experts = set()
    with torch.inference_mode():
        for sidecar_path in sidecars:
            sidecar = json.loads(sidecar_path.read_text())
            experts = [int(value) for value in sidecar.get("experts", [])]
            if observed_experts.intersection(experts):
                raise RuntimeError(f"duplicate packed expert assignment at layer {layer_idx}")
            observed_experts.update(experts)
            part = sidecar_path.with_suffix(".safetensors")
            with safe_open(part, framework="pt", device=str(device)) as handle:
                for expert in experts:
                    for rank in range(TP):
                        start, end = rank * SLICE, (rank + 1) * SLICE
                        for projection, target in (
                            ("gate_proj", gate_up[expert, :intermediate]),
                            ("up_proj", gate_up[expert, intermediate:]),
                        ):
                            module = _load_linear(handle, _packed_prefix(layer_idx, expert, projection, rank), HIDDEN, SLICE, torch.bfloat16)
                            weight = module.get_weight_tensor()
                            target[start:end].copy_(weight.T.to(dtype=target.dtype))
                            del weight, module
                        module = _load_linear(handle, _packed_prefix(layer_idx, expert, "down_proj", rank), SLICE, HIDDEN, torch.bfloat16)
                        weight = module.get_weight_tensor()
                        down[expert, :, start:end].copy_(weight.T.to(dtype=down.dtype))
                        del weight, module
            gc.collect()
            torch.cuda.empty_cache()
    if observed_experts != set(range(EXPERTS)):
        raise RuntimeError(f"packed parts do not cover all experts at layer {layer_idx}")
    # A whole-tensor isfinite check materializes a boolean tensor as large as
    # the reconstructed expert bank (about 9 GiB for gate_up alone) and can OOM
    # on a 24 GiB GPU even when the reconstructed weights and forward fit. Keep
    # this validation bounded to one expert at a time.
    for expert in range(EXPERTS):
        if not torch.isfinite(gate_up[expert]).all().item():
            raise RuntimeError(
                f"non-finite reconstructed gate/up weights at layer {layer_idx} expert {expert}"
            )
        if not torch.isfinite(down[expert]).all().item():
            raise RuntimeError(
                f"non-finite reconstructed down weights at layer {layer_idx} expert {expert}"
            )


def layer_worker(mode: str, layer_idx: int, gpu: int, rows: list[int], input_s: str, output_s: str) -> dict:
    input_slot, output_slot = Path(input_s), Path(output_s)
    try:
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
        torch.backends.cuda.matmul.allow_tf32 = False
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
        config = Glm5NextConfig.from_pretrained(CONFIG_ROOT, local_files_only=True).text_config
        config._attn_implementation = "sdpa"
        source = EvaluationSourceTensors()
        started = time.monotonic()
        layer, loaded_bytes = load_layer(layer_idx, config, source, device)
        if mode == "variant" and layer_idx >= ROUTED_FIRST:
            overwrite_variant_experts(layer, layer_idx, device)
        attention_mask = torch.ones((1, COLUMNS), dtype=torch.bool, device=device)
        completed = skipped = 0
        with torch.inference_mode():
            for row in rows:
                if verified_state(output_slot, row, layer_idx):
                    skipped += 1
                    continue
                state = load_file(state_path(input_slot, row))
                hidden = state["hidden_streams"].to(device)
                prev = state.get("prev_topk_indices")
                if prev is not None:
                    prev = prev.to(device)
                output, next_topk = layer(hidden, attention_mask=attention_mask, past_key_values=None,
                                          use_cache=False, prev_topk_indices=prev)
                if not torch.isfinite(output).all().item():
                    raise RuntimeError(f"non-finite {mode} output layer={layer_idx} row={row}")
                write_state(output_slot, row, layer_idx, output, next_topk)
                completed += 1
                del state, hidden, prev, output, next_topk
        torch.cuda.synchronize(device)
        return {"state": "COMPLETE", "mode": mode, "layer": layer_idx, "gpu": gpu,
                "rows": rows, "completed": completed, "skipped": skipped,
                "loaded_bytes": loaded_bytes, "elapsed_seconds": time.monotonic() - started}
    except Exception:
        raise RuntimeError(traceback.format_exc())


def run_layers(mode: str, output_root: Path = EVAL) -> Path:
    mode_root = output_root / f"{mode}-stream"
    mode_root.mkdir(parents=True, exist_ok=True)
    input_slot, completed_layer = initialize(mode_root)
    for layer_idx in range(completed_layer + 1, LAYERS):
        output_slot = mode_root / ("slot-b" if input_slot.name == "slot-a" else "slot-a")
        manifest = output_slot / "manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text()).get("layer") != layer_idx:
            shutil.rmtree(output_slot)
        output_slot.mkdir(parents=True, exist_ok=True)
        shards = {
            gpu: list(range(position, ROWS, len(GPU_IDS)))
            for position, gpu in enumerate(GPU_IDS)
        }
        context = mp.get_context("spawn")
        results = []
        with ProcessPoolExecutor(max_workers=len(GPU_IDS), mp_context=context) as pool:
            futures = [pool.submit(layer_worker, mode, layer_idx, gpu, shards[gpu], str(input_slot), str(output_slot)) for gpu in GPU_IDS]
            for future in as_completed(futures):
                results.append(future.result())
        if not all(verified_state(output_slot, row, layer_idx) for row in range(ROWS)):
            raise RuntimeError(f"incomplete {mode} state after layer {layer_idx}")
        atomic_json(manifest, {"state": "COMPLETE", "mode": mode, "layer": layer_idx,
                               "workers": sorted(results, key=lambda x: x["gpu"]), "completed_at": now()})
        atomic_json(mode_root / "progress.json", {"state": "RUNNING", "layer": layer_idx,
                    "current_slot": output_slot.name, "updated_at": now()})
        if input_slot.exists():
            shutil.rmtree(input_slot)
        input_slot = output_slot
    return input_slot


def seal_normalized(mode: str, final_slot: Path, output_root: Path = EVAL) -> Path:
    output = output_root / f"{mode}-normalized"
    output.mkdir(parents=True, exist_ok=True)
    source = EvaluationSourceTensors()
    norm_weight = source.get("model.language_model.norm.weight")
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    eps = Glm5NextConfig.from_pretrained(CONFIG_ROOT, local_files_only=True).text_config.rms_norm_eps
    records = []
    for row in range(ROWS):
        hidden = load_file(state_path(final_slot, row))["hidden_streams"].mean(dim=2).float()
        variance = hidden.square().mean(dim=-1, keepdim=True)
        normalized = (hidden * torch.rsqrt(variance + eps)).to(torch.bfloat16) * norm_weight
        target = output / f"row-{row:03d}.safetensors"
        temporary = target.with_name(target.name + ".tmp")
        save_file({"hidden": normalized.contiguous()}, temporary)
        temporary.replace(target)
        records.append({"row": row, "bytes": target.stat().st_size, "sha256": sha256(target)})
    atomic_json(output / "manifest.json", {"state": "COMPLETE", "mode": mode, "rows": ROWS,
                "records": records, "completed_at": now()})
    return output


def compare(bf16_dir: Path, variant_dir: Path) -> dict:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    source = EvaluationSourceTensors()
    head = source.get("lm_head.weight")
    token_rows = load_file(TOKENS)["input_ids"]
    total_kl = total_ce_bf16 = total_ce_q4 = 0.0
    total_agree = total_tokens = 0
    per_row = []
    for row in range(ROWS):
        p_hidden = load_file(bf16_dir / f"row-{row:03d}.safetensors")["hidden"][0, :-1].to(device)
        q_hidden = load_file(variant_dir / f"row-{row:03d}.safetensors")["hidden"][0, :-1].to(device)
        labels = token_rows[row, 1:].to(device)
        row_kl = row_pnll = row_qnll = 0.0
        p_best = torch.full((COLUMNS - 1,), -float("inf"), device=device)
        q_best = torch.full_like(p_best, -float("inf"))
        p_arg = torch.zeros((COLUMNS - 1,), dtype=torch.long, device=device)
        q_arg = torch.zeros_like(p_arg)
        p_lse = torch.full_like(p_best, -float("inf"), dtype=torch.float32)
        q_lse = torch.full_like(q_best, -float("inf"), dtype=torch.float32)
        target_p = torch.empty_like(p_best, dtype=torch.float32)
        target_q = torch.empty_like(q_best, dtype=torch.float32)
        chunks = []
        for start in range(0, head.shape[0], VOCAB_CHUNK):
            stop = min(start + VOCAB_CHUNK, head.shape[0])
            weight = head[start:stop].to(device)
            p = F.linear(p_hidden, weight).float()
            q = F.linear(q_hidden, weight).float()
            chunks.append((start, p.cpu(), q.cpu()))
            p_max, p_idx = p.max(dim=-1); q_max, q_idx = q.max(dim=-1)
            take = p_max > p_best; p_arg[take] = p_idx[take] + start; p_best[take] = p_max[take]
            take = q_max > q_best; q_arg[take] = q_idx[take] + start; q_best[take] = q_max[take]
            p_lse = torch.logaddexp(p_lse, torch.logsumexp(p, dim=-1))
            q_lse = torch.logaddexp(q_lse, torch.logsumexp(q, dim=-1))
            mask = (labels >= start) & (labels < stop)
            if mask.any():
                local = labels[mask] - start
                target_p[mask] = p[mask, local]
                target_q[mask] = q[mask, local]
            del weight, p, q
        row_pnll = float((p_lse - target_p).sum().item())
        row_qnll = float((q_lse - target_q).sum().item())
        for start, p_cpu, q_cpu in chunks:
            p = p_cpu.to(device); q = q_cpu.to(device)
            logp = p - p_lse[:, None]; logq = q - q_lse[:, None]
            row_kl += float((logp.exp() * (logp - logq)).sum().item())
            del p, q, logp, logq
        agree = int((p_arg == q_arg).sum().item())
        count = COLUMNS - 1
        total_kl += row_kl; total_ce_bf16 += row_pnll; total_ce_q4 += row_qnll
        total_agree += agree; total_tokens += count
        per_row.append({"row": row, "tokens": count, "kl_sum": row_kl,
                        "bf16_nll_sum": row_pnll, "variant_nll_sum": row_qnll, "top1_agree": agree})
        del p_hidden, q_hidden, labels, chunks
        gc.collect(); torch.cuda.empty_cache()
    result = {
        "tokens": total_tokens,
        "kl_bf16_to_variant": total_kl / total_tokens,
        "top1_agreement": total_agree / total_tokens,
        "bf16_cross_entropy": total_ce_bf16 / total_tokens,
        "variant_cross_entropy": total_ce_q4 / total_tokens,
        "bf16_perplexity": math.exp(total_ce_bf16 / total_tokens),
        "variant_perplexity": math.exp(total_ce_q4 / total_tokens),
        "perplexity_delta_fraction": math.exp((total_ce_q4 - total_ce_bf16) / total_tokens) - 1.0,
        "per_row": per_row,
    }
    if not all(math.isfinite(result[key]) for key in ("kl_bf16_to_variant", "bf16_perplexity", "variant_perplexity")):
        raise RuntimeError("non-finite quality metric")
    return result


def preflight() -> None:
    if sha256(FIXTURE_EVAL / "manifest.json") != EXPECTED_EVAL_MANIFEST:
        raise RuntimeError("held-out evaluation manifest digest mismatch")
    assembly = json.loads((ARTIFACT / "assembly-status.json").read_text())
    if assembly.get("state") != "COMPLETE" or assembly.get("structural_pass") is not True:
        raise RuntimeError("assembled variant has not passed structural verification")
    manifest = json.loads((ARTIFACT / "EXL3_MANIFEST.json").read_text())
    if manifest.get("state") != "STRUCTURAL_PASS":
        raise RuntimeError("assembled artifact manifest is not a structural pass")
    retained = json.loads((ARTIFACT / "retained" / "manifest.json").read_text())
    if not (
        retained.get("schema") == "glm53-retained-source-precision-v1"
        and retained.get("retained_tensor_count") == 2482
        and retained.get("retained_tensor_bytes") == 33835039608
    ):
        raise RuntimeError("artifact retained BF16 contract is incomplete")
 

def validate_bf16_reference() -> None:
    reference_manifest = json.loads((BF16_NORMALIZED / "manifest.json").read_text())
    if reference_manifest.get("state") != "COMPLETE" or reference_manifest.get("rows") != ROWS:
        raise RuntimeError("sealed BF16 normalized reference is incomplete")
    for item in reference_manifest.get("records", []):
        path = BF16_NORMALIZED / f"row-{int(item['row']):03d}.safetensors"
        if path.stat().st_size != int(item["bytes"]) or sha256(path) != item["sha256"]:
            raise RuntimeError(f"sealed BF16 normalized row mismatch: {path}")
    if len(reference_manifest.get("records", [])) != ROWS:
        raise RuntimeError("sealed BF16 normalized reference row count mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("bf16", "variant", "compare", "all"), default="all")
    args = parser.parse_args()
    preflight()
    bf16_dir = BF16_NORMALIZED
    variant_dir = EVAL / "variant-normalized"
    if args.phase in ("bf16", "compare", "all"):
        validate_bf16_reference()
    if args.phase in ("variant", "all"):
        variant_dir = seal_normalized("variant", run_layers("variant"))
    if args.phase in ("compare", "all"):
        metrics = compare(bf16_dir, variant_dir)
        gates = {
            "kl_bf16_to_variant_at_most": 0.15,
            "top1_agreement_at_least": 0.80,
            "absolute_perplexity_delta_fraction_at_most": 0.05,
        }
        passed = (
            metrics["kl_bf16_to_variant"] <= gates["kl_bf16_to_variant_at_most"]
            and metrics["top1_agreement"] >= gates["top1_agreement_at_least"]
            and abs(metrics["perplexity_delta_fraction"]) <= gates["absolute_perplexity_delta_fraction_at_most"]
        )
        state = "QUALITY_VALIDATED" if passed else "QUALITY_MEASURED_OUTSIDE_Q4_CONTROL_GATES"
        manifest_path = ARTIFACT / "EXL3_MANIFEST.json"
        report = {"state": state, "schema": "glm53-selective-exl3-low-bpw-quality-v1",
                  "variant_label": VARIANT_LABEL,
                  "created_at": now(), "eval_manifest_sha256": EXPECTED_EVAL_MANIFEST,
                  "artifact_manifest_sha256": sha256(manifest_path),
                  "bf16_reference_manifest_sha256": sha256(BF16_NORMALIZED / "manifest.json"),
                  "variant_base_source": "artifact retained BF16 tensors; routed placeholders fully overwritten from packed EXL3 before forward",
                  "quality_measurement_complete": True,
                  "within_q4_control_gates": passed,
                  "acceptance_gates": gates, **metrics}
        atomic_json(EVAL / "quality-report.json", report)
        atomic_json(EVAL / "quality-status.json", {"state": state,
                    "report_sha256": sha256(EVAL / "quality-report.json"), "completed_at": now()})
        evidence = ARTIFACT / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        shutil.copy2(EVAL / "quality-report.json", evidence / "quality-report.json")
        shutil.copy2(EVAL / "quality-status.json", evidence / "quality-status.json")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
