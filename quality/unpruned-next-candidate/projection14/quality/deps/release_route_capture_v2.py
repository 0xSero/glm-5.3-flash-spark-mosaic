#!/usr/bin/env python3
"""Resumable full-trunk natural-route coverage capture for GLM-5.3 Flash.

This stage advances the pinned BF16 calibration rows through all 45 trunk
layers while retaining only two hidden-state slots. It records natural top-8
routes for each routed layer. It does not capture Hessians, encode weights, or
make runtime/quality claims.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as dt
import fcntl
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from reap.glm53_virtual_abliteration import VirtualAbliteration


CAMPAIGN = Path(os.environ.get("GLM53_EXL3_ROOT", "/home/sero/glm53-exl3"))
SOURCE = Path(os.environ.get("GLM53_SOURCE_ROOT", "/home/sero/models/GLM-5.3-Flash-BF16"))
CALIBRATION = Path(os.environ.get("GLM53_CALIBRATION_ROOT", str(CAMPAIGN / "release-calibration-v2")))
DEFAULT_OUTPUT = CAMPAIGN / "release-route-capture-v2"
SOURCE_REVISION = "a6c167b62691b2bac901344b65cb651a70f53e43"
TRANSFORMERS_REVISION = "dabae5fcb924a8eece0e727b627ca5f050b40d40"
ROWS = 600
COLUMNS = 2048
LAYERS = 45
ROUTED_FIRST = 3
EXPERTS = 288
TOP_K = 8
ROUTE_FLOOR = 1024
GPU_IDS = tuple(int(value) for value in os.environ.get("GLM53_EXL3_GPU_IDS", "0,1").split(","))


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
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file(tensors, temporary)
    temporary.replace(path)
    return path.stat().st_size, sha256(path)


def compute_apps() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


class SourceTensors:
    def __init__(self) -> None:
        index = json.loads((SOURCE / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.virtual_abliteration = VirtualAbliteration.from_environment(
            source_root=SOURCE
        )

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"source tensor not indexed: {key}")
        with safe_open(SOURCE / shard, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
        if self.virtual_abliteration is not None:
            tensor = self.virtual_abliteration.get(key, tensor)
        return tensor


def source_key(layer_idx: int, target_key: str) -> str:
    prefix = f"model.language_model.layers.{layer_idx}."
    aliases = {
        "attn_hc.fn": "hc_attn_fn",
        "attn_hc.base": "hc_attn_base",
        "attn_hc.scale": "hc_attn_scale",
        "ffn_hc.fn": "hc_ffn_fn",
        "ffn_hc.base": "hc_ffn_base",
        "ffn_hc.scale": "hc_ffn_scale",
        "self_attn.forget_gate.dt_bias": "self_attn.dt_bias",
        "self_attn.forget_gate.A_log": "self_attn.A_log",
        "self_attn.forget_gate.f_a_proj.weight": "self_attn.f_a_proj.weight",
        "self_attn.forget_gate.f_b_proj.weight": "self_attn.f_b_proj.weight",
    }
    return prefix + aliases.get(target_key, target_key)


def copy_tensor(target: torch.Tensor, source: torch.Tensor, name: str) -> int:
    if tuple(target.shape) != tuple(source.shape):
        raise RuntimeError(
            f"shape mismatch for {name}: target={tuple(target.shape)} source={tuple(source.shape)}"
        )
    if source.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(f"unsupported source dtype for {name}: {source.dtype}")
    if target.dtype != source.dtype:
        target.data = source.to(device=target.device, non_blocking=False)
    else:
        target.copy_(source.to(device=target.device, non_blocking=False))
    return source.numel() * source.element_size()


def load_layer(layer_idx: int, config, source: SourceTensors, device: torch.device):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            layer = Glm5NextTextDecoderLayer(config, layer_idx)
    finally:
        torch.set_default_dtype(previous_dtype)
    layer.eval()

    sparse = hasattr(layer.mlp, "experts")
    special = {"self_attn.conv1d.weight"}
    if sparse:
        special.update({"mlp.experts.gate_up_proj", "mlp.experts.down_proj"})
    copied_bytes = 0
    with torch.inference_mode():
        for target_name, target in list(layer.named_parameters()) + list(layer.named_buffers()):
            if target_name in special:
                continue
            tensor = source.get(source_key(layer_idx, target_name))
            copied_bytes += copy_tensor(target, tensor, target_name)
            del tensor

        if layer.block_type == "linear_attention":
            parts = [
                source.get(source_key(layer_idx, f"self_attn.{name}_conv1d.weight"))
                for name in ("q", "k", "v")
            ]
            convolution = torch.cat(parts, dim=0)
            copied_bytes += copy_tensor(
                layer.self_attn.conv1d.weight, convolution, "self_attn.conv1d.weight"
            )
            del parts, convolution

        if sparse:
            gate_up = layer.mlp.experts.gate_up_proj
            down = layer.mlp.experts.down_proj
            intermediate = layer.mlp.experts.intermediate_dim
            prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
            for expert_idx in range(layer.mlp.experts.num_experts):
                gate = source.get(f"{prefix}.{expert_idx}.gate_proj.weight")
                up = source.get(f"{prefix}.{expert_idx}.up_proj.weight")
                down_source = source.get(f"{prefix}.{expert_idx}.down_proj.weight")
                copied_bytes += copy_tensor(
                    gate_up[expert_idx, :intermediate], gate, f"expert {expert_idx} gate"
                )
                copied_bytes += copy_tensor(
                    gate_up[expert_idx, intermediate:], up, f"expert {expert_idx} up"
                )
                copied_bytes += copy_tensor(
                    down[expert_idx], down_source, f"expert {expert_idx} down"
                )
                del gate, up, down_source

    expected = sum(value.numel() * value.element_size() for value in layer.state_dict().values())
    if copied_bytes != expected:
        raise RuntimeError(f"loaded-byte mismatch: copied={copied_bytes} expected={expected}")
    return layer, copied_bytes


def row_paths(slot: Path, row_idx: int) -> tuple[Path, Path]:
    return slot / f"row-{row_idx:03d}.safetensors", slot / f"row-{row_idx:03d}.json"


def verified_row(slot: Path, row_idx: int, expected_layer: int) -> bool:
    data, sidecar = row_paths(slot, row_idx)
    if not data.is_file() or not sidecar.is_file():
        return False
    try:
        state = json.loads(sidecar.read_text())
        return (
            state.get("layer") == expected_layer
            and state.get("source_revision") == SOURCE_REVISION
            and state.get("bytes") == data.stat().st_size
            and state.get("sha256") == sha256(data)
        )
    except Exception:
        return False


def save_row(
    slot: Path,
    row_idx: int,
    layer_idx: int,
    hidden: torch.Tensor,
    extra: dict | None = None,
) -> dict:
    data, sidecar = row_paths(slot, row_idx)
    hidden_cpu = hidden.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    size, digest = atomic_safetensors(data, {"hidden_streams": hidden_cpu})
    state = {
        "row": row_idx,
        "layer": layer_idx,
        "source_revision": SOURCE_REVISION,
        "shape": list(hidden_cpu.shape),
        "dtype": str(hidden_cpu.dtype),
        "bytes": size,
        "sha256": digest,
        "completed_at": now(),
    }
    if extra:
        state.update(extra)
    atomic_json(sidecar, state)
    return state


def initialize_embeddings(output: Path, row_limit: int) -> Path:
    slot = output / "state-a"
    manifest_path = slot / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("layer") == -1 and manifest.get("rows") == row_limit:
            if all(verified_row(slot, row, -1) for row in range(row_limit)):
                return slot

    slot.mkdir(parents=True, exist_ok=True)
    token_rows = load_file(CALIBRATION / "token_rows.safetensors")["input_ids"][:row_limit]
    source = SourceTensors()
    embedding = source.get("model.language_model.embed_tokens.weight")
    row_states = []
    for row_idx in range(row_limit):
        hidden = embedding[token_rows[row_idx]].unsqueeze(0).unsqueeze(2)
        hidden = hidden.expand(-1, -1, 4, -1).contiguous()
        row_states.append(save_row(slot, row_idx, -1, hidden))
    atomic_json(
        manifest_path,
        {
            "state": "COMPLETE",
            "layer": -1,
            "rows": row_limit,
            "source_revision": SOURCE_REVISION,
            "token_rows_sha256": sha256(CALIBRATION / "token_rows.safetensors"),
            "row_sha256": [state["sha256"] for state in row_states],
            "completed_at": now(),
        },
    )
    del embedding, token_rows
    gc.collect()
    return slot


def worker(
    layer_idx: int,
    gpu_id: int,
    rows: list[int],
    input_slot_string: str,
    output_slot_string: str,
    output_root_string: str,
    axis_by_row: list[str],
) -> dict:
    input_slot = Path(input_slot_string)
    output_slot = Path(output_slot_string)
    output_root = Path(output_root_string)
    worker_status = output_root / f"worker-gpu{gpu_id}.json"
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cuda.matmul.allow_tf32 = False
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig

        config = Glm5NextConfig.from_pretrained(SOURCE, local_files_only=True).text_config
        config._attn_implementation = "sdpa"
        source = SourceTensors()
        started = time.monotonic()
        layer, loaded_bytes = load_layer(layer_idx, config, source, device)
        sparse = hasattr(layer.mlp, "gate")
        route_counts = torch.zeros(EXPERTS, dtype=torch.long)
        axes = sorted(set(axis_by_row))
        axis_counts = {axis: torch.zeros(EXPERTS, dtype=torch.long) for axis in axes}
        captured_indices = None

        def route_hook(_module, _inputs, outputs):
            nonlocal captured_indices
            captured_indices = outputs[2].detach()

        hook = layer.mlp.gate.register_forward_hook(route_hook) if sparse else None
        completed = 0
        skipped = 0
        attention_mask = torch.ones((1, COLUMNS), dtype=torch.bool, device=device)
        with torch.inference_mode():
            for row_idx in rows:
                if verified_row(output_slot, row_idx, layer_idx):
                    if sparse:
                        _, sidecar = row_paths(output_slot, row_idx)
                        prior = json.loads(sidecar.read_text())
                        prior_counts = torch.tensor(prior["natural_route_counts"], dtype=torch.long)
                        route_counts += prior_counts
                        axis_counts[axis_by_row[row_idx]] += prior_counts
                    skipped += 1
                    continue
                input_data, _ = row_paths(input_slot, row_idx)
                hidden = load_file(input_data)["hidden_streams"].to(device=device)
                captured_indices = None
                output_hidden, _ = layer(
                    hidden,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=False,
                    prev_topk_indices=None,
                )
                if not torch.isfinite(output_hidden).all().item():
                    raise RuntimeError(f"non-finite output at layer {layer_idx} row {row_idx}")
                if sparse:
                    if captured_indices is None:
                        raise RuntimeError("router hook did not capture natural expert indices")
                    counts = torch.bincount(captured_indices.flatten().cpu(), minlength=EXPERTS)
                    route_counts += counts
                    axis_counts[axis_by_row[row_idx]] += counts
                    row_extra = {
                        "routing_policy": "natural_top8",
                        "axis": axis_by_row[row_idx],
                        "natural_route_counts": counts.tolist(),
                    }
                else:
                    row_extra = None
                save_row(output_slot, row_idx, layer_idx, output_hidden, row_extra)
                completed += 1
                atomic_json(
                    worker_status,
                    {
                        "state": "RUNNING",
                        "updated_at": now(),
                        "layer": layer_idx,
                        "gpu": gpu_id,
                        "row": row_idx,
                        "completed": completed,
                        "skipped": skipped,
                        "assigned": len(rows),
                    },
                )
                del hidden, output_hidden
        if hook is not None:
            hook.remove()
        torch.cuda.synchronize(device)
        result = {
            "state": "COMPLETE",
            "layer": layer_idx,
            "gpu": gpu_id,
            "assigned": len(rows),
            "completed": completed,
            "skipped": skipped,
            "loaded_bytes": loaded_bytes,
            "elapsed_seconds": time.monotonic() - started,
            "route_counts": route_counts.tolist() if sparse else None,
            "route_counts_by_axis": (
                {axis: counts.tolist() for axis, counts in axis_counts.items()} if sparse else None
            ),
        }
        atomic_json(worker_status, {**result, "updated_at": now()})
        return result
    except Exception as error:
        atomic_json(
            worker_status,
            {
                "state": "FAILED",
                "updated_at": now(),
                "layer": layer_idx,
                "gpu": gpu_id,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def safe_remove_slot(slot: Path, output: Path) -> None:
    resolved_slot = slot.resolve()
    resolved_output = output.resolve()
    if resolved_slot.parent != resolved_output or resolved_slot.name not in {"state-a", "state-b"}:
        raise RuntimeError(f"refusing to remove unexpected state slot: {resolved_slot}")
    if resolved_slot.exists():
        shutil.rmtree(resolved_slot)


def process_layer(
    output: Path,
    layer_idx: int,
    input_slot: Path,
    output_slot: Path,
    row_limit: int,
    axis_by_row: list[str],
) -> dict:
    output_slot.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "layers" / f"layer-{layer_idx:02d}.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("state") == "COMPLETE" and manifest.get("rows") == row_limit:
            if all(verified_row(output_slot, row, layer_idx) for row in range(row_limit)):
                return manifest

    row_shards = [list(range(gpu, row_limit, len(GPU_IDS))) for gpu in range(len(GPU_IDS))]
    context = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=len(GPU_IDS), mp_context=context) as executor:
        futures = [
            executor.submit(
                worker,
                layer_idx,
                gpu_id,
                row_shards[position],
                str(input_slot),
                str(output_slot),
                str(output),
                axis_by_row,
            )
            for position, gpu_id in enumerate(GPU_IDS)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    if not all(verified_row(output_slot, row, layer_idx) for row in range(row_limit)):
        raise RuntimeError(f"layer {layer_idx} output rows are incomplete")
    row_sidecars = [json.loads(row_paths(output_slot, row)[1].read_text()) for row in range(row_limit)]
    sparse = layer_idx >= ROUTED_FIRST
    route_counts = None
    route_counts_by_axis = None
    coverage = None
    if sparse:
        route_counts = [sum(result["route_counts"][expert] for result in results) for expert in range(EXPERTS)]
        axes = sorted(set(axis_by_row))
        route_counts_by_axis = {
            axis: [
                sum(result["route_counts_by_axis"][axis][expert] for result in results)
                for expert in range(EXPERTS)
            ]
            for axis in axes
        }
        coverage = {
            "minimum": min(route_counts),
            "maximum": max(route_counts),
            "zero_hit_count": sum(count == 0 for count in route_counts),
            "below_floor_count": sum(count < ROUTE_FLOOR for count in route_counts),
            "floor": ROUTE_FLOOR,
            "total_routes": sum(route_counts),
            "expected_total_routes": row_limit * COLUMNS * TOP_K,
        }
        if coverage["total_routes"] != coverage["expected_total_routes"]:
            raise RuntimeError(f"route total mismatch at layer {layer_idx}: {coverage}")

    manifest = {
        "state": "COMPLETE",
        "completed_at": now(),
        "layer": layer_idx,
        "rows": row_limit,
        "columns": COLUMNS,
        "source_revision": SOURCE_REVISION,
        "input_layer": layer_idx - 1,
        "output_slot": output_slot.name,
        "row_sha256": [state["sha256"] for state in row_sidecars],
        "workers": sorted(results, key=lambda result: result["gpu"]),
        "route_counts": route_counts,
        "route_counts_by_axis": route_counts_by_axis,
        "coverage": coverage,
        "calibration_capture_claim": True,
        "hessian_capture_claim": False,
        "runtime_claim": False,
        "quality_claim": False,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--last-layer", type=int, default=LAYERS - 1)
    parser.add_argument("--keep-slots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (1 <= args.rows <= ROWS):
        raise RuntimeError("row count outside the pinned calibration range")
    if not (0 <= args.first_layer <= args.last_layer < LAYERS):
        raise RuntimeError("invalid layer range")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with (output / "capture.lock").open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another release-route controller owns capture.lock")

        status_path = output / "status.json"
        atomic_json(status_path, {"state": "PREFLIGHT", "updated_at": now()})
        pilot = json.loads((CAMPAIGN / "pilot-status.json").read_text())
        calibration = json.loads((CALIBRATION / "manifest.json").read_text())
        if pilot.get("state") != "COMPLETE" or calibration.get("status") != "pass":
            raise RuntimeError("pilot or release calibration gate is not complete")
        if calibration.get("rows") != ROWS or calibration.get("columns") != COLUMNS:
            raise RuntimeError("extended calibration dimensions do not match the capture contract")
        if sha256(CALIBRATION / "token_rows.safetensors") != calibration.get(
            "token_rows_sha256"
        ):
            raise RuntimeError("extended calibration token digest mismatch")
        if subprocess.check_output(
            ["git", "-C", str(CAMPAIGN / "transformers"), "rev-parse", "HEAD"], text=True
        ).strip() != TRANSFORMERS_REVISION:
            raise RuntimeError("Transformers revision mismatch")
        apps = compute_apps()
        if apps:
            atomic_json(
                status_path,
                {"state": "GPU_OWNERSHIP_BLOCKED", "updated_at": now(), "gpu_compute_apps": apps},
            )
            return 2
        stat = os.statvfs(output)
        free_bytes = stat.f_bavail * stat.f_frsize
        if free_bytes < 1_100_000_000_000:
            raise RuntimeError(f"storage floor failed: {free_bytes}")

        axis_by_row = calibration["axis_by_row"][: args.rows]
        start_layer = args.first_layer
        input_slot = None
        if args.first_layer == 0:
            contiguous = -1
            for candidate_layer in range(LAYERS):
                candidate_path = output / "layers" / f"layer-{candidate_layer:02d}.json"
                if not candidate_path.is_file():
                    break
                candidate = json.loads(candidate_path.read_text())
                if candidate.get("state") != "COMPLETE" or candidate.get("rows") != args.rows:
                    break
                contiguous = candidate_layer
            if contiguous >= 0:
                start_layer = contiguous + 1
                previous_manifest = json.loads(
                    (output / "layers" / f"layer-{contiguous:02d}.json").read_text()
                )
                input_slot = output / previous_manifest["output_slot"]
            else:
                input_slot = initialize_embeddings(output, args.rows)
        else:
            previous_manifest = json.loads(
                (output / "layers" / f"layer-{args.first_layer - 1:02d}.json").read_text()
            )
            input_slot = output / previous_manifest["output_slot"]

        manifests = []
        for layer_idx in range(start_layer, args.last_layer + 1):
            output_slot = output / ("state-b" if input_slot.name == "state-a" else "state-a")
            atomic_json(
                status_path,
                {
                    "state": "RUNNING",
                    "updated_at": now(),
                    "layer": layer_idx,
                    "rows": args.rows,
                    "input_slot": input_slot.name,
                    "output_slot": output_slot.name,
                },
            )
            manifest = process_layer(
                output, layer_idx, input_slot, output_slot, args.rows, axis_by_row
            )
            manifests.append(manifest)
            if not args.keep_slots:
                safe_remove_slot(input_slot, output)
            input_slot = output_slot

        all_manifests = []
        for layer_idx in range(0, args.last_layer + 1):
            path = output / "layers" / f"layer-{layer_idx:02d}.json"
            if path.is_file():
                all_manifests.append(json.loads(path.read_text()))
        routed = [m for m in all_manifests if m.get("coverage") is not None]
        all_coverage_pass = bool(routed) and all(
            m["coverage"]["zero_hit_count"] == 0 and m["coverage"]["below_floor_count"] == 0
            for m in routed
        )
        final = {
            "state": "COMPLETE" if args.last_layer == LAYERS - 1 else "PARTIAL",
            "updated_at": now(),
            "rows": args.rows,
            "first_layer": args.first_layer,
            "last_layer": args.last_layer,
            "final_state_slot": input_slot.name,
            "routed_layers_in_this_run": len(routed),
            "all_observed_route_coverage_pass": all_coverage_pass,
            "minimum_route_count": min(
                (m["coverage"]["minimum"] for m in routed), default=None
            ),
            "hessian_capture_claim": False,
            "q4_encoding_claim": False,
            "runtime_claim": False,
            "quality_claim": False,
            "next_gate": "hessian_capture_and_integer_cache_encoding" if all_coverage_pass else "route_coverage_review",
        }
        atomic_json(status_path, final)
        return 0 if all_coverage_pass else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        output = parse_args().output
        atomic_json(
            output / "status.json",
            {
                "state": "FAILED",
                "updated_at": now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
