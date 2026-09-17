"""Resident, fixed two-stage native EXL3 REAP loader (no pipeline transport).

Global layer indices/configuration are preserved. Stage zero owns 0..22 plus
embeddings; stage one owns 23..44. Unowned skeleton tensors stay on meta.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import stat
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open

from reap.glm53_exl3_native import NativeEXL3Experts
from reap.glm53_exl3_rocm import EXL3TensorIndex, LayerObservation

RESERVE_BYTES = 12 * 1024**3
STAGE_LAYERS = (tuple(range(23)), tuple(range(23, 45)))
MODEL_PREFIX = "model.language_model."
_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
                "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
                "I64": 8, "U64": 8, "F64": 8}


def _dependencies():
    # Heavy Transformers/model integration imports are deferred for CPU tests.
    from reap import glm53_exl3_observe as observer
    return SimpleNamespace(AutoConfig=observer.AutoConfig,
                           init_empty_weights=observer.init_empty_weights,
                           model_class=observer.Glm5NextTextModel,
                           IndexedTensors=observer.IndexedTensors,
                           source_key=observer._source_key,
                           set_tensor=observer.set_module_tensor_to_device)


def _forward_layer(*args, **kwargs):
    from reap.glm53_exl3_observe import _forward_layer_chunked
    return _forward_layer_chunked(*args, **kwargs)


def _forward_layer_batched(*args, **kwargs):
    from reap.glm53_observation_batched import forward_layer_batched_experts
    return forward_layer_batched_experts(*args, **kwargs)


def _validate_config(config) -> None:
    if config.num_hidden_layers != 45 or config.hidden_size != 4096:
        raise ValueError("resident stages require the original 45-layer, 4096-hidden Flash config")
    if list(config.indexer_types) != ["full"] * 45:
        raise ValueError("resident stages require all full indexers; no cross-stage top-k sharing")


def _owned_parameter(name: str, stage_id: int) -> bool:
    if name == "embed_tokens.weight":
        return stage_id == 0
    match = re.match(r"^layers\.(\d+)\.", name)
    return bool(match and int(match.group(1)) in STAGE_LAYERS[stage_id])


def _packed_layer_estimates(index, layers: tuple[int, ...]) -> dict[int, int]:
    """Read safetensors headers only; never decode or load expert tensors."""
    grouped = {}
    estimates = {layer: 0 for layer in layers if layer >= 3}
    for name in index.weight_map:
        match = re.match(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.", name)
        if match and int(match.group(1)) in estimates:
            grouped.setdefault(index.file_for(name), []).append((int(match.group(1)), name))
    for path, entries in grouped.items():
        with safe_open(path, framework="pt", device="cpu") as handle:
            for layer, name in entries:
                tensor = handle.get_slice(name)
                dtype = tensor.get_dtype()
                if dtype not in _DTYPE_BYTES:
                    raise ValueError(f"unsupported packed tensor dtype: {dtype}")
                estimates[layer] += math.prod(tensor.get_shape()) * _DTYPE_BYTES[dtype]
    if any(value <= 0 for value in estimates.values()):
        raise ValueError("missing packed layer payload estimate")
    return estimates


def _advise_task_weight_cache(root: Path, index, stage_id: int) -> dict[str, int]:
    """Drop only clean cache pages for this stage's indexed read-only weights.

    On Spark UMA, CUDA free can exclude reclaimable Linux file cache populated
    by the mandatory full checksum pass. This advisory does not delete files,
    release live tensor storage, or grant an unverified allocation budget.
    """
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return {"advised_files": 0, "advised_file_bytes": 0, "advice_errors": 0}
    root = root.resolve()
    filenames = {filename for name, filename in index.weight_map.items()
                 if name.startswith(MODEL_PREFIX) and _owned_parameter(name[len(MODEL_PREFIX):], stage_id)}
    paths = []
    for filename in sorted(filenames):
        original = root / filename
        path = original.resolve()
        if (Path(filename).is_absolute() or ".." in Path(filename).parts or original.is_symlink()
                or not path.is_relative_to(root) or path.suffix != ".safetensors"):
            raise ValueError("unsafe task weight path for cache advice")
        paths.append(path)
    result = {"advised_files": 0, "advised_file_bytes": 0, "advice_errors": 0}
    for path in paths:
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("cache advice requires a regular task weight file")
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                result["advised_files"] += 1
                result["advised_file_bytes"] += metadata.st_size
            finally:
                os.close(descriptor)
        except OSError:
            result["advice_errors"] += 1
    return result


def _ensure_room(device, next_bytes: int, *, label: str, reclaim=None) -> None:
    free, _ = torch.cuda.mem_get_info(device)
    if free - next_bytes < RESERVE_BYTES and reclaim is not None:
        before = free
        advice = reclaim()
        free, _ = torch.cuda.mem_get_info(device)
        _progress("stage_task_file_cache_advice", cuda_free_before=before, cuda_free_after=free, **advice)
    if free - next_bytes < RESERVE_BYTES:
        raise RuntimeError(f"resident stage reserve would be breached by {label}: "
                           f"free={free}, next_estimated_bytes={next_bytes}, reserve={RESERVE_BYTES}")


def _progress(event: str, **values) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


@dataclass
class ObservationStage:
    model: Any
    stage_id: int
    owned_layers: tuple[int, ...]
    observations: dict[int, LayerObservation]
    trunk_bytes: int
    cache_bytes: int
    device: torch.device
    prefill_chunk_size: int = 512
    batch_experts: bool = False

    def reset_observations(self) -> None:
        # Keep dictionary identity: all native sinks capture this exact mapping.
        for layer in self.observations:
            self.observations[layer] = LayerObservation.empty()

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        if ids.ndim != 2 or ids.dtype not in (torch.int64, torch.int32) or ids.shape[0] < 1:
            raise ValueError("stage input ids must be a nonempty [batch,sequence] integer tensor")
        if ids.shape[1] < 64 or ids.shape[1] % 64:
            raise ValueError("stage sequence length must be a positive multiple of 64")
        ids = ids.to(self.device)
        if hidden is None:
            if self.stage_id != 0:
                raise ValueError("stage one requires incoming BF16 hidden streams")
            hidden = self.model.embed_tokens(ids).unsqueeze(2).expand(-1, -1, 4, -1)
        if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (*ids.shape, 4, 4096):
            raise ValueError("stage hidden must be BF16 [batch,sequence,4,4096]")
        hidden = hidden.to(self.device)
        forward_layer = _forward_layer_batched if self.batch_experts else _forward_layer
        for layer in self.owned_layers:
            hidden, next_topk = forward_layer(self.model.layers[layer], hidden, ids, None,
                                              self.model.config, chunk_size=self.prefill_chunk_size)
            if next_topk is not None:
                raise RuntimeError("all-full-indexer stage unexpectedly produced shared top-k state")
        return hidden


def load_stage(model_root: Path, stage_id: int, prefill_chunk_size: int = 512, *,
               batch_experts: bool = False) -> ObservationStage:
    if type(stage_id) is not int or stage_id not in (0, 1):
        raise ValueError("stage_id must be zero or one")
    if type(prefill_chunk_size) is not int or prefill_chunk_size < 64 or prefill_chunk_size % 64:
        raise ValueError("prefill chunk size must be a positive multiple of 64")
    if type(batch_experts) is not bool:
        raise ValueError("batch_experts must be an explicit boolean")
    if not torch.cuda.is_available():
        raise RuntimeError("resident native stage requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    root = model_root.resolve()
    deps = _dependencies()
    config = deps.AutoConfig.from_pretrained(root, local_files_only=True).text_config
    _validate_config(config)
    if batch_experts:
        from reap.glm53_observation_batched import validate_config
        validate_config(config)
    config._attn_implementation = "sdpa"
    owned = STAGE_LAYERS[stage_id]
    index = EXL3TensorIndex(root)
    def ensure_room(next_bytes, *, label):
        _ensure_room(device, next_bytes, label=label,
                     reclaim=lambda: _advise_task_weight_cache(root, index, stage_id))
    ensure_room(0, label="initial reserve")
    estimates = _packed_layer_estimates(index, owned)
    _progress("stage_plan", stage_id=stage_id, owned_layers=list(owned),
              packed_estimated_bytes=sum(estimates.values()), packed_layer_estimated_bytes=estimates,
              reserve_bytes=RESERVE_BYTES)
    with deps.init_empty_weights():
        model = deps.model_class(config)
    observations = {layer: LayerObservation.empty() for layer in owned if layer >= 3}
    trunk_bytes, cache_bytes = 0, 0
    try:
        for layer in observations:
            sink = lambda _layer, obs, idx=layer: observations[idx].merge_(obs)
            model.layers[layer].mlp.experts = NativeEXL3Experts(root, layer, observation_sink=sink, tensor_index=index)
        indexed = deps.IndexedTensors(root)
        for name, _ in list(model.named_parameters()) + list(model.named_buffers()):
            if not _owned_parameter(name, stage_id) or ".mlp.experts." in name or name.endswith(".self_attn.conv1d.weight"):
                continue
            source = indexed.get(deps.source_key(name))
            size = source.numel() * source.element_size()
            ensure_room(size, label=f"retained tensor {name}")
            deps.set_tensor(model, name, device, value=source, dtype=source.dtype)
            trunk_bytes += size
            del source
        for layer in owned:
            if model.layers[layer].block_type != "linear_attention":
                continue
            parts = [indexed.get(f"{MODEL_PREFIX}layers.{layer}.self_attn.{axis}_conv1d.weight")
                     for axis in ("q", "k", "v")]
            convolution = torch.cat(parts, dim=0)
            size = convolution.numel() * convolution.element_size()
            ensure_room(size, label=f"layer {layer} convolution")
            deps.set_tensor(model, f"layers.{layer}.self_attn.conv1d.weight", device,
                            value=convolution, dtype=convolution.dtype)
            trunk_bytes += size
            del parts, convolution
        _progress("stage_trunk_ready", stage_id=stage_id, trunk_bytes=trunk_bytes)
        for layer in observations:
            # Small wrapper allocations and CUDA allocator rounding exceed raw
            # packed bytes; budget 10% overhead before each resident layer.
            estimated = math.ceil(estimates[layer] * 1.1)
            ensure_room(estimated, label=f"packed layer {layer}")
            before = torch.cuda.memory_allocated(device)
            _progress("stage_cache_start", stage_id=stage_id, layer=layer,
                      source_packed_bytes=estimates[layer], budget_bytes=estimated)
            model.layers[layer].mlp.experts.materialize_cache(device)
            after = torch.cuda.memory_allocated(device)
            cache_bytes += after - before
            ensure_room(0, label=f"post-load layer {layer}")
            _progress("stage_cache_ready", stage_id=stage_id, layer=layer,
                      allocated_before=before, allocated_after=after, cache_bytes=cache_bytes)
        model.eval()
        _progress("stage_ready", stage_id=stage_id, trunk_bytes=trunk_bytes, cache_bytes=cache_bytes)
        return ObservationStage(model, stage_id, owned, observations, trunk_bytes, cache_bytes,
                                device, prefill_chunk_size, batch_experts=batch_experts)
    except Exception:
        for layer in observations:
            experts = model.layers[layer].mlp.experts
            if isinstance(experts, NativeEXL3Experts):
                experts.clear_cache()
        model.to_empty(device="meta")
        torch.cuda.empty_cache()
        raise
