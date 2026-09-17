"""ROCm-compatible GLM-5.3 selective-EXL3 observation primitives.

The published GLM-5.3 Flash EXL3 checkpoints use rank-sliced MCG/Trellis
weights and a CUDA-only serving primitive.  Observation does not need that
serving kernel, but it does need the *same quantized expert outputs*.  This
module therefore reconstructs one routed expert at a time with NumPy and
ordinary PyTorch operations.  It is deliberately slow, exact, and portable;
the surrounding runner keeps the native Transformers trunk and REAP's
layerwise/statistical contracts.

This module never claims that the checkpoint is a stock Transformers model or
that its CUDA serving runtime works on AMD.
"""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
import torch.nn.functional as F


HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
LOCAL_INTERMEDIATE_SIZE = 512
NUM_EXPERTS = 288
TP_SIZE = 4
TOP_K = 8
ROUTED_SCALING_FACTOR = 2.5
SWIGLU_LIMIT = 10.0

_MCG_MULTIPLIER = np.uint64(0xCBAC1FED)
_MCG_MASK = np.uint32(0x8FFF8FFF)
_MCG_XOR = np.uint32(0x3B603B60)


def _decode_3inst_fp16(window: np.ndarray) -> np.ndarray:
    value = window.astype(np.uint64)
    value = ((value * _MCG_MULTIPLIER) & np.uint64(0xFFFFFFFF)).astype(
        np.uint32
    )
    value = np.uint32((value & _MCG_MASK) ^ _MCG_XOR)
    low = (value & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
    high = (
        ((value >> np.uint32(16)) & np.uint32(0xFFFF))
        .astype(np.uint16)
        .view(np.float16)
    )
    return (low.astype(np.float16) + high.astype(np.float16)).astype(np.float16)


def _decode_lane(tile_words: np.ndarray, lane: int, bits: int) -> np.ndarray:
    width = 8 * bits
    values = []
    for weight in range(8):
        end_bit = (lane * 8 + weight + 257) * bits
        start_bit = end_bit - 16
        first_word = start_bit // 32
        last_word = (end_bit - 1) // 32
        shift = (last_word + 1) * 32 - end_bit
        first = tile_words[..., first_word % width].astype(np.uint64)
        last = tile_words[..., last_word % width].astype(np.uint64)
        merged = (first << np.uint64(32)) | last
        window = ((merged >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(
            np.uint32
        )
        values.append(_decode_3inst_fp16(window))
    return np.stack(values, axis=-1).astype(np.float16)


def reconstruct_trellis(trellis: torch.Tensor) -> torch.Tensor:
    """Reconstruct one native EXL3 MCG matrix on CPU.

    The packed tensor has shape ``[input_tiles, output_tiles, bits * 16]``.
    The returned matrix is laid out as ``[input_features, output_features]``
    and is therefore multiplied as ``x @ weight``.
    """

    if trellis.ndim != 3 or trellis.dtype != torch.int16:
        raise ValueError("trellis must be a rank-3 int16 tensor")
    native = trellis.detach().cpu().contiguous().numpy()
    if native.shape[-1] % 16:
        raise ValueError("trellis payload width is not divisible by 16")
    bits = int(native.shape[-1]) // 16
    if bits not in (2, 3, 4, 5, 6):
        raise ValueError(f"unsupported EXL3 bitrate: {bits}")

    input_tiles, output_tiles, _ = native.shape
    packed = native.view(np.uint16).reshape(
        input_tiles, output_tiles, 8 * bits, 2
    )
    words = packed[..., 0].astype(np.uint32) | (
        packed[..., 1].astype(np.uint32) << np.uint32(16)
    )
    lanes = np.stack([_decode_lane(words, lane, bits) for lane in range(32)])
    blocks = np.zeros(
        (input_tiles, output_tiles, 16, 16), dtype=np.float16
    )
    for lane in range(32):
        row0 = (lane % 4) * 2
        rows = (row0, row0 + 1, row0 + 8, row0 + 9)
        col0 = lane // 8
        col1 = col0 + 4
        parity = (lane >> 2) & 1
        for weight in range(8):
            blocks[
                :,
                :,
                rows[weight % 4],
                2 * (col0 if weight < 4 else col1) + parity,
            ] = lanes[lane, ..., weight]
    output = blocks.transpose(0, 2, 1, 3).reshape(
        input_tiles * 16, output_tiles * 16
    )
    return torch.from_numpy(output.copy())


@lru_cache(maxsize=8)
def _hadamard_128_cpu(dtype: torch.dtype) -> torch.Tensor:
    indices = torch.arange(128, dtype=torch.int64)
    rows = []
    for row in range(128):
        parity = torch.tensor(
            [(row & int(column)).bit_count() & 1 for column in indices],
            dtype=torch.bool,
        )
        rows.append(torch.where(parity, -1.0, 1.0))
    return (torch.stack(rows) / (128.0**0.5)).to(dtype)


@lru_cache(maxsize=16)
def _hadamard_128_device(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Retain the tiny rotation on each exact device, avoiding per-call copies."""
    return _hadamard_128_cpu(dtype).to(device=device)


def _rounded_fp32_cache(value: torch.Tensor, device: torch.device | str,
                        compute_dtype: torch.dtype) -> torch.Tensor:
    """Preserve compute-dtype rounding before storing the FP32 GEMM operand."""
    return value.to(dtype=compute_dtype).to(device=device, dtype=torch.float32)


def apply_hadamard_128(
    value: torch.Tensor,
    *,
    suh: torch.Tensor | None = None,
    svh: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Apply EXL3's persisted 128-wide pre/post rotations."""

    work = value.float()
    if suh is not None:
        work = (work * suh.to(device=work.device, dtype=torch.float32)).to(
            torch.float16
        ).float()
    rows, width = work.shape
    if width % 128:
        raise ValueError(f"rotation width {width} is not divisible by 128")
    hadamard = _hadamard_128_device(work.device, torch.float32)
    work = (work.view(rows, width // 128, 128) @ hadamard).view(rows, width)
    if svh is not None:
        work = work * svh.to(device=work.device, dtype=torch.float32)
    return work.to(output_dtype)


class EXL3TensorIndex:
    """Resolve logical EXL3 tensor names to files in a Hub snapshot."""

    def __init__(self, model_root: str | Path):
        self.model_root = Path(model_root).resolve()
        index_path = self.model_root / "model.safetensors.index.json"
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("model index lacks weight_map")
        self.weight_map: dict[str, str] = {
            str(key): str(value) for key, value in weight_map.items()
        }

    def file_for(self, tensor_name: str) -> Path:
        relative = self.weight_map.get(tensor_name)
        if relative is None:
            raise KeyError(tensor_name)
        path = (self.model_root / relative).resolve()
        if not path.is_relative_to(self.model_root):
            raise ValueError(f"tensor file escapes model root: {relative}")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        return path


@dataclass
class LayerObservation:
    frequency: torch.Tensor
    activation_norm_sum: torch.Tensor
    activation_norm_max: torch.Tensor
    reap_sum: torch.Tensor
    reap_model_scaled_sum: torch.Tensor
    router_weight_sum: torch.Tensor

    @classmethod
    def empty(cls) -> "LayerObservation":
        return cls(
            frequency=torch.zeros(NUM_EXPERTS, dtype=torch.int64),
            activation_norm_sum=torch.zeros(NUM_EXPERTS, dtype=torch.float64),
            activation_norm_max=torch.zeros(NUM_EXPERTS, dtype=torch.float32),
            reap_sum=torch.zeros(NUM_EXPERTS, dtype=torch.float64),
            reap_model_scaled_sum=torch.zeros(NUM_EXPERTS, dtype=torch.float64),
            router_weight_sum=torch.zeros(NUM_EXPERTS, dtype=torch.float64),
        )

    def update(
        self,
        expert: int,
        activation_norm: torch.Tensor,
        model_scaled_weight: torch.Tensor,
    ) -> None:
        norms = activation_norm.detach().to(device="cpu", dtype=torch.float64)
        weights = model_scaled_weight.detach().to(
            device="cpu", dtype=torch.float64
        )
        paper_weights = weights / ROUTED_SCALING_FACTOR
        self.frequency[expert] += norms.numel()
        self.activation_norm_sum[expert] += norms.sum()
        self.activation_norm_max[expert] = torch.maximum(
            self.activation_norm_max[expert], norms.max().float()
        )
        self.reap_sum[expert] += (norms * paper_weights).sum()
        self.reap_model_scaled_sum[expert] += (norms * weights).sum()
        self.router_weight_sum[expert] += paper_weights.sum()

    def merge_(self, other: "LayerObservation") -> None:
        """Merge a batch observation into a corpus-level accumulator."""

        self.frequency += other.frequency
        self.activation_norm_sum += other.activation_norm_sum
        self.activation_norm_max = torch.maximum(
            self.activation_norm_max, other.activation_norm_max
        )
        self.reap_sum += other.reap_sum
        self.reap_model_scaled_sum += other.reap_model_scaled_sum
        self.router_weight_sum += other.router_weight_sum

    def as_dict(self) -> dict[str, Any]:
        counts = self.frequency.clamp_min(1).to(torch.float64)
        return {
            # REAP-compatible pruning aggregates.
            "total_tokens": int(self.frequency.sum().item() // TOP_K),
            "expert_frequency": self.frequency.tolist(),
            "ean_sum": self.activation_norm_sum.tolist(),
            "weighted_ean_sum": self.reap_sum.tolist(),
            "weighted_ean_model_scaled_sum": (
                self.reap_model_scaled_sum.tolist()
            ),
            "weighted_expert_frequency_sum": self.router_weight_sum.tolist(),
            "max_activations": self.activation_norm_max.tolist(),
            "activation_norm_mean": (
                self.activation_norm_sum / counts
            ).to(torch.float32).tolist(),
            "activation_norm_max": self.activation_norm_max.tolist(),
            "reap_score": (self.reap_sum / counts).to(torch.float32).tolist(),
            "reap_score_model_scaled": (
                self.reap_model_scaled_sum / counts
            ).to(torch.float32).tolist(),
            "router_weight_mean": (
                self.router_weight_sum / counts
            ).to(torch.float32).tolist(),
        }


class RocmEXL3Experts(nn.Module):
    """Single-device execution of one rank-sliced GLM-5.3 EXL3 expert layer."""

    def __init__(
        self,
        model_root: str | Path,
        layer: int,
        *,
        observation_sink: Callable[[int, LayerObservation], None] | None = None,
        compute_dtype: torch.dtype = torch.float16,
        tensor_index: EXL3TensorIndex | None = None,
    ) -> None:
        super().__init__()
        if layer not in range(3, 45):
            raise ValueError(f"layer outside routed scope: {layer}")
        if tensor_index is not None and tensor_index.model_root != Path(model_root).resolve():
            raise ValueError("shared tensor index belongs to another checkpoint")
        self.index = tensor_index if tensor_index is not None else EXL3TensorIndex(model_root)
        self.layer = layer
        self.num_experts = NUM_EXPERTS
        self.observation_sink = observation_sink
        self.compute_dtype = compute_dtype
        self.last_observation: LayerObservation | None = None
        self._projection_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    def _prefix(self, expert: int, projection: str, rank: int) -> str:
        return (
            f"model.language_model.layers.{self.layer}.mlp.experts.{expert}."
            f"{projection}.rank{rank}"
        )

    def _projection(
        self,
        value: torch.Tensor,
        *,
        expert: int,
        projection: str,
        rank: int,
        handles: Mapping[Path, Any] | None,
    ) -> torch.Tensor:
        prefix = self._prefix(expert, projection, rank)
        cached = self._projection_cache.get(prefix)
        if cached is None:
            if handles is None:
                raise RuntimeError(f"projection is not cached: {prefix}")
            file_path = self.index.file_for(f"{prefix}.trellis")
            handle = handles[file_path]
            trellis = handle.get_tensor(f"{prefix}.trellis")
            suh = handle.get_tensor(f"{prefix}.suh")
            svh = handle.get_tensor(f"{prefix}.svh")
            marker = handle.get_tensor(f"{prefix}.mcg")
            if marker.numel() not in (0, 1):
                raise ValueError(f"unexpected MCG marker geometry: {prefix}")
            weight = reconstruct_trellis(trellis).to(
                device=value.device, dtype=self.compute_dtype
            )
        else:
            weight, suh, svh = cached

        rotated = apply_hadamard_128(
            value, suh=suh, output_dtype=self.compute_dtype
        )
        # The CUDA reference materializes the decoded FP16 matrix, performs
        # accumulation in FP32, and rounds the projection back to FP16 before
        # the persisted output rotation.  Keep those boundaries on ROCm too.
        output = (rotated.float() @ weight.float()).to(self.compute_dtype)
        del weight
        return apply_hadamard_128(
            output, svh=svh, output_dtype=self.compute_dtype
        )

    def materialize_cache(
        self,
        device: torch.device | str,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Decode this layer once and retain its quantized matrices on GPU.

        A GLM-5.3 routed layer occupies roughly 27 GiB as FP32 GEMM operands,
        up from 13.5 GiB in decoded FP16 form. Values are rounded through the
        compute dtype first, so this removes repeated conversions rather than
        changing numerical precision. No decoded FP16 GPU copy is retained.
        """

        if self._projection_cache:
            raise RuntimeError("projection cache is already materialized")
        total = NUM_EXPERTS * 3 * TP_SIZE
        completed = 0
        all_paths = set().union(
            *(self._paths_for_expert(expert) for expert in range(NUM_EXPERTS))
        )
        with ExitStack() as stack:
            handles = {
                path: stack.enter_context(
                    safe_open(path, framework="pt", device="cpu")
                )
                for path in all_paths
            }
            for expert in range(NUM_EXPERTS):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    for rank in range(TP_SIZE):
                        prefix = self._prefix(expert, projection, rank)
                        handle = handles[
                            self.index.file_for(f"{prefix}.trellis")
                        ]
                        marker = handle.get_tensor(f"{prefix}.mcg")
                        if marker.numel() not in (0, 1):
                            raise ValueError(
                                f"unexpected MCG marker geometry: {prefix}"
                            )
                        weight = _rounded_fp32_cache(
                            reconstruct_trellis(handle.get_tensor(f"{prefix}.trellis")),
                            device, self.compute_dtype,
                        )
                        suh = _rounded_fp32_cache(
                            handle.get_tensor(f"{prefix}.suh"), device, self.compute_dtype,
                        )
                        svh = _rounded_fp32_cache(
                            handle.get_tensor(f"{prefix}.svh"), device, self.compute_dtype,
                        )
                        self._projection_cache[prefix] = (weight, suh, svh)
                        completed += 1
                        if progress is not None:
                            progress(completed, total)

    def clear_cache(self) -> None:
        self._projection_cache.clear()

    def _paths_for_expert(self, expert: int) -> set[Path]:
        return {
            self.index.file_for(
                f"{self._prefix(expert, projection, rank)}.trellis"
            )
            for projection in ("gate_proj", "up_proj", "down_proj")
            for rank in range(TP_SIZE)
        }

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != HIDDEN_SIZE:
            raise ValueError(
                f"expected [tokens,{HIDDEN_SIZE}], got {tuple(hidden_states.shape)}"
            )
        if (
            top_k_index.shape != top_k_weights.shape
            or top_k_index.shape != (hidden_states.shape[0], TOP_K)
        ):
            raise ValueError("GLM-5.3 routing tensors have unexpected geometry")

        final = torch.zeros_like(hidden_states, dtype=self.compute_dtype)
        mask = F.one_hot(top_k_index, num_classes=NUM_EXPERTS).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        observation = LayerObservation.empty()

        all_paths = (
            set()
            if self._projection_cache
            else set().union(*(self._paths_for_expert(expert) for expert in hit))
        )
        context = nullcontext(None) if self._projection_cache else ExitStack()
        with context as stack:
            handles = {
                path: stack.enter_context(  # type: ignore[union-attr]
                    safe_open(path, framework="pt", device="cpu")
                )
                for path in all_paths
            } if stack is not None else None
            for expert in hit:
                top_k_pos, token_idx = torch.where(mask[expert])
                expert_input = hidden_states[token_idx].to(
                    self.compute_dtype
                ).contiguous()
                gate_parts = [
                    self._projection(
                        expert_input,
                        expert=expert,
                        projection="gate_proj",
                        rank=rank,
                        handles=handles,
                    )
                    for rank in range(TP_SIZE)
                ]
                up_parts = [
                    self._projection(
                        expert_input,
                        expert=expert,
                        projection="up_proj",
                        rank=rank,
                        handles=handles,
                    )
                    for rank in range(TP_SIZE)
                ]
                gate = torch.cat(gate_parts, dim=-1).clamp(max=SWIGLU_LIMIT)
                up = torch.cat(up_parts, dim=-1).clamp(
                    min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT
                )
                activated = (F.silu(gate) * up).contiguous()
                down_parts = [
                    self._projection(
                        activated[
                            :,
                            rank
                            * LOCAL_INTERMEDIATE_SIZE : (rank + 1)
                            * LOCAL_INTERMEDIATE_SIZE,
                        ],
                        expert=expert,
                        projection="down_proj",
                        rank=rank,
                        handles=handles,
                    )
                    for rank in range(TP_SIZE)
                ]
                down = torch.stack(down_parts).sum(dim=0)
                weights = top_k_weights[token_idx, top_k_pos].to(down.dtype)
                observation.update(expert, down.float().norm(dim=-1), weights)
                final.index_add_(0, token_idx, down * weights[:, None])
                del gate_parts, up_parts, down_parts, gate, up, activated, down

        self.last_observation = observation
        if self.observation_sink is not None:
            self.observation_sink(self.layer, observation)
        return final.to(hidden_states.dtype)
