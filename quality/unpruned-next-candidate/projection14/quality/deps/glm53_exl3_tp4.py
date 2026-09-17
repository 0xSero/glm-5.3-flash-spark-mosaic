"""Runtime primitives for the custom GLM-5.3 selective EXL3 TP4 artifact.

This module implements the routed-expert surface only. A serving integration
must install ``TPRankEXL3Experts`` in each routed GLM layer while retaining the
source-precision router and shared expert. Runtime acceptance remains a
separate gate from importing this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open

from exllamav3.modules.quant import LinearEXL3


HIDDEN = 4096
LOCAL_INTERMEDIATE = 512
EXPERTS = 288
TP = 4
SWIGLU_LIMIT = 10.0


def _packed_prefix(layer: int, expert: int, projection: str, rank: int) -> str:
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}.rank{rank}"
    )


def _load_linear(
    handle: Any,
    prefix: str,
    in_features: int,
    out_features: int,
    out_dtype: torch.dtype,
) -> LinearEXL3:
    tensors = {suffix: handle.get_tensor(f"{prefix}.{suffix}") for suffix in ("suh", "svh", "trellis", "mcg")}
    return LinearEXL3(
        config=None,
        in_features=in_features,
        out_features=out_features,
        suh=tensors["suh"],
        svh=tensors["svh"],
        trellis=tensors["trellis"],
        mcg=tensors["mcg"],
        out_dtype=out_dtype,
        key=prefix,
    )


class TPRankEXL3Experts(nn.Module):
    """One tensor-parallel rank of all 288 routed experts for one GLM layer."""

    def __init__(
        self,
        artifact_root: str | Path,
        layer: int,
        tp_rank: int,
        device: torch.device | str,
        process_group: Any | None = None,
        out_dtype: torch.dtype = torch.float16,
        swiglu_limit: float = SWIGLU_LIMIT,
    ) -> None:
        super().__init__()
        if not 3 <= layer <= 44:
            raise ValueError(f"layer outside routed scope: {layer}")
        if not 0 <= tp_rank < TP:
            raise ValueError(f"TP rank outside [0, {TP}): {tp_rank}")
        self.artifact_root = Path(artifact_root)
        self.layer = layer
        self.tp_rank = tp_rank
        self.device = torch.device(device)
        self.process_group = process_group
        self.out_dtype = out_dtype
        self.swiglu_limit = float(swiglu_limit)
        self.num_experts = EXPERTS
        self.gate: list[LinearEXL3 | None] = [None] * EXPERTS
        self.up: list[LinearEXL3 | None] = [None] * EXPERTS
        self.down: list[LinearEXL3 | None] = [None] * EXPERTS
        self._load()

    def _load(self) -> None:
        for worker in range(TP):
            path = self.artifact_root / "layers" / f"layer-{self.layer:02d}-part-{worker}.safetensors"
            if not path.is_file():
                raise FileNotFoundError(path)
            with safe_open(path, framework="pt", device=str(self.device)) as handle:
                for expert in range(worker, EXPERTS, TP):
                    self.gate[expert] = _load_linear(
                        handle,
                        _packed_prefix(self.layer, expert, "gate_proj", self.tp_rank),
                        HIDDEN,
                        LOCAL_INTERMEDIATE,
                        self.out_dtype,
                    )
                    self.up[expert] = _load_linear(
                        handle,
                        _packed_prefix(self.layer, expert, "up_proj", self.tp_rank),
                        HIDDEN,
                        LOCAL_INTERMEDIATE,
                        self.out_dtype,
                    )
                    self.down[expert] = _load_linear(
                        handle,
                        _packed_prefix(self.layer, expert, "down_proj", self.tp_rank),
                        LOCAL_INTERMEDIATE,
                        HIDDEN,
                        self.out_dtype,
                    )
        if any(module is None for modules in (self.gate, self.up, self.down) for module in modules):
            raise RuntimeError(f"incomplete EXL3 expert coverage for layer {self.layer} rank {self.tp_rank}")

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        params: dict | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != HIDDEN:
            raise ValueError(f"expected [tokens, {HIDDEN}] hidden states, got {tuple(hidden_states.shape)}")
        if top_k_index.shape != top_k_weights.shape or top_k_index.shape[0] != hidden_states.shape[0]:
            raise ValueError("top-k routing tensors do not match hidden states")
        params = {} if params is None else params
        final = torch.zeros_like(hidden_states, dtype=self.out_dtype)
        mask = F.one_hot(top_k_index, num_classes=EXPERTS).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()
        for expert in hit:
            top_k_pos, token_idx = torch.where(mask[expert])
            x = hidden_states[token_idx].to(self.out_dtype).contiguous()
            gate = self.gate[expert].forward(x, params).clamp(max=self.swiglu_limit)
            up = self.up[expert].forward(x, params).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            activated = (F.silu(gate) * up).contiguous()
            current = self.down[expert].forward(activated, params)
            current = current * top_k_weights[token_idx, top_k_pos, None].to(current.dtype)
            final.index_add_(0, token_idx, current)
        if self.process_group is not None:
            if not dist.is_initialized():
                raise RuntimeError("a process group was provided but torch.distributed is not initialized")
            dist.all_reduce(final, op=dist.ReduceOp.SUM, group=self.process_group)
        return final

    def unload(self) -> None:
        for modules in (self.gate, self.up, self.down):
            for module in modules:
                if module is not None:
                    module.unload()
            modules.clear()
