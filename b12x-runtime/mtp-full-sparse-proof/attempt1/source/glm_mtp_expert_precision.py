"""Opt-in native GLM MTP expert NVFP4, BF16 activations, packed B12x routes.

Only the native layer-45 draft config receives this policy. No global backend
registration, source tensor edits, target quantization, or compile monkeypatch.
See SOURCE_PINS.json and README.md for the qualified component and pending gates.
"""
from dataclasses import replace as dc_replace
import os

import torch
from vllm.config import replace
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.b12x import (
    B12xExperts, _is_current_stream_capturing, _require_b12x_fused_moe,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend, select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.glm_mtp_expert_fp8 import maybe_enable_mtp_expert_fp8
from vllm.model_executor.layers.quantization.online.moe_base import OnlineMoEMethodBase
from vllm.model_executor.layers.quantization.online.nvfp4 import Nvfp4OnlineMoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
from vllm.platforms import current_platform


def _flag(name):
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def _native_heads_and_a16():
    for flag in ("VLLM_MTP_NVFP4_LM_HEAD", "VLLM_MXFP8_LM_HEAD"):
        if os.environ.get(flag) != "0":
            raise ValueError(f"{flag}=0 must be explicit to preserve native heads")
    if os.environ.get("VLLM_B12X_MOE_FP4_FORCE_A16") != "1":
        raise ValueError("VLLM_B12X_MOE_FP4_FORCE_A16=1 is required")


class PackedDraftB12xExperts(B12xExperts):
    """Cache only packed plans; leave workspace_shapes/apply unchanged.

    The first uncaptured lookup resolves the ordinary policy, copies its context
    with ONLY route_mode changed, and constructs the final plan. The provisional
    plan is never cached, warmed, bound, or launched. No captured plan is replaced.
    Upstream run_w4a16_moe additionally enforces its compiled-launch cache during
    capture (kernel.py:12452), even after our per-capacity warmup admission.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._packed_warmed_plans = {}

    def _plan(self, *, tokens, topk, activation, apply_router_weight_on_input=False):
        key = (self._plan_capacity(tokens), int(topk), activation,
               bool(apply_router_weight_on_input))
        plan = self._plans.get(key)
        capturing = _is_current_stream_capturing()
        if plan is not None:
            if capturing and self._packed_warmed_plans.get(key) is not plan:
                raise RuntimeError("Packed MTP plan requires completed eager warmup before capture")
            return plan
        if capturing:
            raise RuntimeError("Packed MTP plans must be created before CUDA capture")
        if self._quant_mode != "w4a16" or key[3] or topk != 8:
            raise ValueError("Packed draft requires W4A16, top-8, output route weighting")
        prepared = self._prepared()
        if prepared.plan.source_format != "modelopt_nvfp4":
            raise ValueError("Packed draft requires native-source NVFP4")
        limit, alpha, beta = self._swiglu_params(activation)
        if limit != 10. or str(activation.value) != "silu":
            raise ValueError("Only native GLM SILU clamp=10 is qualified")
        fused_moe = _require_b12x_fused_moe()
        caps = fused_moe.Caps(max_tokens=key[0], num_topk=key[1],
            device=prepared.w1_fp4.device, weight_plan=prepared.plan,
            core_token_counts=(key[0],), route_num_experts=0,
            quant_mode="w4a16", apply_router_weight_on_input=False,
            swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta,
            frozen=True, w4a16_fast_math=True)
        initial = fused_moe.plan(caps)
        resolution = initial.launch_plan.policy_resolution
        if resolution is None or resolution.config.backend != "w4a16":
            raise ValueError("Missing W4A16 policy resolution")
        context = caps.policy_context.with_override("moe.decode",
            dc_replace(resolution.config, w4a16_route_mode="packed"))
        plan = fused_moe.plan(dc_replace(caps, policy_context=context))
        if (plan.launch_plan.policy_resolution.config.w4a16_route_mode != "packed"
                or plan.caps.w4a16_fast_math is not True
                or plan.caps.weight_plan is not caps.weight_plan):
            raise ValueError("Packed route policy changed precision or weights")
        self._plans[key] = plan
        return plan

    def warmup_launches(self, layer, *, token_counts):
        if _is_current_stream_capturing():
            raise RuntimeError("Packed MTP warmup cannot execute during capture")
        counts = sorted({int(t) for t in token_counts if int(t) > 0})
        self._register_plan_capacities(counts)
        launches = 0
        # Upstream deduplicates regimes, but our admission must prove each plan.
        # Warm one capacity at a time, retaining upstream tensor lifetime+sync.
        for tokens in counts:
            launches += super().warmup_launches(layer, token_counts=(tokens,))
            key = (self._plan_capacity(tokens), int(self.moe_config.experts_per_token),
                   layer.activation, bool(layer.apply_router_weight_on_input))
            self._packed_warmed_plans[key] = self._plans[key]
        return launches

    def packed_plan_metadata(self):
        """Read after eager warmup; logical plan evidence, not a physical trace."""
        if _is_current_stream_capturing():
            raise RuntimeError("Read packed plan metadata outside capture")
        return [{"capacity": key[0], "topk": key[1], "activation": key[2].value,
                 "apply_router_weight_on_input": key[3],
                 "route_mode": plan.launch_plan.policy_resolution.config.w4a16_route_mode,
                 "quant_mode": plan.caps.quant_mode,
                 "fast_math": plan.caps.w4a16_fast_math,
                 "scratch_bytes": plan.layout.total_nbytes,
                 "core_token_counts": list(plan.caps.core_token_counts),
                 "eager_warmup_completed": self._packed_warmed_plans.get(key) is plan,
                 "physical_launch_flags_observed": False}
                for key, plan in self._plans.items()]


class B12xNvfp4DraftMethod(Nvfp4OnlineMoEMethod):
    """Same quantizer/backend as the passed component; draft-only packed class."""
    def __init__(self, *, layer):
        _native_heads_and_a16()
        if not (current_platform.is_cuda() and current_platform.is_device_capability((12, 1))):
            raise ValueError("Native MTP NVFP4 is restricted to SM121")
        moe = layer.moe_config
        if (moe.hidden_dim, moe.intermediate_size, moe.tp_size, moe.num_experts,
                moe.experts_per_token) != (4096, 2048, 1, 288, 8):
            raise ValueError("Expected unpruned native GLM MTP geometry")
        parallel = moe.moe_parallel_config
        if parallel.ep_size != 1 or parallel.dp_size != 1 or parallel.enable_eplb:
            raise ValueError("Only single-Spark TP1/EP1/DP1 without EPLB is supported")
        if moe.in_dtype != torch.bfloat16 or moe.has_bias:
            raise ValueError("Expected BF16 activations and unbiased expert matrices")
        OnlineMoEMethodBase.__init__(self, dc_replace(moe, moe_backend="b12x"))
        self.nvfp4_backend, selected = select_nvfp4_moe_backend(
            config=self.moe, weight_key=kNvfp4Static, activation_key=None)
        if self.nvfp4_backend != NvFp4MoeBackend.B12X or selected is not B12xExperts:
            raise ValueError("Expected pinned B12x W4A16 backend")
        self.experts_cls = PackedDraftB12xExperts

    def process_weights_after_loading(self, layer):
        _native_heads_and_a16()
        return super().process_weights_after_loading(layer)


class MTPExpertOnlyNvfp4Config(Fp8Config):
    def __init__(self):
        super().__init__(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
        self.selected_prefixes = []

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            if prefix != "model.layers.45.mlp.experts":
                raise ValueError("Only native MTP routed experts may use NVFP4")
            method = B12xNvfp4DraftMethod(layer=layer)
            self.selected_prefixes.append(prefix)
            return method
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None


def maybe_enable_mtp_expert_precision(config):
    fp8, nvfp4 = _flag("GLM53_MTP_EXPERT_FP8"), _flag("GLM53_MTP_EXPERT_NVFP4")
    if fp8 and nvfp4:
        raise ValueError("GLM53_MTP_EXPERT_FP8 and GLM53_MTP_EXPERT_NVFP4 are mutually exclusive")
    if not nvfp4:
        return maybe_enable_mtp_expert_fp8(config)
    _native_heads_and_a16()
    hf = config.model_config.hf_config
    if (config.quant_config is not None
            or (hf.architectures or []) != ["Glm5NextMTPModel"]
            or hf.n_routed_experts != 288):
        raise ValueError("NVFP4 requires the separate native 288-expert GLM draft config")
    return replace(config, quant_config=MTPExpertOnlyNvfp4Config())
