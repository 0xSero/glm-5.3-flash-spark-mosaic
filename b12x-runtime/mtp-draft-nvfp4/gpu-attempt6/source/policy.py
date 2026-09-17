"""Experimental SM121 B12x W4A16 draft experts; not wired into a serving image."""
from dataclasses import replace as replace_dataclass
import os

import torch
from vllm.config import replace
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend, select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.online.moe_base import OnlineMoEMethodBase
from vllm.model_executor.layers.quantization.online.nvfp4 import Nvfp4OnlineMoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
from vllm.platforms import current_platform


class B12xNvfp4DraftMethod(Nvfp4OnlineMoEMethod):
    """Reuse upstream weight quantization with a separately gated B12x backend.

    The upstream online method's constructor is SM100/TRTLLM-specific. This
    adapter does not modify it: it selects the existing SM12x B12x W4A16 path
    explicitly, preserving BF16 activations and the original weight quantizer.
    """

    def __init__(self, *, layer):
        if not (current_platform.is_cuda() and current_platform.is_device_capability((12, 1))):
            raise ValueError('Draft adapter is restricted to SM121')
        if os.environ.get('VLLM_B12X_MOE_FP4_FORCE_A16') != '1':
            raise ValueError('Explicit B12x BF16-activation mode is required')
        moe = layer.moe_config
        if (moe.hidden_dim, moe.intermediate_size, moe.tp_size) != (4096, 2048, 1):
            raise ValueError('Only native single-Spark GLM draft geometry is supported')
        if moe.in_dtype != torch.bfloat16 or moe.has_bias:
            raise ValueError('Expected BF16 activations and unbiased expert matrices')
        selected = replace_dataclass(moe, moe_backend='b12x')
        OnlineMoEMethodBase.__init__(self, selected)
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe, weight_key=kNvfp4Static, activation_key=None,
        )
        if self.nvfp4_backend != NvFp4MoeBackend.B12X:
            raise ValueError('B12x backend selection is mandatory')

    def process_weights_after_loading(self, layer):
        if os.environ.get('VLLM_B12X_MOE_FP4_FORCE_A16') != '1':
            raise ValueError('Activation policy changed after construction')
        return super().process_weights_after_loading(layer)


class MTPExpertOnlyNvfp4Config(Fp8Config):
    """Native-source loading metadata, with only layer45 experts overridden."""

    def __init__(self):
        super().__init__(is_checkpoint_fp8_serialized=False, activation_scheme='dynamic')
        self.selected_prefixes = []

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            if prefix != 'model.layers.45.mlp.experts':
                raise ValueError('Only the native MTP routed-expert prefix is allowed')
            self.selected_prefixes.append(prefix)
            return B12xNvfp4DraftMethod(layer=layer)
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None


def enable_for_probe(config):
    """Explicit experimental entry point; no environment-triggered auto-enable."""
    hf = config.model_config.hf_config
    if config.quant_config is not None:
        raise ValueError('A native source draft is required')
    if 'Glm5NextMTPModel' not in (hf.architectures or []) or hf.n_routed_experts != 288:
        raise ValueError('Only the original 288-expert GLM MTP draft is allowed')
    if os.environ.get('GLM53_MTP_EXPERT_FP8', '0') != '0':
        raise ValueError('The FP8 draft override must be disabled for this probe')
    for flag in ('VLLM_MTP_NVFP4_LM_HEAD', 'VLLM_MXFP8_LM_HEAD'):
        if os.environ.get(flag) != '0':
            raise ValueError(f'{flag} must be explicitly disabled to preserve native heads')
    return replace(config, quant_config=MTPExpertOnlyNvfp4Config())
