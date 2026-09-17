"""Opt-in online FP8 for native MTP routed experts; other weights stay native."""
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineMoEMethod

class MTPExpertOnlyFp8Config(Fp8Config):
    def __init__(self):
        super().__init__(is_checkpoint_fp8_serialized=False, activation_scheme='dynamic')
        self.selected_prefixes = []

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            allowed = 'model.layers.45.mlp.experts'
            if prefix != allowed:
                raise ValueError(f'Unexpected routed expert module: {prefix}')
            self.selected_prefixes.append(prefix)
            return Fp8PerTensorOnlineMoEMethod(layer=layer)
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None


def maybe_enable_mtp_expert_fp8(config):
    """Opt in only on an explicitly native, 288-expert GLM MTP draft config."""
    import os
    from vllm.config import replace
    if os.environ.get('GLM53_MTP_EXPERT_FP8', '0') != '1':
        return config
    if config.quant_config is not None:
        raise ValueError('MTP expert FP8 requires a native source draft')
    hf = config.model_config.hf_config
    if 'Glm5NextMTPModel' not in (hf.architectures or []) or hf.n_routed_experts != 288:
        raise ValueError('MTP expert FP8 only supports the original 288-expert GLM draft')
    return replace(config, quant_config=MTPExpertOnlyFp8Config())
