"""Actual-import policy tests; GPU constructor/dispatch are explicitly mocked."""
import dataclasses
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
import policy


@dataclasses.dataclass
class Config:
    model_config: object
    quant_config: object = None


def config(architecture='Glm5NextMTPModel', experts=288, quant=None):
    return Config(SimpleNamespace(hf_config=SimpleNamespace(
        architectures=[architecture], n_routed_experts=experts)), quant)


@dataclasses.dataclass
class Geometry:
    hidden_dim: int = 4096
    intermediate_size: int = 2048
    tp_size: int = 1
    in_dtype: object = torch.bfloat16
    has_bias: bool = False
    moe_backend: str = 'auto'


def rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Unsafe configuration accepted')


native = config()
with patch.dict(os.environ, {'GLM53_MTP_EXPERT_FP8': '0',
                            'VLLM_MTP_NVFP4_LM_HEAD': '0', 'VLLM_MXFP8_LM_HEAD': '0'}):
    enabled = policy.enable_for_probe(native)
    assert enabled is not native and native.quant_config is None
    for bad in (enabled, config(experts=256), config(architecture='Glm5NextForConditionalGeneration')):
        rejected(lambda: policy.enable_for_probe(bad))
    for flag in ('VLLM_MTP_NVFP4_LM_HEAD', 'VLLM_MXFP8_LM_HEAD'):
        with patch.dict(os.environ, {flag: '1'}):
            rejected(lambda: policy.enable_for_probe(native))
with patch.dict(os.environ, {'GLM53_MTP_EXPERT_FP8': '1'}):
    rejected(lambda: policy.enable_for_probe(native))


class Dense(LinearBase):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, x):
        raise AssertionError('No kernels run in CPU scope tests')


q = enabled.quant_config
assert isinstance(q.get_quant_method(Dense(), 'model.layers.45.self_attn.q_proj'), UnquantizedLinearMethod)
assert isinstance(q.get_quant_method(Dense(), 'model.layers.45.mlp.shared_experts.down_proj'), UnquantizedLinearMethod)
assert q.get_quant_method(torch.nn.Embedding(2, 2), 'model.embed_tokens') is None
experts = RoutedExperts.__new__(RoutedExperts)
torch.nn.Module.__init__(experts)
for prefix in ('model.layers.44.mlp.experts', 'model.layers.45.mlp.shared_experts'):
    rejected(lambda: q.get_quant_method(experts, prefix))
with patch.object(policy, 'B12xNvfp4DraftMethod') as backend:
    assert q.get_quant_method(experts, 'model.layers.45.mlp.experts') is backend.return_value
    backend.assert_called_once_with(layer=experts)

geometry = Geometry()
layer = SimpleNamespace(moe_config=geometry)
def initialize(method, selected):
    method.moe = selected
with patch.object(policy, 'current_platform') as platform, \
        patch.object(policy.OnlineMoEMethodBase, '__init__', initialize), \
        patch.object(policy, 'select_nvfp4_moe_backend', return_value=(policy.NvFp4MoeBackend.B12X, object)) as select, \
        patch.dict(os.environ, {'VLLM_B12X_MOE_FP4_FORCE_A16': '1'}):
    platform.is_cuda.return_value = True
    platform.is_device_capability.return_value = True
    method = policy.B12xNvfp4DraftMethod(layer=layer)
    assert method.moe is not geometry and geometry.moe_backend == 'auto'
    assert method.moe.moe_backend == 'b12x'
    assert select.call_args.kwargs['activation_key'] is None
    platform.is_device_capability.return_value = False
    rejected(lambda: policy.B12xNvfp4DraftMethod(layer=layer))
    platform.is_device_capability.return_value = True
    with patch.dict(os.environ, {'VLLM_B12X_MOE_FP4_FORCE_A16': '0'}):
        rejected(lambda: policy.B12xNvfp4DraftMethod(layer=layer))
    for bad in (Geometry(hidden_dim=5120), Geometry(tp_size=2), Geometry(has_bias=True), Geometry(in_dtype=torch.float16)):
        rejected(lambda: policy.B12xNvfp4DraftMethod(layer=SimpleNamespace(moe_config=bad)))
assert not torch.cuda.is_initialized()
print(json.dumps({'state': 'CPU_SCOPE_PASS_GPU_EXECUTION_UNTESTED', 'native_source_draft_only': True,
    'target_and_protected_paths_preserved': True, 'geometry_platform_activation_guards_pass': True,
    'copied_backend_config': True, 'backend_selection_mocked': True, 'cuda_initialized': False}))
