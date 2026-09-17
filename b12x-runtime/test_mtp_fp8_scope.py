"""CPU-only opt-in policy scope; does not claim FP8 GPU quantization ran."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import glm_mtp_expert_fp8 as policy


@dataclasses.dataclass
class Config:
    model_config: object
    quant_config: object = None


def config(architecture='Glm5NextMTPModel', experts=288, quant=None):
    return Config(SimpleNamespace(hf_config=SimpleNamespace(
        architectures=[architecture], n_routed_experts=experts)), quant)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
native = config()
with patch.dict(os.environ, {'GLM53_MTP_EXPERT_FP8': '0'}):
    assert policy.maybe_enable_mtp_expert_fp8(native) is native
with patch.dict(os.environ, {'GLM53_MTP_EXPERT_FP8': '1'}):
    enabled = policy.maybe_enable_mtp_expert_fp8(native)
    assert enabled is not native and enabled.model_config is native.model_config
    assert native.quant_config is None
    assert isinstance(enabled.quant_config, policy.MTPExpertOnlyFp8Config)
    for bad in (enabled, config(experts=256), config(architecture='Glm5NextForConditionalGeneration')):
        try:
            policy.maybe_enable_mtp_expert_fp8(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe source configuration accepted')


class Dense(LinearBase):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, x):
        raise AssertionError('CPU scope test never executes a dense kernel')


quant = enabled.quant_config
assert isinstance(quant.get_quant_method(Dense(), 'model.layers.45.self_attn.q_proj'),
                  UnquantizedLinearMethod)
assert quant.get_quant_method(torch.nn.Embedding(2, 2), 'model.embed_tokens') is None
experts = RoutedExperts.__new__(RoutedExperts)
torch.nn.Module.__init__(experts)
for wrong in ('model.layers.44.mlp.experts', 'model.layers.45.mlp.shared_experts',
              'model.layers.45.mtp_block.mlp.experts'):
    try:
        quant.get_quant_method(experts, wrong)
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong expert prefix accepted')
# The backend constructor selects a GPU implementation. Mock only that final
# call to test dispatch without allocating GPU state; real load is a later gate.
with patch.object(policy, 'Fp8PerTensorOnlineMoEMethod') as constructor:
    result = quant.get_quant_method(experts, 'model.layers.45.mlp.experts')
    constructor.assert_called_once_with(layer=experts)
    assert result is constructor.return_value
assert quant.selected_prefixes == ['model.layers.45.mlp.experts']
assert not torch.cuda.is_initialized()
report = {'state': 'CPU_OPT_IN_PRECISION_SCOPE_PASS_GPU_LOAD_PENDING',
          'default_bf16_unchanged': True, 'native_288_draft_only': True,
          'existing_quantization_rejected': True, 'target_architecture_rejected': True,
          'native_dense_and_embedding_methods': True, 'exact_expert_prefix_only': True,
          'gpu_backend_constructor_mocked_for_scope_only': True,
          'cuda_initialized': False}
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
