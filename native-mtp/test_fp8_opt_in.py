import os, dataclasses, types, torch
from vllm.model_executor.layers.quantization.glm_mtp_expert_fp8 import maybe_enable_mtp_expert_fp8, MTPExpertOnlyFp8Config
@dataclasses.dataclass
class Config:
    model_config: object
    quant_config: object = None
hf=types.SimpleNamespace(architectures=['Glm5NextMTPModel'],n_routed_experts=288)
c=Config(types.SimpleNamespace(hf_config=hf))
os.environ['GLM53_MTP_EXPERT_FP8']='0'
assert maybe_enable_mtp_expert_fp8(c) is c
os.environ['GLM53_MTP_EXPERT_FP8']='1'
d=maybe_enable_mtp_expert_fp8(c)
assert d is not c and d.model_config is c.model_config and c.quant_config is None
assert isinstance(d.quant_config,MTPExpertOnlyFp8Config)
for bad in [d,Config(types.SimpleNamespace(hf_config=types.SimpleNamespace(architectures=['Glm5NextMTPModel'],n_routed_experts=256))),Config(types.SimpleNamespace(hf_config=types.SimpleNamespace(architectures=['Other'],n_routed_experts=288)))]:
    try:maybe_enable_mtp_expert_fp8(bad)
    except ValueError:pass
    else:raise AssertionError('invalid config accepted')
assert not torch.cuda.is_initialized()
print('OPT_IN_CPU_PASS: default unchanged; native288 MTP only; existing quantization rejected; no CUDA initialization')
