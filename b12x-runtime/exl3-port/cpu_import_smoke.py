#!/usr/bin/env python3
"""Run inside the newly compiled image without exposing GPUs. No stub imports."""
import hashlib
import inspect
import json
from pathlib import Path
import torch
import vllm
import exllamav3_ext
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod, load_linear_exl3_cls

assert not torch.cuda.is_initialized(), 'run without GPU initialization'
assert get_quantization_config('exl3') is Exl3Config
assert not inspect.isabstract(Exl3MoEMethod)
for bits in (2,3):
    config=Exl3Config.from_config({'bits':bits,'rank_stacked_tp':4,'quant_method':'exl3'})
    assert config.bits==bits and config.rank_stacked_tp==4
    assert config.get_name()=='exl3'
linear=load_linear_exl3_cls()
assert linear.__module__=='exllamav3.modules.quant.exl3'
assert callable(linear.get_weight_tensor)
for symbol in ('exl3_moe','exl3_moe_max_concurrency'):
    assert callable(getattr(exllamav3_ext,symbol,None)), symbol
assert not torch.cuda.is_initialized()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

print(json.dumps({'state':'CPU_REAL_IMPORT_PASS_GPU_UNTESTED','torch':torch.__version__,
    'vllm':vllm.__version__,'wrapper_sha256':digest(inspect.getfile(Exl3Config)),
    'extension_sha256':digest(exllamav3_ext.__file__),
    'linear_sha256':digest(inspect.getfile(linear)), 'cuda_initialized':False}))
