#!/usr/bin/env python3
"""Real CPU import/ABI gate. This does not claim a B12x CUDA kernel ran."""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
import torch
assert torch.__version__ == '2.13.0+cu130'
assert not torch.cuda.is_initialized()
assert os.environ.get('VLLM_MXFP8_LM_HEAD') == '0'
assert os.environ.get('VLLM_MTP_NVFP4_LM_HEAD') == '0'
# Jovian migrated CUDA ops to the stable-libtorch modules. Legacy vllm._C
# exists for CPU/ROCm, so requiring that filename would reject a valid build.
native_vllm = [importlib.import_module(name) for name in
               ('vllm._C_stable_libtorch', 'vllm._moe_C_stable_libtorch')]
from vllm.model_executor.layers.quantization import get_quantization_config
assert get_quantization_config('exl3')().get_name() == 'exl3'
from vllm.model_executor.layers.quantization.exl3 import load_linear_exl3_cls
assert load_linear_exl3_cls().__module__ == 'exllamav3.modules.quant.exl3'
import exllamav3_ext
for symbol in ('reconstruct', 'exl3_moe', 'exl3_moe_max_concurrency'):
    assert callable(getattr(exllamav3_ext, symbol))
modules = ['b12x.norm.mhc', 'b12x.attention.sparse_mla',
           'b12x.sequence.gdn_decode', 'b12x.sequence.kda_prefill']
for name in modules:
    importlib.import_module(name)
from vllm.models.glm5next.nvidia import model, mtp, multimodal
assert not torch.cuda.is_initialized()
report = {'state': 'CPU_IMPORT_ABI_PASSED_GPU_VALIDATION_PENDING',
          'packages': {n: metadata.version(n) for n in
              ['torch', 'vllm', 'exllamav3', 'b12x', 'transformers',
               'flashinfer-python', 'nvidia-cutlass-dsl', 'apache-tvm-ffi']},
          'sources': json.loads(Path('/opt/source-pins.json').read_text()),
          'runtime_base_image_reference': os.environ.get('GLM53_RUNTIME_BASE_IMAGE'),
          'native_extensions': {}, 'cuda_initialized': False,
          'protected_head_policy': {name: os.environ.get(name) for name in ('VLLM_MXFP8_LM_HEAD','VLLM_MTP_NVFP4_LM_HEAD')},
          'installed_distributions': dict(sorted(
              (d.metadata['Name'], d.version) for d in metadata.distributions()
              if d.metadata.get('Name')))}
for module in (*native_vllm, exllamav3_ext):
    path = Path(module.__file__)
    report['native_extensions'][module.__name__] = {
        'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
import vllm
base = Path(vllm.__file__).parent
report['python_overlay_sha256'] = {
    rel: hashlib.sha256((base/rel).read_bytes()).hexdigest() for rel in (
        'config/model.py',
        'model_executor/layers/quantization/exl3.py',
        'model_executor/layers/quantization/glm_mtp_expert_fp8.py',
        'models/glm5next/nvidia/mtp.py',
        'v1/spec_decode/llm_base_proposer.py',
        'v1/worker/gpu/spec_decode/eagle/utils.py')}
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
