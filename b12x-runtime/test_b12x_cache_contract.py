"""Installed-source cache contract; runtime capacity is measured separately."""
import argparse
import json
from pathlib import Path

import torch
from vllm.model_executor.layers.attention.mla_attention import (
    _canonicalize_sparse_mla_kv_cache_dtype,
)
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xGLM5NextMLASparseBackend as Backend,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
for value in ('auto', 'fp8', 'fp8_e4m3', 'fp8_ds_mla'):
    assert _canonicalize_sparse_mla_kv_cache_dtype(Backend, value) == 'fp8_ds_mla'
assert _canonicalize_sparse_mla_kv_cache_dtype(Backend, 'nvfp4_ds_mla') == 'nvfp4_ds_mla'
pages = []
for block_size in (64, 256, 1024):
    spec = Backend.customize_spec(MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=512,
        dtype=torch.uint8, cache_dtype_str='fp8_ds_mla'))
    assert spec.state_content_bytes == 528
    assert spec.page_tail_bytes_per_token == 33
    assert spec.alignment == 8448
    pages.append({'block_size': block_size, 'page_bytes_per_layer': spec.page_size_bytes})
assert not torch.cuda.is_initialized()
report = {'state': 'CPU_PACKED_CACHE_CONTRACT_PASS_CAPACITY_UNMEASURED',
          'requested_cache_dtype': 'fp8_ds_mla', 'backend': Backend.get_name(),
          'record_bytes': 528, 'index_tail_bytes_per_token': 33,
          'alignment_bytes': 8448,
          'supported_layouts': [str(v) for v in Backend.supported_kv_cache_layouts()],
          'example_page_geometry_only': pages,
          'actual_runtime_block_size': None, 'actual_runtime_token_capacity': None,
          'cuda_initialized': False}
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
