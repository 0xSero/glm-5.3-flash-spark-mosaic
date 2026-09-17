"""CPU actual installed candidate construction; no models or CUDA initialization."""
import json
from types import SimpleNamespace
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager, _is_compatible
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import _sparse_full_capture_request_sizes


def candidates(depth, slots, query, sizes, sparse):
    obj = CudaGraphManager.__new__(CudaGraphManager)
    obj.compilation_config = SimpleNamespace(cudagraph_capture_sizes=sizes, max_cudagraph_capture_size=max(sizes))
    obj.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    obj.max_num_reqs, obj.decode_query_len = slots, query
    obj.specialize_full_decode, obj.varlen_decode = False, False
    obj.full_capture_request_sizes = _sparse_full_capture_request_sizes(slots) if sparse else None
    obj.vllm_config = SimpleNamespace(speculative_config=None)
    obj.lora_capture_cases = [0]
    obj._capture_descs, obj._candidates = {}, {}
    obj._init_candidates()
    return obj._capture_descs.get(CUDAGraphMode.FULL, [])

# Reproduce the exact observed missing draft-decode graphs before fixing inputs.
assert candidates(2, 1, 1, [3], True) == []
rows = []
for depth in (1, 2, 3, 5):
    for slots in (1, 2, 4, 8):
        sizes = sorted({(depth + 1)*n for n in range(1, slots + 1)} |
                       (set(range(1, slots + 1)) if depth > 1 else set()))
        roles = [('target', depth+1, False), ('draft_prefill', depth+1, True)]
        if depth > 1:
            roles.append(('draft_decode', 1, True))
        for role, query, sparse in roles:
            descs = candidates(depth, slots, query, sizes, sparse)
            for n in range(1, slots+1):
                assert any(_is_compatible(d, n, n*query, query, 0, query) for d in descs), (depth, slots, role, n)
            rows.append({'depth': depth, 'slots': slots, 'role': role, 'capture_sizes': sizes,
                         'actual_descriptors': [dict(num_tokens=d.num_tokens, num_reqs=d.num_reqs,
                          uniform_token_count=d.uniform_token_count) for d in descs]})
assert not torch.cuda.is_initialized()
print(json.dumps({'state':'ACTUAL_INSTALLED_CANDIDATE_CONSTRUCTION_PASS',
    'negative_old_D2C1_reproduced':True,'configs':16,'role_checks':len(rows),
    'cuda_initialized':False,'rows':rows},indent=2))
