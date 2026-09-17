"""Load all 288 native MTP experts through the production loader; no serving claim."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from vllm.config import set_current_vllm_config
from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.layers.quantization.glm_mtp_expert_fp8 import MTPExpertOnlyFp8Config
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils
from policy import enable_for_probe, MTPExpertOnlyNvfp4Config

MODE = os.environ['GLM53_DRAFT_PROBE_MODE']
assert MODE in ('fp8', 'nvfp4')
assert os.environ['GLM53_MTP_EXPERT_FP8'] == ('1' if MODE == 'fp8' else '0')
for flag in ('VLLM_MTP_NVFP4_LM_HEAD', 'VLLM_MXFP8_LM_HEAD'):
    assert os.environ[flag] == '0'
torch.set_num_threads(4)
torch.set_default_dtype(torch.bfloat16)
torch.cuda.set_per_process_memory_fraction(32 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
c = EngineArgs(model='/model', tokenizer='/model', skip_tokenizer_init=True,
    dtype='bfloat16', quantization='exl3', max_model_len=262144, max_num_seqs=1,
    max_num_batched_tokens=2048, kv_cache_dtype='fp8_ds_mla', block_size=256,
    attention_backend='B12X', additional_config={'kda_prefill_backend': 'b12x'},
    enable_prefix_caching=False, speculative_config={'method': 'mtp', 'model': '/mtp',
    'num_speculative_tokens': 1, 'attention_backend': 'B12X'}).create_engine_config()
expected_class = MTPExpertOnlyFp8Config if MODE == 'fp8' else MTPExpertOnlyNvfp4Config
tracked = []
original_track = DefaultModelLoader.track_weights_loading
original_get = eagle_utils.get_model


def track(self, model, loaded):
    original_track(self, model, loaded)
    tracked.append({'runtime_parameters': len(dict(model.named_parameters())),
                    'loaded_parameters': len(loaded), 'strict_completeness_pass': True})


def get_model(*, vllm_config, model_config):
    assert vllm_config.model_config is model_config is c.speculative_config.draft_model_config
    if MODE == 'nvfp4':
        vllm_config = enable_for_probe(vllm_config)
    assert isinstance(vllm_config.quant_config, expected_class)
    return original_get(vllm_config=vllm_config, model_config=model_config)


DefaultModelLoader.track_weights_loading = track
eagle_utils.get_model = get_model
init_distributed_environment(world_size=1, rank=0, local_rank=0, backend='gloo',
    distributed_init_method='file:///tmp/glm-full-draft-load-dist')
with set_current_vllm_config(c):
    ensure_model_parallel_initialized(1, 1)
    model = eagle_utils.load_eagle_model(SimpleNamespace(model=SimpleNamespace()), c)
assert tracked and isinstance(model.quant_config, expected_class)
assert model.quant_config.selected_prefixes == ['model.layers.45.mlp.experts']
params = dict(model.named_parameters())
assert all(p.device.type == 'cuda' for p in params.values())
expert_prefix = 'model.layers.45.mtp_block.mlp.experts.routed_experts.'
baseline_path = Path('/proof/native-parameter-baseline.json')
baseline = json.loads(baseline_path.read_text())
native = {}
for name, expected in baseline.items():
    if name.startswith(expert_prefix):
        continue
    if name.endswith('indexer.wk_weights_proj.weight'):
        names = [name.replace('wk_weights_proj', 'wk'), name.replace('wk_weights_proj', 'weights_proj')]
        values = [params[n] for n in names]
        assert all(str(v.dtype) == expected['dtype'] for v in values)
        assert [sum(v.shape[0] for v in values), values[0].shape[1]] == expected['shape']
        native[name] = {'runtime_names': names, 'native_dtype': True}
        continue
    param = params[name]
    if name.endswith('index_kpool_compress_ape'):
        # Current source keeps BF16; the historical baseline used an FP32 upcast.
        assert param.dtype == torch.bfloat16 and expected['dtype'] == 'torch.float32'
    else:
        assert str(param.dtype) == expected['dtype'], (name, param.dtype, expected)
    native[name] = {'shape': list(param.shape), 'dtype': str(param.dtype)}

index_path = Path('/mtp/model.safetensors.index.json')
index = json.loads(index_path.read_text())['weight_map']
assert len(index) == 891
comparisons = []
for source, target in [('model.language_model.embed_tokens.weight', 'model.embed_tokens.weight'),
                       ('lm_head.weight', 'model.layers.45.shared_head.head.weight')]:
    with safe_open('/mtp/' + index[source], framework='pt', device='cpu') as f:
        view = f.get_slice(source)
        actual = params[target]
        assert list(actual.shape) == view.get_shape() and actual.dtype == torch.bfloat16
        for row in range(0, actual.shape[0], 4096):
            assert torch.equal(actual[row:row+4096].detach().cpu(), view[row:row+4096])
    comparisons.append({'source': source, 'runtime': target, 'all_values_exact': True})
for name in index:
    if name.endswith('index_kpool_compress_ape'):
        with safe_open('/mtp/' + index[name], framework='pt', device='cpu') as f:
            source = f.get_tensor(name)
        targets = [p for n, p in params.items() if n.endswith('index_kpool_compress_ape')]
        assert len(targets) == 1 and torch.equal(targets[0].detach().cpu(), source)

prepared = {}
if MODE == 'nvfp4':
    from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
    routed = model.get_submodule(expert_prefix.rstrip('.'))
    assert type(routed.quant_method).__name__ == 'B12xNvfp4DraftMethod'
    kernel = routed.quant_method.moe_kernel.fused_experts
    assert isinstance(kernel, B12xExperts)
    assert kernel._quant_mode == 'w4a16'
    values = kernel._prepared()
    fields = [f.name for f in dataclasses.fields(values)] if dataclasses.is_dataclass(values) else vars(values)
    for name in fields:
        value = getattr(values, name)
        if isinstance(value, torch.Tensor):
            prepared[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                'bytes': value.numel()*value.element_size()}
    assert values.w1_fp4.dtype == values.w2_fp4.dtype == torch.uint8
else:
    for suffix in ('w13_weight', 'w2_weight'):
        assert params[expert_prefix+suffix].dtype == torch.float8_e4m3fn
torch.cuda.synchronize()
report = {'state': 'FULL_DRAFT_LOAD_PASS_SERVING_PENDING', 'draft_expert_mode': MODE,
    'draft_experts': 288, 'strict_tracking': tracked, 'native_index_keys': len(index),
    'native_parameter_audit': native, 'head_embedding_comparisons': comparisons,
    'native_parameter_baseline_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    'native_index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
    'prepared_tensor_metadata': prepared, 'cuda_allocated_bytes': torch.cuda.memory_allocated(),
    'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
    'runtime_parameter_bytes': sum(p.numel()*p.element_size() for p in params.values()),
    'native_parameter_bytes': sum(p.numel()*p.element_size() for n,p in params.items() if not n.startswith(expert_prefix)),
    'target_weights_loaded': False, 'kv_allocated': False, 'forward_executed': False,
    'full_serving_acceptance': False}
Path('/out/full-load.json').write_text(json.dumps(report, indent=2)+'\n')
print('FULL_DRAFT_LOAD_PASS', json.dumps(report), flush=True)
