"""Eight native MTP experts: real NVFP4/B12x execution, not full draft acceptance.

Run only on an exclusively assigned Spark after the quality job finishes.
The manual component loader does not qualify the full 288-expert online loader.
"""
import contextlib
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from vllm.config import set_current_vllm_config
from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.utils import replace_parameter
from vllm.models.glm5next.nvidia.model import Glm5NextMoE

from policy import MTPExpertOnlyNvfp4Config

OUT = Path('/out')
OUT.mkdir(exist_ok=True)
MODEL = Path('/model')
IDS = list(range(8))
PREFIX = 'model.language_model.layers.45.mlp.'
LIMIT = 2 * 1024**3
report = {'state': 'STARTED', 'full_loader_acceptance': False,
          'model_quality_claim': False, 'speed_claim': False,
          'original_expert_ids': IDS, 'cells': [], 'source_tensors': {}}


def save():
    tmp = OUT / 'gpu-component.json.tmp'
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(OUT / 'gpu-component.json')


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def unpack(packed, scales, global_scale):
    """Independent CPU E2M1/group16 reference; no B12x dequantization code."""
    assert packed.device.type == scales.device.type == 'cpu'
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                        -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
    n, half_k = packed.shape
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(n, half_k * 2)
    values = lut[codes.long()].reshape(n, -1, 16)
    return (values * scales.float().unsqueeze(-1) * global_scale.float()).reshape(n, -1)


def relative_l2(actual, expected):
    return float(torch.linalg.vector_norm(actual.float() - expected.float()) /
                 torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))


save()
started = time.monotonic()
torch.set_num_threads(4)
torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(1947)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
report['reference_tf32_enabled'] = False
torch.cuda.set_per_process_memory_fraction(LIMIT / torch.cuda.get_device_properties(0).total_memory)
index_path = MODEL / 'model.safetensors.index.json'
index = json.loads(index_path.read_text())['weight_map']
report['source_index_sha256'] = hashlib.sha256(index_path.read_bytes()).hexdigest()
report['source_config_sha256'] = hashlib.sha256((MODEL / 'config.json').read_bytes()).hexdigest()
for script in ('policy.py', 'gpu_component.py'):
    report[script + '_sha256'] = hashlib.sha256((Path(__file__).parent / script).read_bytes()).hexdigest()
c = EngineArgs(model=str(MODEL), tokenizer=str(MODEL), skip_tokenizer_init=True,
               dtype='bfloat16', quantization='exl3', max_model_len=4096,
               max_num_seqs=8, max_num_batched_tokens=32,
               enable_prefix_caching=False).create_engine_config()
original = c.model_config.hf_config.get_text_config()
assert original.n_routed_experts == 288
cfg = copy.copy(original)
cfg.n_routed_experts = len(IDS)
assert cfg.num_experts_per_token == 8 and cfg.n_group == 1
assert cfg.scoring_func == 'sigmoid' and cfg.moe_renormalize
assert cfg.swiglu_limit == 10
init_distributed_environment(world_size=1, rank=0,
                             distributed_init_method='file:///tmp/mtp-nvfp4-component-dist',
                             local_rank=0, backend='gloo')
with set_current_vllm_config(c):
    ensure_model_parallel_initialized(1, 1)

with set_current_vllm_config(c), contextlib.ExitStack() as stack, torch.inference_mode():
    handles = {}

    def get(name):
        path = MODEL / index[name]
        if path not in handles:
            handles[path] = stack.enter_context(safe_open(str(path), framework='pt', device='cpu'))
        value = handles[path].get_tensor(name)
        report['source_tensors'][name] = {'file': index[name], 'shape': list(value.shape),
                                         'dtype': str(value.dtype), 'sha256': digest(value)}
        return value

    policy = MTPExpertOnlyNvfp4Config()
    with torch.device('cuda'):
        model = Glm5NextMoE(cfg, c.parallel_config, policy, prefix='model.layers.45.mlp')
    router = get(PREFIX + 'gate.weight')[IDS]
    bias = get(PREFIX + 'gate.e_score_correction_bias')[IDS]
    model.gate.weight.copy_(router)
    model.gate.e_score_correction_bias.copy_(bias)
    assert torch.equal(model.gate.weight.cpu(), router)
    assert torch.equal(model.gate.e_score_correction_bias.cpu(), bias)
    shared_gate = get(PREFIX + 'shared_experts.gate_proj.weight')
    shared_up = get(PREFIX + 'shared_experts.up_proj.weight')
    shared_down = get(PREFIX + 'shared_experts.down_proj.weight')
    model.shared_experts.gate_up_proj.weight.copy_(torch.cat((shared_gate, shared_up)))
    model.shared_experts.down_proj.weight.copy_(shared_down)
    assert torch.equal(model.shared_experts.gate_up_proj.weight.cpu(), torch.cat((shared_gate, shared_up)))
    assert torch.equal(model.shared_experts.down_proj.weight.cpu(), shared_down)
    native = {proj: torch.stack([get(PREFIX + f'experts.{e}.{proj}.weight') for e in IDS])
              for proj in ('gate_proj', 'up_proj', 'down_proj')}
    assert all(value.dtype == torch.bfloat16 for value in native.values())
    routed = model.experts.routed_experts
    assert routed.w13_weight.device.type == routed.w2_weight.device.type == 'meta'
    replace_parameter(routed, 'w13_weight', torch.cat((native['gate_proj'], native['up_proj']), 1).cuda())
    replace_parameter(routed, 'w2_weight', native['down_proj'].cuda())
    method = routed.quant_method
    method._quantize_weights(routed)
    raw = {name: getattr(routed, name).detach().cpu().clone() for name in
           ('w13_weight', 'w13_weight_scale', 'w13_weight_scale_2',
            'w2_weight', 'w2_weight_scale', 'w2_weight_scale_2')}
    assert raw['w13_weight'].dtype == raw['w2_weight'].dtype == torch.uint8
    assert raw['w13_weight_scale'].dtype == raw['w2_weight_scale'].dtype == torch.float8_e4m3fn
    report['raw_quantized_bytes'] = sum(t.numel() * t.element_size() for t in raw.values())
    report['native_expert_bytes'] = sum(t.numel() * t.element_size() for t in native.values())
    report['packed_tensors'] = {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': digest(v)}
                                for k, v in raw.items()}
    method._setup_kernel(routed)
    routed._already_called_process_weights_after_loading = True
    kernel = method.moe_kernel.fused_experts
    assert isinstance(kernel, B12xExperts)
    report['actual_kernel_class'] = type(kernel).__module__ + '.' + type(kernel).__name__
    report['actual_quant_mode'] = str(kernel._quant_mode)
    assert 'a16' in str(kernel._quant_mode).lower(), report['actual_quant_mode']
    assert policy.selected_prefixes == ['model.layers.45.mlp.experts']
    assert original.n_routed_experts == 288
    report['native_router_shared_exact'] = True
    save()

    def reference(x, ids, weights, use_quantized):
        # Dequantize one expert at a time to bound device scratch memory.
        result = torch.zeros_like(x, dtype=torch.float32)
        clamp_hits = {'gate_above_10': 0, 'up_outside_10': 0}
        for e in range(8):
            if use_quantized:
                gate_up = unpack(raw['w13_weight'][e], raw['w13_weight_scale'][e], raw['w13_weight_scale_2'][e]).cuda()
                down = unpack(raw['w2_weight'][e], raw['w2_weight_scale'][e], raw['w2_weight_scale_2'][e]).cuda()
                gate, up = gate_up.chunk(2, 0)
            else:
                gate = native['gate_proj'][e].float().cuda()
                up = native['up_proj'][e].float().cuda()
                down = native['down_proj'][e].float().cuda()
            g = F.linear(x.float(), gate)
            u = F.linear(x.float(), up)
            clamp_hits['gate_above_10'] += int((g > 10).sum())
            clamp_hits['up_outside_10'] += int((u.abs() > 10).sum())
            hidden = F.silu(g.clamp(max=10)) * u.clamp(min=-10, max=10)
            y = F.linear(hidden, down)
            route_weight = (weights * (ids == e)).sum(-1, keepdim=True)
            result += y * route_weight
            del gate, up, down, g, u, hidden, y
            if use_quantized:
                del gate_up
        return result, clamp_hits

    for tokens, input_scale in ((1, .1), (1, 8.), (2, .1), (4, .1), (8, .1)):
        x = torch.randn(tokens, 4096, device='cuda', dtype=torch.bfloat16) * input_scale
        with set_forward_context(None, c, num_tokens=tokens):
            logits = model.gate(x)[0]
            scores = logits.float().sigmoid()
            route_ids = (scores + model.gate.e_score_correction_bias.float()).topk(8, -1).indices
            weights = scores.gather(1, route_ids)
            weights = weights / weights.sum(-1, keepdim=True) * cfg.routed_scaling_factor
            actual = model(x)
            shared = model.shared_experts(x)
            quant_ref, clamp_hits = reference(x, route_ids, weights, True)
            native_ref, _ = reference(x, route_ids, weights, False)
            quant_ref += shared.float()
            native_ref += shared.float()
            if input_scale == 8.:
                assert min(clamp_hits.values()) > 0, clamp_hits
            assert torch.isfinite(actual).all()
            # Fixed tolerance against independently decoded weights; quantization
            # loss vs native weights is measured separately and never waved through.
            rms = float(quant_ref.square().mean().sqrt())
            torch.testing.assert_close(actual.float(), quant_ref, atol=max(.003, .003*rms), rtol=.03)
            cell = {'tokens': tokens, 'input_scale': input_scale, 'clamp_hits': clamp_hits,
                    'quantized_reference_max_abs': float((actual.float() - quant_ref).abs().max()),
                    'quantized_reference_relative_l2': relative_l2(actual, quant_ref),
                    'native_reference_relative_l2': relative_l2(actual, native_ref)}
            assert cell['quantized_reference_relative_l2'] < .015
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    model(x)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = model(x)
            errors = []
            for scale in (.05, .1, .5):
                x.copy_(torch.randn_like(x) * scale)
                expected = model(x)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.isfinite(output).all()
                torch.testing.assert_close(output, expected, atol=.001, rtol=.01)
                errors.append(float((output - expected).abs().max()))
            cell['changed_input_graph_replay_max_abs'] = errors
            report['cells'].append(cell)
            save()
            print('CELL_PASS', json.dumps(cell), flush=True)
            del graph, output, expected, actual, shared, quant_ref, native_ref
            torch.cuda.empty_cache()
    report.update(state='GPU_COMPONENT_PASS', elapsed_seconds=time.monotonic() - started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  resident_cuda_allocated_bytes=torch.cuda.memory_allocated())
    assert report['peak_cuda_allocated_bytes'] <= LIMIT
    save()
    print('GPU_COMPONENT_PASS', json.dumps(report), flush=True)
