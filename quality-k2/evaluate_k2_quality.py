#!/usr/bin/env python3
"""Full-trunk uniform K2 quality using the unchanged low-bpw BF16 head comparison.

Launch with one GPU per Spark. This is offline quality, not serving throughput.
The original evaluator and its two imports must be on PYTHONPATH.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re

FIXTURE_SHA = '46255621cd2da8e05546edc3498cfbee518da092af2f44ae243f79da97c0f00b'
TEACHER_SHA = 'a1604015f76d6a6b6f032ffb7e3359a770e8dd2564ca89e3b00eb8bcb4d93314'
BASE_SHA = '0c02949a77d28fe7c45786e964a1241cbba8d33b076335ca6a08a4fcac28729a'
from validate_source import validate as validate_k2_source, REPO as SOURCE_REPO, REVISION as SOURCE_REVISION
_base = None
_worker = None


def prepare_native_imports():
    """Reuse the observer's real quantization-only import path, not attention stubs."""
    native = importlib.import_module('reap.glm53_exl3_native')
    wrapper, extension = native._native_modules()
    linear = wrapper.load_linear_exl3_cls()
    if linear.__module__ != 'exllamav3.modules.quant.exl3' or not callable(getattr(linear, 'get_weight_tensor', None)):
        raise RuntimeError('observer did not provide the real EXL3 reconstruction class')
    return wrapper, extension, linear


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_path(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('unsafe artifact index path')
    path = root / relative
    if not path.is_file():
        raise ValueError(f'missing indexed artifact file: {relative}')
    return path


def index_contract(root):
    """Index-only geometry validation; full artifact bytes need structural audit."""
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    text = config.get('text_config', config)
    experts = text['n_routed_experts']
    if type(experts) is not int or not 8 <= experts <= 288:
        raise ValueError('invalid candidate expert count')
    if text.get('num_hidden_layers') != 45 or text.get('hidden_size') != 4096:
        raise ValueError('not expected GLM-5.3-Flash trunk geometry')
    weights = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    packed = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.rank([0-3])\.(suh|svh|trellis|mcg)$')
    actual = set()
    for key, file in weights.items():
        match = packed.fullmatch(key)
        if match and int(match[1]) < 45:
            actual.add(key)
    expected = {f'model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.rank{rank}.{suffix}'
                for layer in range(3, 45) for expert in range(experts)
                for proj in ('gate_proj', 'up_proj', 'down_proj') for rank in range(4)
                for suffix in ('suh', 'svh', 'trellis', 'mcg')}
    if actual != expected:
        raise ValueError(f'packed expert index coverage mismatch: missing={len(expected-actual)} extra={len(actual-expected)}')
    for file in set(weights.values()):
        checked_path(root, file)
    return experts, weights


def configured():
    global _base, _worker
    if _base is not None:
        return _base
    prepare_native_imports()
    base = importlib.import_module('evaluate_low_bpw_quality')
    if sha(base.__file__) != BASE_SHA:
        raise ValueError('baseline evaluator differs from pinned comparison implementation')
    root = Path(os.environ['GLM53_PRUNED_ARTIFACT'])
    experts, weights = index_contract(root)
    base.ARTIFACT = root
    base.CONFIG_ROOT = Path(os.environ['GLM53_K2_INVENTORY']).parent
    base.EVAL = Path(os.environ['GLM53_PRUNED_OUTPUT'])
    base.FIXTURE_EVAL = Path(os.environ['GLM53_PRUNED_FIXTURE'])
    base.TOKENS = base.FIXTURE_EVAL / 'token_rows.safetensors'
    base.BF16_NORMALIZED = Path(os.environ['GLM53_PRUNED_TEACHER'])
    base.EXPERTS = experts
    base.GPU_IDS = (0,)
    original_source = base.EvaluationSourceTensors

    class CandidateTensors(original_source):
        def get(self, key):
            if base.ROUTED_SOURCE.fullmatch(key):
                return super().get(key)
            path = checked_path(root, self.weight_map[key])
            with base.safe_open(path, framework='pt', device='cpu') as handle:
                return handle.get_tensor(key)

    base.EvaluationSourceTensors = CandidateTensors

    def overwrite(layer, layer_idx, device):
        # Preserve original reconstruction arithmetic, resolving packed tensors
        # by actual candidate index instead of depending on legacy part sidecars.
        bank = layer.mlp.experts
        intermediate = bank.intermediate_dim
        if bank.num_experts != experts or intermediate != 2048:
            raise ValueError('candidate model and packed geometry disagree')
        with base.torch.inference_mode():
            for expert in range(experts):
                for rank in range(4):
                    start, end = rank * 512, (rank + 1) * 512
                    for projection in ('gate_proj', 'up_proj', 'down_proj'):
                        prefix = base._packed_prefix(layer_idx, expert, projection, rank)
                        tensors = {}
                        groups = {}
                        for suffix in ('suh', 'svh', 'trellis', 'mcg'):
                            key = prefix + '.' + suffix
                            groups.setdefault(weights[key], []).append(key)
                        for relative, keys in groups.items():
                            with base.safe_open(checked_path(root, relative), framework='pt', device=str(device)) as handle:
                                for key in keys:
                                    tensors[key] = handle.get_tensor(key)
                        class Reader:
                            def get_tensor(self, key):
                                return tensors[key]
                        is_down = projection == 'down_proj'
                        module = base._load_linear(Reader(), prefix, 512 if is_down else 4096,
                                                   4096 if is_down else 512, base.torch.bfloat16)
                        weight = module.get_weight_tensor().T
                        if is_down:
                            target = bank.down_proj[expert, :, start:end]
                        else:
                            offset = 0 if projection == 'gate_proj' else intermediate
                            target = bank.gate_up_proj[expert, offset + start:offset + end]
                        target.copy_(weight.to(dtype=target.dtype))
                        del module, weight, tensors
                if not base.torch.isfinite(bank.gate_up_proj[expert]).all().item() or not base.torch.isfinite(bank.down_proj[expert]).all().item():
                    raise ValueError('nonfinite reconstructed expert')
    base.overwrite_variant_experts = overwrite
    _worker = base.layer_worker
    base.layer_worker = layer_worker
    _base = base
    return base


def layer_worker(*args):
    configured()
    return _worker(*args)


def validate_normalized(base, directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows = manifest.get('records', [])
    if manifest.get('state') != 'COMPLETE' or manifest.get('rows') != 32 or [r['row'] for r in rows] != list(range(32)):
        raise ValueError('normalized reference does not cover exact32 rows')
    for entry in rows:
        path = directory / f"row-{entry['row']:03d}.safetensors"
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('normalized tensor integrity failed')
        with base.safe_open(path, framework='pt', device='cpu') as handle:
            hidden = handle.get_tensor('hidden')
            if tuple(hidden.shape) != (1, 2048, 4096) or hidden.dtype != base.torch.bfloat16 or not base.torch.isfinite(hidden).all():
                raise ValueError('normalized tensor geometry/dtype/finiteness failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'capture', 'compare', 'all'), default='all')
    parser.add_argument('--source-inventory', type=Path, required=True,
                        help='Pinned K2 full133-file HF inventory plus canonical metadata bundle.')
    for name in ('artifact', 'output', 'fixture', 'teacher'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ('artifact', 'output', 'fixture', 'teacher'):
        os.environ['GLM53_PRUNED_' + name.upper()] = str(getattr(args, name).resolve())
    os.environ['GLM53_K2_INVENTORY'] = str(args.source_inventory.resolve())
    source_proof = validate_k2_source(args.artifact,args.source_inventory,full_hash=True)
    base = configured()
    if sha(base.FIXTURE_EVAL / 'manifest.json') != FIXTURE_SHA or sha(base.BF16_NORMALIZED / 'manifest.json') != TEACHER_SHA:
        raise ValueError('fixture or teacher identity mismatch')
    fixture = json.loads((base.FIXTURE_EVAL / 'manifest.json').read_text())
    if sha(base.TOKENS) != fixture['token_rows_sha256']:
        raise ValueError('fixture token digest mismatch')
    ids = base.load_file(base.TOKENS)['input_ids']
    if tuple(ids.shape) != (32, 2048):
        raise ValueError('wrong evaluation token geometry')
    validate_normalized(base, base.BF16_NORMALIZED)
    candidate_sha = source_proof['manifest_sha256']
    artifact_mode = 'original_uniform_k2_unpruned_control'
    dependencies = ('release_route_capture_v2', 'glm53_exl3_tp4')
    identity = {'schema': 'glm53-k2-full-trunk-quality-identity-v1',
                'artifact_mode': artifact_mode,
                'original_source_inventory_sha256': sha(args.source_inventory) if args.source_inventory else None,
                'original_source_repo': SOURCE_REPO if args.source_inventory else None,
                'original_source_revision': SOURCE_REVISION if args.source_inventory else None,
                'candidate_index_sha256': sha(args.artifact / 'model.safetensors.index.json'),
                'candidate_config_sha256': sha(args.artifact / 'config.json'),
                'candidate_manifest_sha256': candidate_sha,
                'source_validation': source_proof,
                'canonical_capture_config_sha256': sha(base.CONFIG_ROOT / 'config.json'),
                'fixture_manifest_sha256': FIXTURE_SHA, 'teacher_manifest_sha256': TEACHER_SHA,
                'baseline_evaluator_sha256': BASE_SHA, 'adapter_sha256': sha(__file__),
                'quantization_import_policy': 'Existing observer vLLM quantization-only namespace loader; real LinearEXL3 and compiled extension, no attention substitute.',
                'native_import_sha256': {name: sha(importlib.import_module(name).__file__) for name in
                    ('reap.glm53_exl3_native', 'vllm.model_executor.layers.quantization.exl3',
                     'exllamav3.modules.quant.exl3', 'exllamav3_ext')},
                'dependency_sha256': {name: sha(importlib.import_module(name).__file__) for name in dependencies},
                'experts_per_layer': base.EXPERTS, 'trunk_layers': 45, 'positions': 65504,
                'head_arithmetic': 'BF16 torch.nn.functional.linear, then FP32 logits; vocabulary chunks4096',
                'mtp_policy': 'Preserved artifact MTP excluded from target-trunk quality, matching teacher baseline.',
                'capture_policy': 'All45 decoder layers, propagated hidden streams and previous DSA attention token indices, final native RMSNorm.'}
    base.EVAL.mkdir(parents=True, exist_ok=True)
    binding = base.EVAL / 'evaluation-identity.json'
    if binding.exists():
        if json.loads(binding.read_text()) != identity:
            raise ValueError('refusing to resume outputs from a different candidate/evaluator')
    else:
        if any(base.EVAL.iterdir()):
            raise ValueError('unbound existing output directory')
        base.atomic_json(binding, identity)
    if args.phase == 'preflight':
        print(json.dumps({'state': 'PREFLIGHT_PASS', **identity}))
        return
    variant = base.EVAL / 'variant-normalized'
    if args.phase in ('capture', 'all'):
        variant = base.seal_normalized('variant', base.run_layers('variant', base.EVAL), base.EVAL)
    if args.phase in ('compare', 'all'):
        validate_normalized(base, variant)
        metrics = base.compare(base.BF16_NORMALIZED, variant)
        if metrics['tokens'] != 65504:
            raise ValueError('comparison did not cover exact65504 positions')
        report = {'schema': 'glm53-k2-full-trunk-quality-v1', 'state': 'QUALITY_MEASURED',
                  'quality_accepted': False, 'runtime_accepted': False, **identity,
                  'variant_normalized_manifest_sha256': sha(variant / 'manifest.json'), **metrics}
        report['schema'] = 'glm53-k2-full-trunk-quality-v1'
        base.atomic_json(base.EVAL / 'quality-report.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'per_row'}, indent=2))


if __name__ == '__main__':
    main()
