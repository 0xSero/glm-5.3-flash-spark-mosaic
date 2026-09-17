#!/usr/bin/env python3
"""Small, position-matched offline/serving logprob probe. No broad quality claim."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import urllib.request

FIXTURE_SHA = '46255621cd2da8e05546edc3498cfbee518da092af2f44ae243f79da97c0f00b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def position_contract(ids, hidden_position=2046):
    if len(ids) != 2048 or hidden_position != 2046:
        raise ValueError('expected last scored position of the frozen 2048-token fixture')
    if any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('invalid token IDs')
    return ids[:hidden_position + 1], ids[hidden_position + 1]


def terminal_contract(inspect):
    if isinstance(inspect, list):
        if len(inspect) != 1:
            raise ValueError('expected one capture container')
        inspect = inspect[0]
    state = inspect.get('State', {})
    if (not inspect.get('Id') or state.get('Running') is not False or
            state.get('ExitCode') != 0 or not state.get('FinishedAt') or
            state['FinishedAt'].startswith('0001-')):
        raise ValueError('capture must have actually terminated successfully before GPU preparation')
    return inspect['Id']


def prepare(args):
    # Validate completed quality and identities before any CUDA import/init.
    quality, root, fixture = args.quality, args.artifact, args.fixture
    identity = read(quality / 'evaluation-identity.json')
    report = read(quality / 'quality-report.json')
    norm = quality / 'variant-normalized'
    manifest = read(norm / 'manifest.json')
    terminal_contract(read(args.capture_inspect))
    if (manifest.get('state') != 'COMPLETE' or manifest.get('rows') != 32 or
            [r['row'] for r in manifest.get('records', [])] != list(range(32)) or
            report.get('variant_normalized_manifest_sha256') != sha(norm / 'manifest.json')):
        raise ValueError('quality comparison and normalized capture are not sealed together')
    for key, name in [('candidate_manifest_sha256', 'EXL3_MANIFEST.json'),
                      ('candidate_index_sha256', 'model.safetensors.index.json'),
                      ('candidate_config_sha256', 'config.json')]:
        if identity[key] != sha(root / name) or report.get(key) != identity[key]:
            raise ValueError('artifact/quality identity mismatch: ' + key)
    if sha(fixture / 'manifest.json') != FIXTURE_SHA or identity['fixture_manifest_sha256'] != FIXTURE_SHA:
        raise ValueError('fixture identity mismatch')
    token_path = fixture / 'token_rows.safetensors'
    if sha(token_path) != read(fixture / 'manifest.json')['token_rows_sha256']:
        raise ValueError('fixture tokens changed')
    index = read(root / 'model.safetensors.index.json')['weight_map']
    head_file = index['lm_head.weight']
    if Path(head_file).is_absolute() or '..' in Path(head_file).parts:
        raise ValueError('unsafe head path')
    head_path = root / head_file
    declared = {x['path']: x for x in read(root / 'EXL3_MANIFEST.json')['files']}
    if sha(head_path) != declared[head_file]['sha256']:
        raise ValueError('native head shard integrity failure')
    # Parent must independently reserve this GPU. Fail rather than share active capture.
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise ValueError('GPU preparation requires an idle quality node: compute PIDs exist')
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from safetensors import safe_open
    torch.cuda.set_device(0)
    with safe_open(str(head_path), framework='pt', device='cpu') as f:
        head = f.get_tensor('lm_head.weight')
    if head.dtype != torch.bfloat16 or head.ndim != 2 or head.shape[1] != 4096:
        raise ValueError('head must retain native BF16 geometry')
    ids = load_file(token_path)['input_ids']
    if tuple(ids.shape) != (32, 2048):
        raise ValueError('fixture geometry mismatch')
    probes = []
    rows = list(range(32)) if args.rows == 32 else list(range(0, 32, 4))
    for row in rows:
        record = manifest['records'][row]
        path = norm / f'row-{row:03d}.safetensors'
        if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('normalized row integrity failure')
        hidden = load_file(path)['hidden']
        if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != (1, 2048, 4096) or not torch.isfinite(hidden).all():
            raise ValueError('normalized hidden dtype/shape/finiteness mismatch')
        hidden = hidden[0, :-1].to('cuda:0')
        lse = torch.full((2047,), -float('inf'), dtype=torch.float32, device='cuda:0')
        last_logits = []
        with torch.inference_mode():
            for start in range(0, head.shape[0], 4096):
                # Same 2047-position GEMM geometry and chunk reduction as quality comparator.
                logits = F.linear(hidden, head[start:start + 4096].to('cuda:0')).float()
                lse = torch.logaddexp(lse, torch.logsumexp(logits, dim=-1))
                last_logits.append(logits[-1].cpu())
            logp = torch.cat(last_logits) - lse[-1].cpu()
        if not torch.isfinite(logp).all():
            raise ValueError('nonfinite expected logprobs')
        order = torch.argsort(logp, descending=True, stable=True)[:20].tolist()
        prefix, label = position_contract(ids[row].tolist())
        probes.append({'row': row, 'hidden_position': 2046, 'prefix_ids': prefix,
                       'label_token_id': label, 'expected_top1': order[0],
                       'top1_gap': float(logp[order[0]] - logp[order[1]]),
                       'expected_top_logprobs': {str(i): float(logp[i]) for i in order},
                       'normalized_row_sha256': record['sha256']})
        del hidden, logits, lse
    write_new(args.output, {'schema': 'glm53-runtime-parity-probes-v1', 'state': 'PREPARED_NOT_RUN',
        'candidate_manifest_sha256': identity['candidate_manifest_sha256'],
        'evaluation_identity_sha256': sha(quality / 'evaluation-identity.json'),
        'quality_report_sha256': sha(quality / 'quality-report.json'),
        'capture_inspect_sha256': sha(args.capture_inspect),
        'normalized_manifest_sha256': sha(norm / 'manifest.json'),
        'fixture_manifest_sha256': FIXTURE_SHA, 'token_rows_sha256': sha(token_path),
        'head_shard_sha256': sha(head_path), 'script_sha256': sha(__file__),
        'arithmetic': 'BF16 F.linear on all2047positions, FP32logits, vocabchunk4096 sequential logaddexp/logsumexp',
        'torch': torch.__version__, 'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'probes': probes})


def token_id(value):
    match = re.fullmatch(r'token_id:(\d+)', value)
    if not match:
        raise ValueError('UNSUPPORTED: server did not expose exact token-ID logprob keys')
    return int(match[1])


def compare_response(probe, response, tolerance):
    usage = response.get('usage', {})
    if usage.get('prompt_tokens') != len(probe['prefix_ids']) or usage.get('completion_tokens') != 1:
        raise ValueError('UNSUPPORTED: exact prompt/output token accounting unavailable or changed')
    choices = response.get('choices', [])
    if len(choices) != 1:
        raise ValueError('UNSUPPORTED: expected one completion')
    logs = choices[0].get('logprobs') or {}
    if len(logs.get('tokens', [])) != 1 or len(logs.get('top_logprobs', [])) != 1:
        raise ValueError('UNSUPPORTED: single-token raw logprobs unavailable')
    actual = token_id(logs['tokens'][0])
    observed = {token_id(k): float(v) for k, v in logs['top_logprobs'][0].items()}
    expected = {int(k): v for k, v in probe['expected_top_logprobs'].items()}
    if len(observed) < 2 or any(not math.isfinite(x) for x in observed.values()):
        raise ValueError('UNSUPPORTED: insufficient or nonfinite top logprobs')
    shared = set(expected) & set(observed)
    drift = max((abs(expected[i] - observed[i]) for i in shared), default=None)
    covered = probe['expected_top1'] in observed and actual in expected
    top1 = actual == probe['expected_top1']
    return {'row': probe['row'], 'state': 'MEASURED', 'expected_top1': probe['expected_top1'],
            'actual_top1': actual, 'top1_match': top1, 'top1_gap': probe['top1_gap'],
            'near_tie_002': probe['top1_gap'] <= .02, 'topk_intersection': len(shared),
            'expected_topk_count': len(expected), 'returned_topk_count': len(observed),
            'max_shared_logprob_drift': drift, 'both_top1_logprobs_covered': covered,
            'strict_probe_pass': top1 and covered and drift is not None and drift <= tolerance,
            'observed_top_logprobs': observed}


def api(args):
    probes, receipt = read(args.probes), read(args.runtime_receipt)
    if probes.get('schema') != 'glm53-runtime-parity-probes-v1':
        raise ValueError('wrong probes schema')
    if (receipt.get('candidate_manifest_sha256') != probes['candidate_manifest_sha256'] or
            receipt.get('model') != args.model or receipt.get('logprobs_mode') != 'raw_logprobs' or
            not receipt.get('evidence_sha256')):
        raise ValueError('UNSUPPORTED: runtime model digest/raw_logprobs/evidence identity not proven')
    if args.output.exists():
        raise ValueError('output exists')
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('OPENAI_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['OPENAI_API_KEY']
    results = []
    for probe in probes['probes']:
        payload = {'model': args.model, 'prompt': probe['prefix_ids'], 'max_tokens': 1,
                   'temperature': 0, 'logprobs': 20, 'return_tokens_as_token_ids': True,
                   'return_token_ids': True, 'add_special_tokens': False}
        request = urllib.request.Request(args.base_url.rstrip('/') + '/completions',
                                         json.dumps(payload).encode(), headers)
        response = None
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as r:
                response = json.load(r)
            result = compare_response(probe, response, args.logprob_tolerance)
        except Exception as exc:
            result = {'row': probe['row'], 'state': 'UNSUPPORTED_OR_FAILED', 'error': str(exc)}
        result['raw_response'] = response
        results.append(result)
        if result['state'] != 'MEASURED':
            break
    write_new(args.output, {'schema': 'glm53-runtime-parity-result-v1',
        'state': 'MEASURED' if len(results) == len(probes['probes']) and all(r['state'] == 'MEASURED' for r in results) else 'INCOMPLETE',
        'probes_sha256': sha(args.probes), 'runtime_receipt_sha256': sha(args.runtime_receipt),
        'logprob_tolerance': args.logprob_tolerance, 'runtime': receipt, 'results': results,
        'claim': 'Small final-position raw-logprob probe only; FP8 KV, kernel arithmetic and near ties may produce drift. No broad quality acceptance.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    p = sub.add_parser('prepare')
    for name in ('quality', 'artifact', 'fixture', 'capture-inspect', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--rows', type=int, choices=(8, 32), default=8)
    p.set_defaults(func=prepare)
    p = sub.add_parser('api')
    for name in ('probes', 'runtime-receipt', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--base-url', required=True, help='including /v1')
    p.add_argument('--model', required=True)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--logprob-tolerance', type=float, default=.05)
    p.set_defaults(func=api)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    if getattr(args, 'logprob_tolerance', 0) < 0:
        parser.error('tolerance must be nonnegative')
    args.func(args)


if __name__ == '__main__':
    main()
