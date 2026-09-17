#!/usr/bin/env python3
"""Supplement sealed GLM quality with full-vocabulary token KL tails."""
import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

FIXTURE = '46255621cd2da8e05546edc3498cfbee518da092af2f44ae243f79da97c0f00b'
TEACHER = 'a1604015f76d6a6b6f032ffb7e3359a770e8dd2564ca89e3b00eb8bcb4d93314'
TOKENS = '5b77e320eaccc959e5f731639aa4d9b908027e7647192305e374c945b44c7d44'
HEAD = '5155051210040e80b6f9a011d7f3e9f68b466fdb1a43af044612b40c34882f5a'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, allow_nan=False, indent=2)
        f.write('\n')


def verify_rows(folder, expected):
    assert sha(folder / 'manifest.json') == expected
    manifest = read(folder / 'manifest.json')
    assert manifest['state'] == 'COMPLETE' and manifest['rows'] == 32
    assert [r['row'] for r in manifest['records']] == list(range(32))
    for row in manifest['records']:
        p = folder / f"row-{row['row']:03d}.safetensors"
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']


def chunk_stats(hidden, head, labels, chunk):
    import torch
    import torch.nn.functional as F
    n = hidden.shape[0]
    best = torch.full((n,), -float('inf'), device=hidden.device)
    arg = torch.zeros(n, dtype=torch.long, device=hidden.device)
    lse = best.clone()
    target = torch.empty_like(lse)
    for start in range(0, head.shape[0], chunk):
        weight = head[start:start + chunk].to(hidden.device)
        logits = F.linear(hidden, weight).float()
        assert torch.isfinite(logits).all()
        values, indices = logits.max(-1)
        take = values > best
        arg[take] = indices[take] + start
        best[take] = values[take]
        lse = torch.logaddexp(lse, torch.logsumexp(logits, -1))
        mask = (labels >= start) & (labels < start + weight.shape[0])
        target[mask] = logits[mask, labels[mask] - start]
    return lse, arg, float((lse - target).sum().item())


def token_kl(p_hidden, q_hidden, head, p_lse, q_lse, chunk):
    import torch
    import torch.nn.functional as F
    total = torch.zeros_like(p_lse, dtype=torch.float64)
    for start in range(0, head.shape[0], chunk):
        weight = head[start:start + chunk].to(p_hidden.device)
        logp = F.linear(p_hidden, weight).float() - p_lse[:, None]
        logq = F.linear(q_hidden, weight).float() - q_lse[:, None]
        total += (logp.exp() * (logp - logq)).sum(-1, dtype=torch.float64)
    assert torch.isfinite(total).all()
    return total


def self_test():
    import torch
    import torch.nn.functional as F
    torch.manual_seed(42)
    p, q, head = torch.randn(9, 5), torch.randn(9, 5), torch.randn(17, 5)
    labels = torch.arange(9)
    pl, pa, pn = chunk_stats(p, head, labels, 4)
    ql, qa, qn = chunk_stats(q, head, labels, 4)
    got = token_kl(p, q, head, pl, ql, 4)
    lp = F.log_softmax(F.linear(p, head), -1)
    lq = F.log_softmax(F.linear(q, head), -1)
    expected = (lp.exp() * (lp - lq)).sum(-1, dtype=torch.float64)
    torch.testing.assert_close(got, expected, atol=3e-6, rtol=3e-6)
    assert pa.tolist() == lp.argmax(-1).tolist()
    assert abs(pn - float(-lp[torch.arange(9), labels].sum())) < 1e-5
    torch.testing.assert_close(token_kl(p, p, head, pl, pl, 4), torch.zeros(9, dtype=torch.float64))
    tied = torch.zeros_like(head)
    assert chunk_stats(p, tied, labels, 4)[1].tolist() == [0] * 9
    assert not torch.cuda.is_initialized()
    print('CPU_FULL_VOCAB_EQUIVALENCE_IDENTITY_TIES_PASS', flush=True)


def run(a):
    root, source, out = a.release, a.result, a.output
    report = read(source / 'quality-report.json')
    assert report['positions'] == report['tokens'] == 65504
    assert report['fixture_manifest_sha256'] == FIXTURE
    assert report['teacher_manifest_sha256'] == TEACHER
    assert not out.exists(), 'refusing to overwrite a supplement'
    verify_rows(root / 'bf16-normalized', TEACHER)
    verify_rows(source / 'variant-normalized', report['variant_normalized_manifest_sha256'])
    assert sha(root / 'quality-eval/manifest.json') == FIXTURE
    token_path = root / 'quality-eval/token_rows.safetensors'
    assert sha(token_path) == TOKENS
    candidate = root / 'k2-massmax-k256'
    head_name = read(candidate / 'model.safetensors.index.json')['weight_map']['lm_head.weight']
    assert not Path(head_name).is_absolute() and '..' not in Path(head_name).parts
    head_path = candidate / head_name
    assert sha(head_path) == HEAD
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    torch.backends.cuda.matmul.allow_tf32 = False
    tokens = load_file(token_path)['input_ids']
    assert tokens.shape == (32, 2048) and tokens.dtype == torch.int64
    with safe_open(str(head_path), framework='pt', device='cpu') as f:
        head = f.get_tensor('lm_head.weight')
    assert head.dtype == torch.bfloat16 and head.shape[1] == 4096
    assert tokens.min() >= 0 and tokens.max() < head.shape[0]
    out.mkdir(parents=True)
    started = time.monotonic()
    rows, values = [], []
    with torch.inference_mode():
        for r in range(32):
            hidden = []
            for folder in (root / 'bf16-normalized', source / 'variant-normalized'):
                h = load_file(folder / f'row-{r:03d}.safetensors')['hidden']
                assert h.dtype == torch.bfloat16 and h.shape == (1, 2048, 4096) and torch.isfinite(h).all()
                hidden.append(h[0, :-1].to('cuda:0'))
            labels = tokens[r, 1:].to('cuda:0')
            pl, pa, pn = chunk_stats(hidden[0], head, labels, 4096)
            ql, qa, qn = chunk_stats(hidden[1], head, labels, 4096)
            kl = token_kl(*hidden, head, pl, ql, 4096).cpu()
            agree = int((pa == qa).sum())
            baseline = report['per_row'][r]
            assert baseline['row'] == r and baseline['tokens'] == 2047
            assert abs(pn - baseline['bf16_nll_sum']) <= 0.001
            assert abs(qn - baseline['variant_nll_sum']) <= 0.001
            assert agree == baseline['top1_agree']
            delta = abs(float(kl.sum()) - baseline['kl_sum'])
            assert delta <= max(0.002, 2e-6 * abs(baseline['kl_sum'])), (r, delta)
            rows.append({'row': r, 'kl_sum': float(kl.sum()), 'baseline_kl_sum_abs_delta': delta, 'bf16_nll_sum': pn, 'variant_nll_sum': qn, 'top1_agree': agree})
            values.extend(kl.tolist())
            print(json.dumps({'row_complete': r, 'seconds': time.monotonic() - started}), flush=True)
    x = torch.tensor(values, dtype=torch.float64)
    assert len(values) == 65504 and torch.isfinite(x).all()
    save(out / 'token-kl.json', values)
    result = {'state': 'KL_TAILS_VERIFIED', 'positions': len(values), 'candidate_manifest_sha256': report['candidate_manifest_sha256'],
              'original_report_sha256': sha(source / 'quality-report.json'), 'variant_normalized_manifest_sha256': report['variant_normalized_manifest_sha256'],
              'teacher_manifest_sha256': TEACHER, 'fixture_manifest_sha256': FIXTURE, 'head_shard_sha256': HEAD,
              'script_sha256': sha(__file__), 'token_kl_sha256': sha(out / 'token-kl.json'), 'mean': float(x.mean()),
              'quantiles': {str(v): float(torch.quantile(x, v, interpolation='linear')) for v in (0.5, 0.95, 0.99, 0.999)},
              'maximum': float(x.max()), 'minimum': float(x.min()), 'negative_count': int((x < 0).sum()), 'nonfinite_count': 0,
              'arithmetic': 'Two-pass BF16 head GEMM2047positions/vocabchunk4096; FP32 logits/lognormalizers/products; FP64 vocabulary summation; linear quantiles; no clamping',
              'baseline_row_tolerance': 'NLL abs<=0.001; KL sum abs<=max(0.002,2e-6*abs(original)); top1 exact',
              'torch': torch.__version__, 'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
              'allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.monotonic() - started, 'per_row': rows}
    save(out / 'report.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_row'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, default=Path('/release'))
    p.add_argument('--result', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        self_test()
    else:
        if a.result is None or a.output is None:
            p.error('--result and --output are required')
        run(a)
