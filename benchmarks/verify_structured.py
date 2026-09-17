"""Recompute benchmark timing, semantic acceptance and native accounting from raw cells."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--last-integer', type=int, required=True)
a = p.parse_args()
root = a.directory.resolve()
sys.path.insert(0, str(root))
from timing import summarize
from native_metrics import compare

manifest = json.loads((root / 'manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
records = []
for path in sorted(root.glob('[0-9]*-*.json'),
                   key=lambda p: (int(p.name.split('-')[0]), p.name)):
    d = json.loads(path.read_text())
    row = d['row']
    actual = json.loads(row['content'])
    assert actual == {'numbers': list(range(1, a.last_integer + 1))}, path
    assert all(type(n) is int for n in actual['numbers']), path
    assert row['success'] and row['finish_reason'] == 'stop', path
    assert d['before']['idle'] and d['after']['idle'], path
    assert row['usage']['prompt_tokens'] == d['actual_prompt_tokens'], path
    assert d['target'] - 4 <= d['actual_prompt_tokens'] <= d['target'], path
    computed = summarize([row], row['ended_monotonic'] - row['started_monotonic'])
    assert computed == d['summary'], path
    native = compare(d['before']['request_metrics'], d['after']['request_metrics'], [row])
    mtp = {k: v - d['before']['speculative_counters'][k] for k, v in d['after']['speculative_counters'].items()}
    drafted = sum(v for k, v in mtp.items() if k.endswith('num_draft_tokens_total'))
    accepted = sum(v for k, v in mtp.items() if k.endswith('num_accepted_tokens_total'))
    assert all(v >= 0 for v in mtp.values()) and 0 < accepted <= drafted
    window = computed['matched_decode']
    assert window['status'] in ('MATCHED_SCREEN', 'MATCHED_SUSTAINED'), path
    records.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'prompt_tokens': d['actual_prompt_tokens'], 'output_tokens': row['usage']['completion_tokens'],
        'repeat': d['repeat'], 'status': window['status'], 'window_seconds': window['window_seconds'],
        'total_decode_tok_s': window['total_aggregate_decode_tok_s'],
        'per_request_decode_tok_s': window['mean_per_request_decode_tok_s'],
        'native_prefill_tok_s': native['prompt_tokens_per_sum_request_prefill_second'],
        'ttft_seconds': computed['ttft_p50_seconds'], 'drafted': drafted, 'accepted': accepted})
lines = ['# Verified structured generation', '',
    'C1; temperature0; low reasoning; cold prefix cache; native MTP enabled. Correct normal-stop JSON answers required. '
    'Every timing value was recomputed from exact emitted token IDs. Warmups and sub30-second screens remain labeled. '
    'This counting task does not establish broad model quality or MTP speedup.', '',
    '| Input tokens | Output tokens | Repeat | Class | Window s | TOTAL decode tok/s | Per-request decode tok/s | Native prefill tok/s | TTFT s |',
    '|---:|---:|---|---|---:|---:|---:|---:|---:|']
for r in records:
    lines.append(f"| {r['prompt_tokens']} | {r['output_tokens']} | {r['repeat']} | {r['status']} | {r['window_seconds']:.2f} | {r['total_decode_tok_s']:.2f} | {r['per_request_decode_tok_s']:.2f} | {r['native_prefill_tok_s']:.2f} | {r['ttft_seconds']:.2f} |")
(root / 'TABLE.md').write_text('\n'.join(lines) + '\n')
(root / 'verified.json').write_text(json.dumps({'records': records, 'last_integer': a.last_integer,
    'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
print('\n'.join(lines))
