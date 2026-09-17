"""Summarize sampled telemetry over hash-bound accepted measurement intervals."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
root = a.directory
samples = [json.loads(line) for line in (root / 'telemetry.jsonl').read_text().splitlines()]
assert all(b['monotonic'] > a['monotonic'] for a, b in zip(samples, samples[1:]))
verified = json.loads((root / 'verified.json').read_text())
rows = []
for record in verified['records']:
    if not record['repeat'].startswith('measured'):
        continue
    path = root / record['file']
    assert sha(path) == record['sha256']
    row = json.loads(path.read_text())['row']
    start, end = row['started_monotonic'], row['ended_monotonic']
    inside = [s for s in samples if start <= s['monotonic'] <= end]
    before = [s for s in samples if s['monotonic'] < start]
    after = [s for s in samples if s['monotonic'] > end]
    brackets = before[-1:] + inside + after[:1]
    gaps = [b['monotonic'] - a['monotonic'] for a, b in zip(brackets, brackets[1:])]
    covered = bool(before and after and inside and gaps and max(gaps) <= 6)
    good = [s for s in inside if s['exit_code'] == 0]
    def values(index):
        return [float(s['gpu_csv'].split(',')[index]) for s in good if s['gpu_csv'].split(',')[index].strip() not in ('[N/A]', 'N/A')]
    temp, power, clocks = values(0), values(1), values(2)
    rows.append({'file': path.name, 'sha256': record['sha256'], 'full_interval_sampled': covered,
        'samples': len(inside), 'failed_gpu_samples': len(inside)-len(good),
        'max_bracket_gap_seconds': max(gaps) if gaps else None,
        'peak_temp_C': max(temp) if temp else None, 'peak_power_W': max(power) if power else None,
        'minimum_sm_clock_MHz': min(clocks) if clocks else None,
        'minimum_mem_available_kB': min(int(s['mem_available'].split()[0]) for s in inside) if inside else None})
result = {'scope': 'Two-second samples, not continuous maxima; no CPU temperature or power-setting changes. Full sampling requires enclosing samples and no gap over six seconds.',
    'telemetry_sha256': sha(root / 'telemetry.jsonl'), 'verified_sha256': sha(root / 'verified.json'),
    'summarizer_sha256': sha(Path(__file__)), 'rows': rows}
(root / 'telemetry-summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(rows, indent=2))
