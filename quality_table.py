#!/usr/bin/env python3
"""Render only complete, identity-matched full-trunk quality measurements."""
import argparse
import hashlib
import json
import math
from pathlib import Path

IDENTITY = ('fixture_manifest_sha256', 'teacher_manifest_sha256',
            'baseline_evaluator_sha256', 'head_arithmetic', 'positions', 'trunk_layers')


def validate(report):
    if report.get('state') != 'QUALITY_MEASURED':
        raise ValueError('A projection or partial run cannot enter the quality table')
    if report.get('tokens') != 65504 or report.get('positions') != 65504:
        raise ValueError('Expected all65504 matched next-token positions')
    rows = report.get('per_row', [])
    if [r['row'] for r in rows] != list(range(32)) or any(r['tokens'] != 2047 for r in rows):
        raise ValueError('Expected exactly32 complete, distinct2047-position rows')
    expected = {
        'kl_bf16_to_variant': sum(r['kl_sum'] for r in rows) / 65504,
        'top1_agreement': sum(r['top1_agree'] for r in rows) / 65504,
        'bf16_cross_entropy': sum(r['bf16_nll_sum'] for r in rows) / 65504,
        'variant_cross_entropy': sum(r['variant_nll_sum'] for r in rows) / 65504,
    }
    expected['bf16_perplexity'] = math.exp(expected['bf16_cross_entropy'])
    expected['variant_perplexity'] = math.exp(expected['variant_cross_entropy'])
    expected['perplexity_delta_fraction'] = math.expm1(expected['variant_cross_entropy'] - expected['bf16_cross_entropy'])
    for key, value in expected.items():
        if not math.isfinite(value) or not math.isclose(report[key], value, rel_tol=1e-8, abs_tol=1e-9):
            raise ValueError(f'Aggregate disagrees with actual row evidence: {key}')
    if any(not report.get(key) for key in IDENTITY):
        raise ValueError('Incomplete evaluator identity')
    return report


def render(items):
    first = validate(items[0][1])
    for _, report in items[1:]:
        validate(report)
        if any(report[key] != first[key] for key in IDENTITY):
            raise ValueError('Reports use different fixtures, teachers or arithmetic')
        if not math.isclose(report['bf16_cross_entropy'], first['bf16_cross_entropy'], abs_tol=1e-6):
            raise ValueError('BF16 control changed across reports')
    lines = ['# Matched GLM quality measurements', '',
             'Frozen WikiText2 panel:32×2048 tokens,65,504 next-token positions; full-vocabulary BF16 teacher comparison. This panel does not establish broad task quality or serving acceptance.', '',
             '| Candidate | Experts/layer | KL (lower better) | Top-1 agreement | Perplexity | PPL change vs BF16 |',
             '|---|---:|---:|---:|---:|---:|']
    lines.append(f"| BF16 teacher |288|0|100%|{first['bf16_perplexity']:.5f}|0%|")
    for label, r in items:
        if '|' in label or '\n' in label:
            raise ValueError('Invalid table label')
        lines.append(f"|{label}|{r['experts_per_layer']}|{r['kl_bf16_to_variant']:.6f}|{100*r['top1_agreement']:.3f}%|{r['variant_perplexity']:.5f}|{100*r['perplexity_delta_fraction']:+.3f}%|")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    items, bindings = [], []
    for spec in args.report:
        label, path = spec.split('=', 1)
        raw = Path(path).read_bytes()
        items.append((label, json.loads(raw)))
        bindings.append({'label': label, 'report_sha256': hashlib.sha256(raw).hexdigest()})
    table = render(items)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'TABLE.md').write_text(table)
    (args.output / 'reports.json').write_text(json.dumps({'reports': bindings, 'measurements': dict(items), 'selection_accepted': False}, indent=2) + '\n')
    print(table)


if __name__ == '__main__':
    main()
