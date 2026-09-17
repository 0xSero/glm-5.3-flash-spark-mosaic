#!/usr/bin/env python3
"""Measure prompt perplexity on the real server using a sealed private reference."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from run import request_json, metrics, validate_runtime, write
from runtime_parity import token_id


def score(ids, teacher_top1, response):
    usage = response['usage']
    if usage['prompt_tokens'] != 2048 or usage['completion_tokens'] != 1:
        raise ValueError('Prompt/output accounting changed')
    logs = response['choices'][0]['logprobs']
    if any(len(logs[k]) != 2049 for k in ('tokens', 'token_logprobs', 'top_logprobs')):
        raise ValueError('Incomplete echoed prompt logprobs')
    if [token_id(t) for t in logs['tokens'][:2048]] != ids:
        raise ValueError('Echoed prompt IDs differ from the frozen fixture')
    if logs['token_logprobs'][0] is not None:
        raise ValueError('Unexpected likelihood for first prompt token')
    if len(teacher_top1) != 2047:
        raise ValueError('Wrong teacher prediction geometry')
    nll, agree, covered, tied, gt_correct = 0.0, 0, 0, 0, 0
    for pos in range(1, 2048):
        lp = logs['token_logprobs'][pos]
        if not isinstance(lp, (int, float)) or not math.isfinite(lp) or lp > 1e-5:
            raise ValueError('Invalid ground-truth logprob')
        alternatives = {token_id(k): v for k, v in logs['top_logprobs'][pos].items()}
        if len(alternatives) < 20 or any(not math.isfinite(v) for v in alternatives.values()):
            raise ValueError('Expected at least20 finite exact-token alternatives')
        if ids[pos] not in alternatives or not math.isclose(alternatives[ids[pos]], lp, abs_tol=1e-6):
            raise ValueError('Ground-truth logprob disagrees with alternatives')
        highest = max(alternatives.values())
        winners = [k for k, value in alternatives.items() if value == highest]
        # If every returned score ties, unseen ties may exist. Do not claim
        # complete argmax coverage in that case. Otherwise all maxima fit top20.
        if min(alternatives.values()) < highest:
            predicted = min(winners)  # Same stable first-ID convention as teacher argmax.
            covered += 1
            agree += predicted == teacher_top1[pos-1]
            gt_correct += predicted == ids[pos]
        tied += len(winners) > 1
        nll -= lp
    return {'positions': 2047, 'runtime_nll_sum': nll,
            'teacher_top1_agree': agree, 'argmax_covered_positions': covered,
            'runtime_ground_truth_top1_correct': gt_correct, 'tied_max_positions': tied}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--runtime-receipt', type=Path, required=True)
    p.add_argument('--base-url', required=True)
    p.add_argument('--metrics-url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reference = json.loads(a.reference.read_text())
    for key, expected in {
        'fixture_manifest_sha256': '46255621cd2da8e05546edc3498cfbee518da092af2f44ae243f79da97c0f00b',
        'teacher_manifest_sha256': 'a1604015f76d6a6b6f032ffb7e3359a770e8dd2564ca89e3b00eb8bcb4d93314',
        'head_shard_sha256': '5155051210040e80b6f9a011d7f3e9f68b466fdb1a43af044612b40c34882f5a',
    }.items():
        if reference.get(key) != expected:
            raise ValueError('Reference identity mismatch: '+key)
    runtime = json.loads(a.runtime_receipt.read_text())
    validate_runtime(runtime, a.model, 262144)
    if runtime.get('logprobs_mode') != 'raw_logprobs':
        raise ValueError('Raw logprob runtime identity required')
    inputs, teacher = reference['input_ids'], reference['teacher_top1']
    if len(inputs) != 32 or len(teacher) != 32:
        raise ValueError('Expected all32 frozen rows')
    if any(len(row) != 2048 for row in inputs) or any(len(row) != 2047 for row in teacher):
        raise ValueError('Reference geometry mismatch')
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    binding = {'reference_sha256': hashlib.sha256(a.reference.read_bytes()).hexdigest(),
               'runtime_receipt_sha256': hashlib.sha256(a.runtime_receipt.read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(a.output/'identity.json', binding)
    for index, (ids, expected) in enumerate(zip(inputs, teacher)):
        if not metrics(a.metrics_url, a.model)['idle']:
            raise ValueError('Runtime not exclusively idle before quality row')
        started = time.monotonic()
        response = request_json(a.base_url.rstrip('/')+'/completions', {
            'model': a.model, 'prompt': ids, 'max_tokens': 1, 'temperature': 0,
            'echo': True, 'logprobs': 20, 'return_tokens_as_token_ids': True,
            'return_token_ids': True, 'add_special_tokens': False}, timeout=300)
        write(a.output/f'private-response-{index:02d}.json', response)
        result = score(ids, expected, response)
        result.update(row=index, seconds=time.monotonic()-started,
                      raw_response_sha256=hashlib.sha256((a.output/f'private-response-{index:02d}.json').read_bytes()).hexdigest())
        rows.append(result)
        write(a.output/'progress.json', {'completed_rows':len(rows), 'rows':rows})
        print(json.dumps(result), flush=True)
    positions = sum(r['positions'] for r in rows)
    nll = sum(r['runtime_nll_sum'] for r in rows)
    covered = sum(r['argmax_covered_positions'] for r in rows)
    if positions != 65504 or not covered:
        raise ValueError('Incomplete quality measurement')
    report = dict(binding, state='ACTUAL_SERVING_QUALITY_MEASURED', positions=positions,
                  runtime_cross_entropy=nll/positions, runtime_perplexity=math.exp(nll/positions),
                  teacher_top1_agreement=sum(r['teacher_top1_agree'] for r in rows)/covered,
                  argmax_covered_positions=covered,
                  tied_max_positions=sum(r['tied_max_positions'] for r in rows), per_row=rows,
                  scope='Actual raw API prompt likelihood and BF16 teacher argmax on the frozen panel. No full-vocabulary KL or broad task-quality claim.')
    write(a.output/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='per_row'}))


if __name__ == '__main__':
    main()
