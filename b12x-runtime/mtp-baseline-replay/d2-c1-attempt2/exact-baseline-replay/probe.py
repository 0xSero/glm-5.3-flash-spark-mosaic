"""Isolated C1 structured decode with exact tokens and independently scored output."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import run as baseline
import native_metrics
from timing import summarize

ROOT = Path(__file__).parent
BASE = 'http://127.0.0.1:18080'
MODEL = 'glm-5.3-flash'
TEMPLATE = {'enable_thinking': True, 'reasoning_effort': 'low'}
TASK = '\nReturn one JSON object with key numbers containing every integer from 1 through 300, in ascending order. Finish immediately after 300.\n'
FILLER = 'The archive contains ordinary warehouse inventory records.\n'


def save(name, value):
    baseline.write(ROOT / name, value)


def count(prompt):
    return baseline.request_json(BASE + '/tokenize', {'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'add_generation_prompt': True, 'chat_template_kwargs': TEMPLATE})['count']


def prepare(target):
    low, high = 0, target
    while low < high:
        mid = (low + high + 1) // 2
        if count(FILLER * mid + TASK) <= target:
            low = mid
        else:
            high = mid - 1
    base = FILLER * low
    low, high = 0, target - count(base + TASK) + 8
    while low < high:
        mid = (low + high + 1) // 2
        if count(base + ' x' * mid + TASK) <= target:
            low = mid
        else:
            high = mid - 1
    prompt = base + ' x' * low + TASK
    actual = count(prompt)
    assert target - 4 <= actual <= target, 'Prompt outside documented four-token tolerance'
    return prompt, actual


def cell(target, repeat):
    prompt, actual = prepare(target)
    before = baseline.metrics(BASE + '/metrics', MODEL)
    assert before['idle'], 'Other requests present'
    payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0, 'max_tokens': 2048, 'stream': True,
        'stream_options': {'include_usage': True}, 'return_token_ids': True,
        'chat_template_kwargs': TEMPLATE, 'response_format': {'type': 'json_object'}}
    started = time.monotonic()
    row = {'success': False, 'started_monotonic': started, 'events': []}
    chunks = []
    done = False
    with urllib.request.urlopen(urllib.request.Request(BASE + '/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}), timeout=1800) as response:
        for raw in response:
            now = time.monotonic()
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            chunk = json.loads(data)
            chunks.append(chunk)
            assert not chunk.get('error'), chunk
            if chunk.get('usage'):
                row['usage'] = chunk['usage']
            for choice in chunk.get('choices', []):
                delta = choice.get('delta') or {}
                text = delta.get('content') or ''
                reasoning = delta.get('reasoning') or delta.get('reasoning_content') or ''
                ids = choice.get('token_ids')
                if text or reasoning or ids:
                    row['events'].append({'monotonic': now, 'token_ids': ids, 'content': text, 'reasoning': reasoning})
                if choice.get('finish_reason'):
                    row['finish_reason'] = choice['finish_reason']
    row['ended_monotonic'] = time.monotonic()
    row['content'] = ''.join(e['content'] for e in row['events'])
    try:
        parsed = json.loads(row['content'])
        correct = parsed == {'numbers': list(range(1, 301))} and all(type(x) is int for x in parsed['numbers'])
    except (ValueError, TypeError, KeyError):
        correct = False
    row['success'] = bool(done and correct and row.get('finish_reason') == 'stop' and row.get('usage', {}).get('prompt_tokens') == actual)
    after = baseline.metrics(BASE + '/metrics', MODEL)
    result = {'target': target, 'actual_prompt_tokens': actual, 'repeat': repeat, 'concurrency': 1, 'content_class': 'structured',
        'cache': 'cold', 'payload': payload, 'row': row, 'chunks': chunks,
        'before': before, 'after': after, 'summary': summarize([row], row['ended_monotonic'] - started)}
    save(f'{target}-{repeat}.json', result)
    assert row['success'], 'Incorrect or incomplete final response; result preserved'
    assert after['idle'], 'Unexpected concurrent activity'
    return {'target': target, 'repeat': repeat, 'summary': result['summary'], 'usage': row['usage']}


if __name__ == '__main__':
    save('manifest.json', {'files': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
        for n in ['probe.py', 'run.py', 'timing.py', 'native_metrics.py']},
        'limits': 'C1 structured generation only; counting task is not a general quality benchmark. Warmup rows stay separate.'})
    results = []
    try:
        for target in (1024, 4096, 16384, 65536, 131072, 200000, 260096):
            for repeat in ('warmup', 'measured1', 'measured2'):
                save('status.json', {'phase': 'running', 'target': target, 'repeat': repeat})
                results.append(cell(target, repeat))
                save('results.json', results)
        save('status.json', {'phase': 'completed'})
    except Exception as error:
        save('status.json', {'phase': 'failed', 'error': repr(error)})
        raise
