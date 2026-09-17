"""Bounded multi-needle API acceptance, with exact server-tokenized budgets."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

BASE = 'http://127.0.0.1:18080'
MODEL = 'glm-5.3-flash'
ROOT = Path(os.environ['PROBE_OUTPUT'])
ROOT.mkdir(parents=True, exist_ok=True)
TEMPLATE = {'reasoning_effort': 'low', 'enable_thinking': True}
EXPECTED = {'amber': 'V7K2Q9', 'cobalt': 'N4R8P3', 'jade': 'T6M1W5'}
FILLER = 'The warehouse log records ordinary sealed cartons. Inventory checks found no exceptions.\n'
RESERVE = 128


def save(name, value):
    temporary = ROOT / (name + '.tmp')
    serialized = json.dumps(value, indent=2, allow_nan=False)
    try:
        temporary.write_text(serialized)
        temporary.replace(ROOT / name)
    except OSError as error:
        if error.errno != 28:
            raise
        fallback = Path('/dev/shm') / ('glm53-context-' + name)
        fallback.write_text(serialized)
        print('RESULT_FALLBACK=' + str(fallback), flush=True)


def request(path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    return urllib.request.urlopen(urllib.request.Request(
        BASE + path, data=data, headers={'Content-Type': 'application/json'}), timeout=timeout)


def metrics():
    with request('/metrics') as response:
        text = response.read().decode()
    gauges = [line for line in text.splitlines() if line.startswith((
        'vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
    assert len(gauges) == 2 and all(float(line.rsplit(' ', 1)[1]) == 0 for line in gauges), gauges
    return text


def tokenize(prompt):
    with request('/tokenize', {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
                              'add_generation_prompt': True, 'chat_template_kwargs': TEMPLATE}) as response:
        return json.load(response)['count']


def prepare(target):
    def body(repeats, padding=0):
        parts = ['Read the inventory below and retrieve the three audit codes.\n']
        previous = 0
        for fraction, (key, value) in zip((.05, .5, .95), EXPECTED.items()):
            boundary = int(repeats * fraction)
            parts.extend([FILLER * (boundary - previous), f'\nAUDIT CODE {key}: {value}\n'])
            previous = boundary
        parts.extend([FILLER * (repeats - previous), ' x' * padding,
                      '\nReturn a JSON object with exactly amber, cobalt, jade and their audit codes.'])
        return ''.join(parts)
    unit = max(1, tokenize(FILLER) - tokenize(''))
    low, high = 0, target // unit + 32
    while tokenize(body(high)) < target:
        high *= 2
    while low < high:
        mid = (low + high + 1) // 2
        if tokenize(body(mid)) <= target:
            low = mid
        else:
            high = mid - 1
    prompt = body(low)
    padding = target - tokenize(prompt)
    for _ in range(8):
        prompt = body(low, padding)
        count = tokenize(prompt)
        if count == target:
            return prompt, count
        padding += target - count
        assert padding >= 0
    raise ValueError('Exact prompt token budget could not be established')


def probe(target):
    save('progress.json', {'phase': 'preparing', 'target_prompt_tokens': target, 'time': time.time()})
    prompt, count = prepare(target)
    payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
               'temperature': 0, 'max_tokens': RESERVE, 'chat_template_kwargs': TEMPLATE,
               'response_format': {'type': 'json_object'}, 'stream': True,
               'stream_options': {'include_usage': True}}
    before = metrics()
    save('progress.json', {'phase': 'request_active', 'prompt_tokens': count, 'reserve': RESERVE,
                           'time': time.time(), 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
    started = time.monotonic()
    first = None
    chunks, content, reasoning = [], [], []
    finish, usage, done = None, None, False
    with request('/v1/chat/completions', payload, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            chunk = json.loads(data)
            chunks.append(chunk)
            if chunk.get('error'):
                raise ValueError(chunk['error'])
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices', []):
                delta = choice.get('delta') or {}
                text = delta.get('content') or ''
                thought = delta.get('reasoning') or delta.get('reasoning_content') or ''
                if (text or thought) and first is None:
                    first = time.monotonic()
                content.append(text)
                reasoning.append(thought)
                finish = choice.get('finish_reason') or finish
    elapsed = time.monotonic() - started
    try:
        correct = json.loads(''.join(content)) == EXPECTED
    except ValueError:
        correct = False
    accepted = bool(done and finish == 'stop' and correct and usage and usage['prompt_tokens'] == count)
    result = {'accepted': accepted, 'prompt_tokens': count, 'output_reserve': RESERVE,
              'total_budget': count + RESERVE, 'usage': usage, 'finish_reason': finish,
              'elapsed_seconds': elapsed, 'ttft_seconds': None if first is None else first - started,
              'content': ''.join(content), 'reasoning': ''.join(reasoning), 'chunks': chunks,
              'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'template': TEMPLATE, 'metrics_before': before, 'metrics_after': metrics(),
              'limits': 'Synthetic three-code retrieval only; does not establish broad long-context quality.'}
    save(str(target) + '.json', result)
    return accepted


try:
    if os.environ.get('PROBE_FULL_ONLY') == '1':
        prior = json.loads(Path(os.environ['PROBE_PRIOR_SHORT']).read_text())
        assert prior['accepted'] and prior['prompt_tokens'] == 4096
    else:
        assert probe(4096), '4096-token smoke failed; full-context test not started'
    probe(262144 - RESERVE)
    save('progress.json', {'phase': 'completed', 'time': time.time()})
except Exception as error:
    save('error.json', {'type': type(error).__name__, 'message': str(error), 'time': time.time()})
    raise
