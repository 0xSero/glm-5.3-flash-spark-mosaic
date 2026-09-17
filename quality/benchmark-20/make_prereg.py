#!/usr/bin/env python3
"""P1 pre-registration generator (CAMPAIGN-12H.md section 2).

Writes, deterministically from seed 20260912:
  samples-20260912.json  - 20 fixed benchmark samples, stratified 5 x 4 task families,
                           drawn from the EXISTING benchmark/acceptance suite (no held-out quality).
  subpanel-every8-20260912.json - every 8th scored position of the frozen 65,504-position panel.
Re-running this script must reproduce both files byte-for-byte.
"""
import hashlib, json, random
from pathlib import Path

SEED = 20260912
SESSION = Path('/Users/sero/sessions/glm53-single-spark-release-20260911')
OUT = SESSION / 'quality' / 'benchmark-20'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def nonce(sample_id):
    return hashlib.sha256(f'benchmark-20:{SEED}:{sample_id}'.encode()).hexdigest()[:16]

def code(sample_id, key):
    return hashlib.sha256(f'benchmark-20:{SEED}:{sample_id}:{key}'.encode()).hexdigest()[:16]

# ---- harness identities (all existing files; hashes pinned at pre-registration) ----
H = {
 'structured_probe': 'benchmarks/structured-sustained-20260912/probe.py',
 'structured_run':   'benchmarks/structured-sustained-20260912/run.py',
 'structured_plan':  'benchmarks/structured-sustained-20260912/plan.json',
 'reasoning_probe':  'benchmarks/reasoning-effort-check/probe.py',
 'reasoning_long':   'benchmarks/reasoning-effort-check/long_context.py',
 'accept_long':      'acceptance/long_context.py',
 'accept_run':       'acceptance/run.py',
 'behavior':         'acceptance/behavior_acceptance.py',
}
HARNESS = {k: {'path': v, 'sha256': sha(SESSION / v)} for k, v in H.items()}

DEPTHS = [1024, 4096, 16384, 65536, 131072, 200000, 260096]   # plan.json targets
COMMON = {'temperature': 0, 'model': '<served id of the arc point>', 'concurrency': 1, 'prefix_cache': 'cold'}

def pool_structured():
    rows = []
    for d in DEPTHS:
        sid = f'sd-json300-{d}'
        rows.append({'id': sid, 'family': 'structured_decode',
            'prompt_source': {'harness': HARNESS['structured_probe'], 'builder': 'probe.prepare(target)',
                'task_text': '\nReturn one JSON object with key numbers containing every integer from 1 through 300, in ascending order. Finish immediately after 300.\n',
                'filler': 'The archive contains ordinary warehouse inventory records.\n',
                'target_prompt_tokens': d, 'tolerance': 'target-4 <= actual <= target (server /tokenize count)'},
            'request': COMMON | {'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'low'},
                'output_reserve_tokens': 2048, 'stream': True, 'return_token_ids': True},
            'expected_behavior': {'content_json_equals': {'numbers': list(range(1, 301))},
                'all_integers_type_int': True, 'finish_reason': 'stop', 'server_idle_before_and_after': True},
            'scorer': 'benchmarks/verify_structured.py semantics (--last-integer 300): exact JSON equality, normal stop'})
    rows.append({'id': 'sd-toolcall-weather', 'family': 'structured_decode',
        'prompt_source': {'harness': HARNESS['behavior'], 'case': 'tools',
            'prompt_text': 'Use the weather tool for Paris, France.',
            'tools': 'get_weather(city:string, country:string) as defined in behavior_acceptance.cases()', 'tool_choice': 'auto'},
        'request': COMMON | {'chat_template_kwargs': {'enable_thinking': True}},
        'expected_behavior': {'tool_calls_count': 1, 'function_name': 'get_weather',
            'arguments_json_equals': {'city': 'Paris', 'country': 'France'}, 'finish_reason_in': ['stop', 'tool_calls']},
        'scorer': 'behavior_acceptance.evaluate(name="tools")'})
    return rows

def pool_retrieval():
    rows = []
    for d in [4096, 16384, 65536, 131072, 262016]:
        sid = f'rl-audit3-{d}'
        rows.append({'id': sid, 'family': 'retrieval_long_context',
            'prompt_source': {'harness': HARNESS['reasoning_long'], 'builder': 'long_context.prepare(target)',
                'lead': 'Read the inventory below and retrieve the three audit codes.\n',
                'filler': 'The warehouse log records ordinary sealed cartons. Inventory checks found no exceptions.\n',
                'needle_fractions': [0.05, 0.5, 0.95], 'needle_keys': ['amber', 'cobalt', 'jade'],
                'question': '\nReturn a JSON object with exactly amber, cobalt, jade and their audit codes.',
                'target_prompt_tokens': d, 'tolerance': 'exact server /tokenize count == target',
                'note': 'targets other than 4096/262016 are parameterizations of the same unchanged harness (probe(target))'},
            'pre_registered_codes': {'amber': 'V7K2Q9', 'cobalt': 'N4R8P3', 'jade': 'T6M1W5'},
            'request': COMMON | {'chat_template_kwargs': {'reasoning_effort': 'low', 'enable_thinking': True},
                'response_format': {'type': 'json_object'}, 'output_reserve_tokens': 128, 'stream': True},
            'expected_behavior': {'content_json_equals': {'amber': 'V7K2Q9', 'cobalt': 'N4R8P3', 'jade': 'T6M1W5'},
                'finish_reason': 'stop', 'prompt_tokens_equals_target': True, 'cached_tokens_in': [None, 0]},
            'scorer': 'exact JSON equality of the three codes + normal stop + exact token accounting'})
    sid = 'rl-registry4-262016'
    exp = {k: code(sid, k) for k in ('amber', 'birch', 'cobalt', 'dahlia')}
    rows.append({'id': sid, 'family': 'retrieval_long_context',
        'prompt_source': {'harness': HARNESS['accept_long'], 'builder': 'long_context.prepare_retrieval(tokenizer, 262144-2048, template)',
            'needle_fractions': [0.05, 0.35, 0.65, 0.95], 'needle_keys': ['amber', 'birch', 'cobalt', 'dahlia'],
            'record_format': 'Registry record: {name} has retrieval value {value}.',
            'question': '\nReturn only a JSON object with keys amber, birch, cobalt, dahlia and their exact registry values.\n',
            'target_prompt_tokens': 262016, 'tolerance': 'target-4 <= actual <= target',
            'determinism': 'harness draws secrets.token_hex; runner must substitute the pre-registered trial nonce and values below and hash the patched file in the receipt'},
        'pre_registered_trial_nonce': nonce(sid), 'pre_registered_values': exp,
        'request': COMMON | {'chat_template_kwargs': {'enable_thinking': True}, 'output_reserve_tokens': 2048, 'stream': True},
        'expected_behavior': {'content_json_equals': exp, 'finish_reason': 'stop',
            'context_accounting': 'prompt in [262144-2048-4, 262144-2048], prompt+completion <= 262144', 'cached_tokens_in': [None, 0]},
        'scorer': 'acceptance/long_context.evaluate(): semantic_retrieval_pass and context_accounting_pass and idle_after'})
    return rows

def pool_reasoning():
    rows = []
    for effort in ('low', 'medium', 'high'):
        for jm in (False, True):
            sid = f'rs-6x7-{effort}-{"json" if jm else "text"}'
            req = COMMON | {'chat_template_kwargs': {'reasoning_effort': effort, 'enable_thinking': True}, 'output_reserve_tokens': 512}
            if jm: req['response_format'] = {'type': 'json_object'}
            rows.append({'id': sid, 'family': 'reasoning',
                'prompt_source': {'harness': HARNESS['reasoning_probe'], 'invocation': f'probe.py {effort}' + (' json' if jm else ''),
                    'prompt_text': 'Return only a JSON object with keys answer and numbers. answer must be 6 times 7, and numbers must be the integers from 1 through 20 in order. No other text.',
                    'note': 'effort is the argv parameter of the unchanged harness; low/json variants are the already-recorded cells'},
                'request': req,
                'expected_behavior': {'content_json_equals': {'answer': 42, 'numbers': list(range(1, 21))}, 'finish_reason': 'stop',
                    'note': 'raw content must parse as JSON (no code fences); the recorded low/text cell failed exactly this way on K2 FP8 r4'},
                'scorer': 'probe.py semantic_pass and finish_reason == stop'})
    return rows

def pool_generation():
    rows = []
    for d in DEPTHS:
        sid = f'gn-tips240-{d}'
        rows.append({'id': sid, 'family': 'generation',
            'prompt_source': {'harness': HARNESS['accept_run'], 'builder': 'run.prepare(tokenizer, target, nonce, template)',
                'prefix': 'Independent run {nonce}. Treat this archive as background context.\n',
                'filler': 'The archive records a cache entry. A reader verifies the revision before reuse. Workers preserve unrelated records.\n',
                'task_text': '\nWrite 240 numbered Python comment lines with distinct, concrete debugging tips. Finish after line 240.\n',
                'target_prompt_tokens': d, 'tolerance': 'target-4 <= actual <= target'},
            'pre_registered_nonce': nonce(sid),
            'request': COMMON | {'chat_template_kwargs': {'enable_thinking': True}, 'output_reserve_tokens': 2048, 'stream': True, 'return_token_ids': True},
            'expected_behavior': {'finish_reason': 'stop', 'content_lines_starting_with_hash': 240,
                'lines_numbered_1_to_240_in_order': True, 'distinct_lines': 240,
                'note': 'natural stop within the harness reserve; finish_reason=length is a FAIL for this sample (it is a pass only for the speed screen)'},
            'scorer': 'parse content: exactly 240 lines matching ^#\\s*(\\d+)[.):]?\\s+\\S, numbers 1..240 ascending, no duplicate line text, finish_reason stop'})
    for name, prompt, expect in (
        ('text', 'Return exactly FINAL_BASELINE_OK as your final answer, with no punctuation or surrounding text.', 'FINAL_BASELINE_OK'),
        ('arabic', 'أجب فقط بالعبارة التالية دون علامات اقتباس: مرحبا بالعالم', 'مرحبا بالعالم'),
        ('chinese', '只回答以下文字，不要加引号或其他内容：你好，世界', '你好，世界'),
        ('polish', 'Odpowiedz wyłącznie tym tekstem, bez cudzysłowu: Witaj, świecie', 'Witaj, świecie')):
        rows.append({'id': f'gn-behavior-{name}', 'family': 'generation',
            'prompt_source': {'harness': HARNESS['behavior'], 'case': name, 'prompt_text': prompt},
            'request': COMMON | {'chat_template_kwargs': {'enable_thinking': True}},
            'expected_behavior': {'content_equals': expect, 'finish_reason': 'stop'},
            'scorer': f'behavior_acceptance.evaluate(name="{name}")'})
    return rows

POOLS = [('structured_decode', pool_structured()), ('retrieval_long_context', pool_retrieval()),
         ('reasoning', pool_reasoning()), ('generation', pool_generation())]

def draw():
    rng = random.Random(SEED)
    chosen, ledger = [], []
    for fam, pool in POOLS:          # fixed family order; 5 per family; pools sorted by id before sampling
        ids = sorted(r['id'] for r in pool)
        pick = rng.sample(ids, 5)    # random.Random(20260912).sample on the sorted id list, in this family order
        ledger.append({'family': fam, 'pool_size': len(ids), 'pool_ids': ids, 'selected_ids': pick})
        byid = {r['id']: r for r in pool}
        chosen += [byid[i] for i in pick]
    return chosen, ledger

def main():
    samples, ledger = draw()
    assert len(samples) == 20 and len({s['id'] for s in samples}) == 20
    fixture = json.loads((SESSION / 'quality-eval' / 'manifest.json').read_text())
    doc = {
        'schema': 'glm53-benchmark-20-prereg-v1',
        'campaign': 'CAMPAIGN-12H section 2 / P1 pre-registration',
        'created_utc': '2026-09-12T22:49:11Z',
        'seed': SEED,
        'draw_rule': 'For each family in the order [structured_decode, retrieval_long_context, reasoning, generation]: sort pool ids, then random.Random(20260912).sample(sorted_ids, 5) with ONE shared generator across families (state carried in order). Regenerate with make_prereg.py.',
        'generator_sha256_note': 'make_prereg.py hash is recorded in the CAMPAIGN-12H ledger receipt, not here (self-reference).',
        'selection_uses_heldout_quality': False,
        'heldout_panel_excluded': {
            'excluded_source': 'benchmarks/*-serving-quality*/ and quality-eval/ (frozen WikiText-2 panel, 65,504 positions)',
            'fixture_manifest_sha256': sha(SESSION / 'quality-eval' / 'manifest.json'),
            'token_rows_sha256': fixture['token_rows_sha256'],
            'statement': 'No sample prompt is drawn from, derived from, or overlaps the held-out quality panel text.'},
        'scoring': {
            'per_sample': 'binary pass/fail per expected_behavior + scorer',
            'score(point)': 'passes / 20',
            'retention': 'score(point) / score(denominator); denominators per CAMPAIGN-12H section 4.4 (A1 vs A0; A2d/A2/controls vs A0, labelled)',
            'per_family_breakdown': 'report passes per family (5 each) alongside the total; families are not weighted',
            'runtime': 'each sample is one isolated C1 request on the digest-pinned image, server idle before/after, cold prefix cache, temperature 0',
            'output_reserve_note': 'output_reserve_tokens are the existing harness context-accounting reserves, unchanged; every expected answer completes with finish_reason=stop well inside them'},
        'harness_identities': HARNESS,
        'families': ledger,
        'samples': samples,
    }
    text = json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=False) + '\n'
    (OUT / 'samples-20260912.json').write_text(text, encoding='utf-8')

    # ---- sub-panel: every 8th scored position of the frozen panel ----
    ROWS, COLS = 32, 2048
    scored_per_row = COLS - 1          # 2047: position (row, c) predicts token (row, c+1), c = 0..2046
    total = ROWS * scored_per_row      # 65,504
    idx = list(range(0, total, 8))     # flat = row*2047 + col
    assert len(idx) == 8188 and total == 65504
    per_row = {}
    for f in idx:
        per_row.setdefault(f // scored_per_row, []).append(f % scored_per_row)
    canon = json.dumps(idx, separators=(',', ':')).encode()
    sub = {
        'schema': 'glm53-kl-sweep-subpanel-v1',
        'campaign': 'CAMPAIGN-12H section 2 / P1 pre-registration',
        'created_utc': '2026-09-12T22:49:11Z',
        'purpose': 'Per-layer KL sensitivity sweep only (M1). Never used for reported numbers. sweep_subpanel_overlaps_panel=true.',
        'parent_panel': {
            'description': 'Frozen WikiText-2 held-out panel: 32 rows x 2048 tokens; 2047 scored next-token positions per row; 65,504 scored positions.',
            'dataset': f"{fixture['dataset_repo']} {fixture['dataset_config']} split={fixture['dataset_split']} revision={fixture['dataset_revision']}",
            'selection_policy': fixture['selection_policy'],
            'fixture_manifest_sha256': sha(SESSION / 'quality-eval' / 'manifest.json'),
            'token_rows_sha256': fixture['token_rows_sha256'],
            'raw_text_sha256': fixture['raw_text_sha256'],
            'teacher_manifest_sha256': 'a1604015f76d6a6b6f032ffb7e3359a770e8dd2564ca89e3b00eb8bcb4d93314',
            'teacher': 'zai-org/GLM-5.3-Flash-BF16 @ a6c167b62691b2bac901344b65cb651a70f53e43',
            'evaluator': {'adapter': 'quality/evaluate_pruned_quality.py', 'adapter_sha256': sha(SESSION / 'quality' / 'evaluate_pruned_quality.py'),
                          'baseline': 'quality/deps/evaluate_low_bpw_quality.py', 'baseline_sha256': sha(SESSION / 'quality' / 'deps' / 'evaluate_low_bpw_quality.py')},
            'position_contract': 'hidden[row, c] (c=0..2046) -> BF16 lm_head -> FP32 logits, label = input_ids[row, c+1]; flat index = row*2047 + c; KL is BF16-teacher-to-candidate over the full 154,880-row head, vocab chunks of 4096.'},
        'derivation_rule': 'flat index f in {0, 8, 16, ..., 65496}: f = 8*k, k = 0..8187; row = f // 2047, col = f % 2047. Every 8th scored position in flat row-major order over the 65,504 positions.',
        'positions': 8188,
        'stride': 8, 'offset': 0,
        'per_row_counts': {str(r): len(per_row[r]) for r in range(ROWS)},
        'subpanel_index_sha256': hashlib.sha256(canon).hexdigest(),
        'subpanel_index_sha256_input': 'sha256 of json.dumps(flat_indices, separators=(",",":")) as UTF-8',
        'metric_on_subpanel': 'mean over the 8,188 positions of per-position KL(teacher || candidate); delta-KL per layer = mean_sub(masked) - mean_sub(unmasked) on A2d',
        'flat_indices': idx,
        'per_row_cols': {str(r): per_row[r] for r in range(ROWS)},
    }
    (OUT / 'subpanel-every8-20260912.json').write_text(json.dumps(sub, indent=1) + '\n', encoding='utf-8')
    print('samples', len(samples), 'subpanel', len(idx))

if __name__ == '__main__':
    main()
