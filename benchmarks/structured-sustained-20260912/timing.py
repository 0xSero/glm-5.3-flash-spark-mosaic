"""Pure timing math: token IDs are never inferred from text or SSE chunk counts."""
import math
import statistics


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    x = (len(values)-1)*q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi]-values[lo])*(x-lo)


def capacity_status(prompt_tokens, output_reserve, context, concurrency, active_slots, kv_tokens):
    if prompt_tokens + output_reserve > context:
        return 'UNSUPPORTED_REQUEST_CONTEXT'
    if concurrency > active_slots:
        return 'UNSUPPORTED_ACTIVE_SLOTS'
    if concurrency * (prompt_tokens + output_reserve) > kv_tokens:
        return 'UNSUPPORTED_KV_CAPACITY'
    return 'PLANNED'


def matched_window(rows, minimum_seconds=30, maximum_silence=5):
    if not rows or not all(r.get('success') for r in rows):
        return {'status': 'UNAVAILABLE_FAILED_REQUEST'}
    if any(not r['events'] or any(e.get('token_ids') is None for e in r['events']) for r in rows):
        return {'status': 'UNAVAILABLE_EXACT_TOKEN_IDS'}
    if any(sum(len(e['token_ids']) for e in r['events']) != r['usage']['completion_tokens'] for r in rows):
        return {'status': 'UNAVAILABLE_TOKEN_ACCOUNTING_MISMATCH'}
    start = max(r['events'][0]['monotonic'] for r in rows)
    end = min(r['events'][-1]['monotonic'] for r in rows)
    if end <= start:
        return {'status': 'UNAVAILABLE_NO_COMMON_WINDOW'}
    duration = end-start
    streams = []
    for r in rows:
        events = [e for e in r['events'] if start < e['monotonic'] <= end]
        ids = [i for e in events for i in e['token_ids']]
        times = [start] + [e['monotonic'] for e in events] + [end]
        if not ids or max(b-a for a,b in zip(times,times[1:])) > maximum_silence:
            return {'status': 'UNAVAILABLE_SILENT_STREAM', 'window_seconds': duration}
        streams.append({'token_ids': ids, 'tokens': len(ids), 'decode_tok_s': len(ids)/duration})
    return {'status': 'MATCHED_SUSTAINED' if duration >= minimum_seconds else 'MATCHED_SCREEN',
            'start_monotonic':start,'end_monotonic':end,'window_seconds':duration,
            'total_aggregate_decode_tok_s':sum(s['tokens'] for s in streams)/duration,
            'mean_per_request_decode_tok_s':statistics.fmean(s['decode_tok_s'] for s in streams),
            'streams':streams,'counting_interval':'(start,end]; client arrival time; speculative bundles stay intact'}


def summarize(rows, cell_seconds, **window_options):
    result = {'requests':len(rows),'successes':sum(bool(r.get('success')) for r in rows),
              'cell_seconds':cell_seconds,'matched_decode':matched_window(rows,**window_options)}
    good = [r for r in rows if r.get('success')]
    if len(good)!=len(rows) or not good:
        return result
    ttfts=[r['events'][0]['monotonic']-r['started_monotonic'] for r in good]
    result.update(ttft_p50_seconds=percentile(ttfts,.5),ttft_p90_seconds=percentile(ttfts,.9),
                  mean_prompt_tokens_per_ttft=statistics.fmean(r['usage']['prompt_tokens']/t for r,t in zip(good,ttfts)),
                  total_output_tokens_per_end_to_end_wall_second=sum(r['usage']['completion_tokens'] for r in good)/cell_seconds)
    # This is a labelled fallback, never the matched-window total decode column.
    estimates=[]
    for r in good:
        duration=r['events'][-1]['monotonic']-r['events'][0]['monotonic']
        first=r['events'][0].get('token_ids')
        if duration>0:
            estimates.append((r['usage']['completion_tokens']-(len(first) if first is not None else 1))/duration)
    result['mean_client_decode_tok_s_estimate']=statistics.fmean(estimates) if estimates else None
    return result
