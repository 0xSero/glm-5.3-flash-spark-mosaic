#!/usr/bin/env python3
"""A2d plan: mass-greedy non-uniform prune of the 3.05bpw base, keep-216 class.

Allocator is the plan-v1 reference (build_plan.py::allocation) run with
floor=184, block=8, budget=3016 removals (campaign §3 pre-registered count).
Within-layer ranking is the sealed massmax_domain order; retained ids are
sorted ascending and renumbered contiguously.  CPU only, no held-out feedback.
"""
import argparse, hashlib, heapq, json, math, sys, time
from pathlib import Path

LAYERS = tuple(range(3, 45)); N = 288
EXPECTED = {
    'seal.json': 'bb2d12f209d2606319a0f1e3c40b0f54db2d4d3e9d40cfe49779b510f3c5596a',
    'observations.json': 'b2e36095306132b24a68accd0263fa42cf319d335b1533e4c85e9348a618cabf',
    'candidates.json': '4fd8de1e8d0d3e05158b0989a6b18813430f30b280f5499870e7e75b06d02ecd',
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def require(ok, msg):
    if not ok: raise ValueError(msg)

def allocation(scores, rankings, budget, floor, block=8):
    # verbatim semantics of plan-v1 build_plan.py::allocation (heap on (cost, layer): ties -> lower layer)
    require(budget % block == 0, 'budget must be divisible by block')
    require(8 <= floor <= N and floor % block == 0, 'invalid floor')
    require(0 <= budget <= len(scores) * (N - floor), 'infeasible removal budget')
    counts = {l: N for l in scores}; normalized = {}; heap = []; ledger = []
    for layer in sorted(scores):
        values = scores[layer]; rank = rankings[layer]
        require(len(values) == N and sorted(rank) == list(range(N)), 'invalid expert coverage')
        require(all(math.isfinite(v) and v >= 0 for v in values) and sum(values) > 0, 'invalid scores')
        require(rank == sorted(range(N), key=lambda e: (-values[e], e)), 'ranking differs from massmax criterion')
        normalized[layer] = [v / sum(values) for v in values]
        if N > floor: heapq.heappush(heap, (sum(normalized[layer][e] for e in rank[N - block:N]), layer))
    for step in range(budget // block):
        cost, layer = heapq.heappop(heap); old = counts[layer]; new = old - block; counts[layer] = new
        ledger.append({'step': step, 'layer': layer, 'from': old, 'to': new,
                       'removed_original_ids': sorted(rankings[layer][new:old]), 'normalized_proxy_removed': cost})
        if new > floor: heapq.heappush(heap, (sum(normalized[layer][e] for e in rankings[layer][new - block:new]), layer))
    return counts, ledger, normalized

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--removed', type=int, default=3016)
    ap.add_argument('--floor', type=int, default=184)
    ap.add_argument('--expert-bytes', type=int, default=9474060, help='per-expert packed bytes from 3.05 headers')
    ap.add_argument('--router-row-bytes', type=int, default=8196)
    ap.add_argument('--regress', action='store_true', help='reproduce plan-v1 ledger (floor 240, 672 removals)')
    a = ap.parse_args()
    sealed = a.root / 'sealed-observations'; inputs = {}
    for n, d in EXPECTED.items():
        require(sha(sealed / n) == d, f'sealed observations changed: {n}'); inputs['sealed-observations/' + n] = d
    seal = read(sealed / 'seal.json'); obs = read(sealed / 'observations.json'); cand = read(sealed / 'candidates.json')
    rows = cand['selection']['massmax_domain']; require(set(rows) == set(map(str, LAYERS)), 'wrong layer coverage')
    scores = {}; rankings = {}; domain_scores = {}
    for l in LAYERS:
        k = str(l); scores[l] = rows[k]['scores']; rankings[l] = rows[k]['ranked_experts_high_to_low']; domain_scores[l] = {}
        for domain, group in obs['domain_observations'].items():
            mass = group[k]['weighted_ean_sum']; denom = sum(mass); require(denom > 0, 'empty domain mass')
            domain_scores[l][domain] = [v / denom for v in mass]
        rec = [max(s[e] for s in domain_scores[l].values()) for e in range(N)]
        require(all(math.isclose(x, y, rel_tol=1e-13, abs_tol=1e-15) for x, y in zip(scores[l], rec)), 'sealed score does not reproduce')
    if a.regress:
        counts, ledger, _ = allocation(scores, rankings, 672, 240)
        ref = read(a.root / 'quality/nonuniform-pruning-study/plan-v1/allocation-ledger.json')
        require(ledger == ref, 'regression: plan-v1 ledger not reproduced'); print('regression OK: plan-v1 ledger reproduced bit-for-bit'); return
    counts, ledger, normalized = allocation(scores, rankings, a.removed, a.floor)
    ids = {str(l): sorted(rankings[l][:counts[l]]) for l in LAYERS}
    pruned = {str(l): sorted(rankings[l][counts[l]:]) for l in LAYERS}
    removed = sum(N - c for c in counts.values()); require(removed == a.removed, 'budget mismatch')
    detail = {}; pooled_kept = 0.0; pooled_tot = 0.0
    for l in LAYERS:
        kept = ids[str(l)]; pool = obs['observations'][str(l)]['weighted_ean_sum']
        pooled_kept += sum(pool[e] for e in kept); pooled_tot += sum(pool)
        detail[str(l)] = {'keep': counts[l], 'normalized_proxy_removed': sum(normalized[l][e] for e in pruned[str(l)]),
                          'pooled_mass_retention': sum(pool[e] for e in kept) / sum(pool),
                          'minimum_domain_mass_retention': min(sum(s[e] for e in kept) for s in domain_scores[l].values())}
    uniform_kept = 0.0
    for l in LAYERS:
        pool = obs['observations'][str(l)]['weighted_ean_sum']; uniform_kept += sum(pool[e] for e in rankings[l][:216])
    plan = {
        'schema': 'dynamic-container-plan-v1', 'name': 'A2d', 'state': 'FROZEN_PLAN',
        'point': {'bpw': 3.05, 'prune': removed / (N * len(LAYERS)), 'removed_experts': removed, 'retained_experts': sum(counts.values()),
                  'mean_keep': sum(counts.values()) / len(LAYERS), 'ranking': 'massmax_domain', 'layer_floor': a.floor, 'block': 8,
                  'removal_count_source': 'CAMPAIGN-12H.md §3 pre-registered A2d/A2 removal count (B_arc=96 GB planning prior); no measured B_arc exists'},
        'base': {'repo': 'turboderp/GLM-5.3-Flash-exl3', 'branch': '3.05bpw', 'revision': '332ab457b709b7ba30dd9a448be5de03b80a7ac9', 'total_bytes': 125263712676},
        'inputs_sha256': inputs, 'observation_model_identity': cand['model_identity'],
        'allocation_proxy': 'Within-layer massmax_domain normalized to unit sum; cross-layer sum is heuristic, not measured counterfactual sensitivity.',
        'allocator': 'plan-v1 build_plan.py::allocation (heap on (cost, layer); ties -> lower layer index)',
        'selection_uses_heldout_quality': False,
        'keep_by_layer': {str(l): counts[l] for l in LAYERS},
        'retained_original_ids_by_layer': ids, 'pruned_original_ids_by_layer': pruned,
        'original_id_to_contiguous_id': {str(l): {str(e): i for i, e in enumerate(ids[str(l)])} for l in LAYERS},
        'layer_details': detail, 'ledger': ledger,
        'summary': {'min_keep': min(counts.values()), 'max_keep': max(counts.values()),
                    'normalized_proxy_removed_total': sum(d['normalized_proxy_removed'] for d in detail.values()),
                    'pooled_mass_retained': pooled_kept / pooled_tot, 'pooled_mass_retained_uniform216': uniform_kept / pooled_tot},
        'byte_model': {'expert_payload_bytes_each': a.expert_bytes, 'router_row_bytes_each': a.router_row_bytes,
                       'removed_tensor_bytes': removed * (a.expert_bytes + a.router_row_bytes),
                       'projected_index_total_size': 122286101684 - removed * (a.expert_bytes + a.router_row_bytes),
                       'projected_total_files_bytes_approx': 125263712676 - removed * (a.expert_bytes + a.router_row_bytes)},
        'text_config_patch': {'n_routed_experts': 288, 'routed_experts_per_layer': {str(l): counts[l] for l in LAYERS}, 'retained_expert_ids_by_layer': ids},
        'protected_policy': 'Every non-target tensor byte-exact, including all MTP tensors and all 288 MTP experts (mtp.safetensors copied verbatim); only target router rows and bias entries follow retained original IDs.',
        'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    body = json.dumps(plan, indent=1, sort_keys=True)
    plan['plan_sha256'] = hashlib.sha256(body.encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(plan, indent=1, sort_keys=True) + '\n'); a.out.chmod(0o444)
    print(json.dumps({'plan_sha256': plan['plan_sha256'], 'point': plan['point'], 'summary': plan['summary'], 'byte_model': plan['byte_model'],
                      'keep_by_layer': plan['keep_by_layer']}, indent=1))
if __name__ == '__main__': main()
