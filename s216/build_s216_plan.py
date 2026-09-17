#!/usr/bin/env python3
"""S216 plan: UNIFORM keep-216 prune of the 3.05bpw base ranked by sensitivity x log(routes+1).

sensitivity[l][e] = mean of the paired K3 per-projection reconstruction-error proxies (gate/up/down) for expert e of
layer l (k3-projection-errors.json, 12,096 experts, layers 3..44).  routes[l][e] = fresh top-8 routing counts captured
from the served unpruned 2.05bpw exl3_plain model on spark-2384 (SGLang expert-distribution recorder, stat mode) over a
domain-stratified REAP calibration sample plus generated continuations.  score = sensitivity * ln(routes + 1); the
216 highest-score experts per layer are kept, sorted ascending and renumbered contiguously.  MTP layer 45 is kept verbatim.
Output schema matches the U216 plan so prune_u216.py / verify_u216.py / census_plan_aware.py apply unchanged.
"""
import argparse, hashlib, json, math, time
from pathlib import Path
LAYERS = tuple(range(3, 45)); N = 288
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok, msg):
    if not ok: raise ValueError(msg)
ap = argparse.ArgumentParser()
ap.add_argument('--k3', required=True); ap.add_argument('--routing', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--u216-plan', default=None, help='optional: report overlap with the massmax U216 keep sets')
ap.add_argument('--keep', type=int, default=216); ap.add_argument('--expert-bytes', type=int, default=9474060); ap.add_argument('--router-row-bytes', type=int, default=8196)
a = ap.parse_args(); K = a.keep
k3 = json.loads(Path(a.k3).read_text())['parts']
sens = {l: [None] * N for l in LAYERS}
for p in k3:
    for r in p['reports']:
        pr = r['proxy']; require(set(pr) == {'gate_proj', 'up_proj', 'down_proj'}, 'bad proxy keys')
        require(sens[p['layer']][r['expert']] is None, 'duplicate expert'); sens[p['layer']][r['expert']] = (pr['gate_proj'] + pr['up_proj'] + pr['down_proj']) / 3
require(all(v is not None for l in LAYERS for v in sens[l]), 'incomplete K3 coverage')
rt = json.loads(Path(a.routing).read_text()); counts = rt['counts']
require(set(map(int, counts)) >= set(LAYERS), f'routing layers {sorted(map(int, counts))} do not cover 3..44')
routes = {l: [int(x) for x in counts[str(l)]] for l in LAYERS}
require(all(len(routes[l]) == N for l in LAYERS), 'routing width != 288')
score = {l: [sens[l][e] * math.log(routes[l][e] + 1) for e in range(N)] for l in LAYERS}
ranking = {l: sorted(range(N), key=lambda e: (-score[l][e], e)) for l in LAYERS}
ids = {str(l): sorted(ranking[l][:K]) for l in LAYERS}; pruned = {str(l): sorted(ranking[l][K:]) for l in LAYERS}
removed = len(LAYERS) * (N - K)
detail = {}; tot_routes_kept = 0; tot_routes = 0; tot_sens_kept = 0.0; tot_sens = 0.0; overlap = {}
u216 = json.loads(Path(a.u216_plan).read_text())['retained_original_ids_by_layer'] if a.u216_plan else None
for l in LAYERS:
    kept = ids[str(l)]; rk = sum(routes[l][e] for e in kept); ra = sum(routes[l]); sk = sum(sens[l][e] for e in kept); sa = sum(sens[l])
    tot_routes_kept += rk; tot_routes += ra; tot_sens_kept += sk; tot_sens += sa
    zero = sum(1 for e in range(N) if routes[l][e] == 0)
    detail[str(l)] = {'keep': K, 'route_mass_retention': rk / ra, 'sensitivity_retention': sk / sa, 'zero_route_experts': zero,
                      'min_routes_kept': min(routes[l][e] for e in kept), 'max_routes_pruned': max(routes[l][e] for e in pruned[str(l)]),
                      'score_threshold': score[l][ranking[l][K - 1]]}
    if u216: overlap[str(l)] = len(set(kept) & set(u216[str(l)]))
plan = {
    'schema': 'dynamic-container-plan-v1', 'name': 'S216', 'state': 'FROZEN_PLAN',
    'point': {'bpw': 3.05, 'prune': removed / (N * len(LAYERS)), 'removed_experts': removed, 'retained_experts': K * len(LAYERS),
              'mean_keep': K, 'ranking': 'k3_sensitivity_x_log_routes', 'uniform_keep': K, 'layer_floor': K, 'block': None,
              'removal_count_source': f'uniform keep {K} per MoE layer x {len(LAYERS)} layers (SGLang requires a single n_routed_experts)'},
    'base': {'repo': 'turboderp/GLM-5.3-Flash-exl3', 'branch': '3.05bpw', 'revision': '332ab457b709b7ba30dd9a448be5de03b80a7ac9', 'total_bytes': 125263712676},
    'inputs_sha256': {'k3-projection-errors.json': sha(a.k3), 'routing-counts.json': sha(a.routing)},
    'routing_capture': {k: rt[k] for k in ('total_routes', 'per_batch_routes', 'receipt') if k in rt} | {'source_model': 'turboderp/GLM-5.3-Flash-exl3 2.05bpw (unpruned) served by exl3_plain on spark-2384',
                        'note': 'unpruned 3.05bpw (~127 GB of weights) does not fit a 121 GB Spark; router weights are unquantised and identical across bpw branches'},
    'allocation_proxy': 'none (uniform); within-layer ranking = mean(K3 gate/up/down proxy error) x ln(routes+1)',
    'allocator': 'uniform top-K of combined score, ties lower id',
    'selection_uses_heldout_quality': False,
    'keep_by_layer': {str(l): K for l in LAYERS},
    'retained_original_ids_by_layer': ids, 'pruned_original_ids_by_layer': pruned,
    'original_id_to_contiguous_id': {str(l): {str(e): i for i, e in enumerate(ids[str(l)])} for l in LAYERS},
    'scores_by_layer': {str(l): score[l] for l in LAYERS}, 'sensitivity_by_layer': {str(l): sens[l] for l in LAYERS}, 'routes_by_layer': {str(l): routes[l] for l in LAYERS},
    'layer_details': detail, 'overlap_with_u216_keep': overlap,
    'summary': {'min_keep': K, 'max_keep': K, 'route_mass_retained': tot_routes_kept / tot_routes, 'sensitivity_retained': tot_sens_kept / tot_sens,
                'mean_overlap_with_u216': (sum(overlap.values()) / len(overlap)) if overlap else None},
    'byte_model': {'expert_payload_bytes_each': a.expert_bytes, 'router_row_bytes_each': a.router_row_bytes,
                   'removed_tensor_bytes': removed * (a.expert_bytes + a.router_row_bytes),
                   'projected_index_total_size': 122286101684 - removed * (a.expert_bytes + a.router_row_bytes)},
    'text_config_patch': {'n_routed_experts': K, 'routed_experts_per_layer': {str(l): K for l in LAYERS}, 'retained_expert_ids_by_layer': ids},
    'mtp_layer': {'layer': 45, 'experts': 288, 'policy': 'kept verbatim (mtp.safetensors byte-identical); no K3 proxy or routing exists for layer 45'},
    'protected_policy': 'Every non-target tensor byte-exact, including all MTP tensors and all 288 MTP experts; only target router rows and bias entries follow retained original IDs.',
    'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
body = json.dumps(plan, indent=1, sort_keys=True); plan['plan_sha256'] = hashlib.sha256(body.encode()).hexdigest()
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
if out.exists(): out.chmod(0o644)
out.write_text(json.dumps(plan, indent=1, sort_keys=True) + '\n'); out.chmod(0o444)
print(json.dumps({'plan_sha256': plan['plan_sha256'], 'point': plan['point'], 'summary': plan['summary'],
                  'layer_zero_route_experts': {l: d['zero_route_experts'] for l, d in detail.items() if d['zero_route_experts']}}, indent=1))
