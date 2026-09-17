#!/usr/bin/env python3
"""R216 plan: UNIFORM keep-216 prune of the 3.05bpw base ranked by sensitivity x ln(routes+1) with the DENSER R216 route capture.

Queue item 3 (driver run 3). T216 refuted the 'routes are noise' hypothesis (KL 0.855 vs S216 0.801,
samples 6/20 vs 12/20), so the score keeps S216's sensitivity × ln(routes+1) form but feeds it the
denser routing-counts-r216.json: the SAME sealed prefill dumps (provenance-preserving recompute) plus
15 further generation prompts (2048-token cap each) captured from the unpruned 2.05bpw recorder on
spark-2384. Guards: S216 plan embedded-sha reproduction (sensitivity source), routing file must carry
provenance with the sealed S216 counts sha 1f0e98c8… and complete receipts. Schema matches
S216/U216/T216 so prune_u216.py / verify_u216.py / census_plan_aware.py apply unchanged. MTP layer 45
kept verbatim.
"""
import argparse, hashlib, json, math, time
from pathlib import Path
LAYERS = tuple(range(3, 45)); N = 288
SEALED_S216_COUNTS_SHA = '1f0e98c8d782f067ae575e7570074f5aec3249a481086da08912a5faed05bebb'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok, msg):
    if not ok: raise ValueError(msg)
ap = argparse.ArgumentParser()
ap.add_argument('--s216-plan', required=True, help='sealed S216 plan: source of sensitivity_by_layer + provenance chain')
ap.add_argument('--sensitivity-check', required=True, help='independent reduced copy (sensitivity.json) for the equality guard')
ap.add_argument('--routing', required=True, help='routing-counts-r216.json from combine_reduce.py (denser capture)')
ap.add_argument('--u216-plan', default=None, help='optional: overlap with the massmax U216 keep sets')
ap.add_argument('--out', required=True)
ap.add_argument('--keep', type=int, default=216); ap.add_argument('--expert-bytes', type=int, default=9474060); ap.add_argument('--router-row-bytes', type=int, default=8196)
a = ap.parse_args(); K = a.keep
s216 = json.loads(Path(a.s216_plan).read_text())
body_check = json.dumps({k: v for k, v in s216.items() if k != 'plan_sha256'}, indent=1, sort_keys=True)
require(s216['name'] == 'S216' and s216['plan_sha256'] == hashlib.sha256(body_check.encode()).hexdigest(), 'S216 plan file does not match its embedded plan_sha256')
sealed_pin = s216['inputs_sha256']['k3-projection-errors.json']
sens = {l: list(s216['sensitivity_by_layer'][str(l)]) for l in LAYERS}
require(all(len(sens[l]) == N for l in LAYERS), 'sensitivity width != 288')
chk = json.loads(Path(a.sensitivity_check).read_text())['k3']
require(set(map(int, chk)) == set(LAYERS), 'cross-check layer set mismatch')
for l in LAYERS:
    require(all(abs(chk[str(l)][e] - sens[l][e]) <= 1e-12 for e in range(N)), f'layer {l}: cross-check sensitivity disagrees with sealed plan')
rt = json.loads(Path(a.routing).read_text())
prov = rt.get('provenance') or {}
require(prov.get('sealed_s216_counts_sha256') == SEALED_S216_COUNTS_SHA, 'routing file provenance does not pin the sealed S216 counts')
require(prov.get('new_batches') and prov.get('old_batches'), 'routing file provenance lacks batch lists')
require(rt.get('total_routes', 0) > 222285, 'denser capture must exceed the sealed 222,285 prefill routes')
counts = rt['counts']
require(set(map(int, counts)) >= set(LAYERS), 'routing layers do not cover 3..44')
routes = {l: [int(x) for x in counts[str(l)]] for l in LAYERS}
require(all(len(routes[l]) == N for l in LAYERS), 'routing width != 288')
score = {l: [sens[l][e] * math.log(routes[l][e] + 1) for e in range(N)] for l in LAYERS}
ranking = {l: sorted(range(N), key=lambda e: (-score[l][e], e)) for l in LAYERS}
ids = {str(l): sorted(ranking[l][:K]) for l in LAYERS}; pruned = {str(l): sorted(ranking[l][K:]) for l in LAYERS}
removed = len(LAYERS) * (N - K)
detail = {}; tot_routes_kept = 0; tot_routes = 0; tot_sens_kept = 0.0; tot_sens = 0.0
u216 = json.loads(Path(a.u216_plan).read_text())['retained_original_ids_by_layer'] if a.u216_plan else None
overlap_u = {}; overlap_s = {}
for l in LAYERS:
    kept = ids[str(l)]; rk = sum(routes[l][e] for e in kept); ra = sum(routes[l]); sk = sum(sens[l][e] for e in kept); sa = sum(sens[l])
    tot_routes_kept += rk; tot_routes += ra; tot_sens_kept += sk; tot_sens += sa
    zero = sum(1 for e in range(N) if routes[l][e] == 0)
    detail[str(l)] = {'keep': K, 'route_mass_retention': rk / ra, 'sensitivity_retention': sk / sa, 'zero_route_experts': zero,
                      'min_routes_kept': min(routes[l][e] for e in kept), 'max_routes_pruned': max(routes[l][e] for e in pruned[str(l)]),
                      'score_threshold': score[l][ranking[l][K - 1]]}
    if u216: overlap_u[str(l)] = len(set(kept) & set(u216[str(l)]))
    overlap_s[str(l)] = len(set(kept) & set(s216['retained_original_ids_by_layer'][str(l)]))
plan = {
    'schema': 'dynamic-container-plan-v1', 'name': 'R216', 'state': 'FROZEN_PLAN',
    'point': {'bpw': 3.05, 'prune': removed / (N * len(LAYERS)), 'removed_experts': removed, 'retained_experts': K * len(LAYERS),
              'mean_keep': K, 'ranking': 'k3_sensitivity_x_log_routes_r216', 'uniform_keep': K, 'layer_floor': K, 'block': None,
              'removal_count_source': f'uniform keep {K} per MoE layer x {len(LAYERS)} layers (SGLang requires a single n_routed_experts)'},
    'base': s216['base'],
    'inputs_sha256': {'s216-plan.json': sha(a.s216_plan), 'routing-counts-r216.json': sha(a.routing),
                      'provenance_note': f'sensitivity_by_layer inherited verbatim from the sealed S216 plan (sha {s216["plan_sha256"][:16]}…, pins k3-projection-errors.json {sealed_pin[:16]}…; '
                                         'independent copy sensitivity.json cross-checked at max |diff| 0.0); routes = routing-counts-r216.json: sealed prefill dumps recomputed '
                                         'from disk + 15 further generation prompts (2048-token cap) captured 2026-09-14 from the unpruned 2.05bpw recorder on spark-2384; '
                                         f'provenance inside the routing file pins the sealed S216 counts sha {SEALED_S216_COUNTS_SHA[:16]}…'},
    'routing_capture': {k: rt[k] for k in ('total_routes', 'per_batch_routes', 'receipt') if k in rt} | {'source_model': 'turboderp/GLM-5.3-Flash-exl3 2.05bpw (unpruned) served by exl3_plain on spark-2384',
                        'note': 'R216 densification: same prefill batches as the sealed S216 counts + gen2-000..007 (15 prompts, 2048-token cap); capture scripts r216/capture_gen_r216.py + r216/combine_reduce.py'},
    'allocation_proxy': 'none (uniform); within-layer ranking = mean(K3 gate/up/down proxy error) x ln(routes+1), routes from the denser R216 capture',
    'allocator': 'uniform top-K of combined score, ties lower id',
    'selection_uses_heldout_quality': False,
    'keep_by_layer': {str(l): K for l in LAYERS},
    'retained_original_ids_by_layer': ids, 'pruned_original_ids_by_layer': pruned,
    'original_id_to_contiguous_id': {str(l): {str(e): i for i, e in enumerate(ids[str(l)])} for l in LAYERS},
    'scores_by_layer': {str(l): score[l] for l in LAYERS}, 'sensitivity_by_layer': {str(l): sens[l] for l in LAYERS}, 'routes_by_layer': {str(l): routes[l] for l in LAYERS},
    'layer_details': detail, 'overlap_with_u216_keep': overlap_u, 'overlap_with_s216_keep': overlap_s,
    'summary': {'min_keep': K, 'max_keep': K, 'route_mass_retained': tot_routes_kept / tot_routes, 'sensitivity_retained': tot_sens_kept / tot_sens,
                'mean_overlap_with_u216': (sum(overlap_u.values()) / len(overlap_u)) if overlap_u else None,
                'mean_overlap_with_s216': sum(overlap_s.values()) / len(overlap_s)},
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
