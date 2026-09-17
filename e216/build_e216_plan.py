#!/usr/bin/env python3
"""E216 plan: UNIFORM keep-216 prune of the 3.05bpw base ranked by the REAP-NATIVE score
(mean over routed tokens of renormalized router weight x expert output L2 norm).

Queue item 4 (driver run 5). The REAP-native criterion is the one implemented by the reference
repo (reap-strict-serialization): src/reap/pruning_metrics.py line 198 — reap[e] = mean over ROUTED
tokens of (ean_norm * active_router_weights), with top-k router weights renormalized to sum 1
(2026-03-11 fix); src/reap/prune.py prunes the experts with the LOWEST such saliency (topk
largest=False). The sealed 0xSero observations (sealed-observations/observations.json, schema
glm53-original-records-seal-v1, state SEALED_PARTIAL_ORIGINAL_RECORDS) contain, per layer 3..44 x
288 experts, weighted_ean_sum (= sum of router_weight x ||expert_output|| over routed tokens) and
expert_frequency (= routed-token count), so the REAP-native score = weighted_ean_sum /
expert_frequency. The fresh R216 route capture is used for DIAGNOSTICS only (route-mass retention),
NOT in the score. MTP layer 45 kept verbatim.
"""
import argparse, hashlib, json, time
from pathlib import Path
LAYERS = tuple(range(3, 45)); N = 288
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok, msg):
    if not ok: raise ValueError(msg)
ap = argparse.ArgumentParser()
ap.add_argument('--observations', required=True, help='sealed-observations/observations.json')
ap.add_argument('--r216-plan', required=True, help='leader plan: base field + overlap baseline')
ap.add_argument('--s216-plan', required=True); ap.add_argument('--u216-plan', required=True); ap.add_argument('--t216-plan', required=True)
ap.add_argument('--routing', required=True, help='routing-counts-r216.json (diagnostics only)')
ap.add_argument('--reap-src', required=True, help='reap-strict-serialization checkout root (provenance shas)')
ap.add_argument('--out', required=True)
ap.add_argument('--keep', type=int, default=216); ap.add_argument('--expert-bytes', type=int, default=9474060); ap.add_argument('--router-row-bytes', type=int, default=8196)
a = ap.parse_args(); K = a.keep
obsw = json.loads(Path(a.observations).read_text())
require(obsw.get('schema') == 'glm53-original-records-seal-v1', f"unexpected seal schema {obsw.get('schema')}")
require(obsw.get('state') == 'SEALED_PARTIAL_ORIGINAL_RECORDS', f"unexpected seal state {obsw.get('state')}")
obs = obsw['observations']
require(sorted(map(int, obs)) == list(LAYERS), 'observation layers do not cover 3..44 exactly')
reap_score = {}; wsum = {}; freq = {}; minfreq = None
for l in LAYERS:
    L = obs[str(l)]
    w, f = L['weighted_ean_sum'], L['expert_frequency']
    require(len(w) == N and len(f) == N, f'layer {l} width != 288')
    wsum[l] = [float(x) for x in w]; freq[l] = [float(x) for x in f]
    reap_score[l] = [w[e] / f[e] if f[e] > 0 else 0.0 for e in range(N)]
    m = min(f)
    minfreq = m if minfreq is None else min(minfreq, m)
require(minfreq > 0, f'an expert is unseen in the sealed data (min frequency {minfreq})')
ranking = {l: sorted(range(N), key=lambda e: (-reap_score[l][e], e)) for l in LAYERS}
ids = {str(l): sorted(ranking[l][:K]) for l in LAYERS}; pruned = {str(l): sorted(ranking[l][K:]) for l in LAYERS}
removed = len(LAYERS) * (N - K)
r216 = json.loads(Path(a.r216_plan).read_text())
body_check = json.dumps({k: v for k, v in r216.items() if k != 'plan_sha256'}, indent=1, sort_keys=True)
require(r216['name'] == 'R216' and r216['plan_sha256'] == hashlib.sha256(body_check.encode()).hexdigest(), 'R216 plan file does not match its embedded plan_sha256')
rt = json.loads(Path(a.routing).read_text())
require(rt.get('total_routes', 0) > 222285, 'diagnostic routing file does not look like the denser capture')
routes = {l: [int(x) for x in rt['counts'][str(l)]] for l in LAYERS}
s216p = json.loads(Path(a.s216_plan).read_text())
u216p = json.loads(Path(a.u216_plan).read_text()); t216p = json.loads(Path(a.t216_plan).read_text())
detail = {}; tot_routes_kept = 0; tot_routes = 0
ov = {'r216': {}, 's216': {}, 'u216': {}, 't216': {}}
others = {'r216': r216['retained_original_ids_by_layer'], 's216': s216p['retained_original_ids_by_layer'],
          'u216': u216p['retained_original_ids_by_layer'], 't216': t216p['retained_original_ids_by_layer']}
for l in LAYERS:
    kept = ids[str(l)]
    rk = sum(routes[l][e] for e in kept); ra = sum(routes[l]); tot_routes_kept += rk; tot_routes += ra
    detail[str(l)] = {'keep': K, 'score_threshold': reap_score[l][ranking[l][K - 1]],
                      'max_score_pruned': reap_score[l][ranking[l][K]],
                      'route_mass_retention_diag': rk / ra,
                      'zero_route_experts_diag': sum(1 for e in range(N) if routes[l][e] == 0)}
    for name, ksets in others.items():
        ov[name][str(l)] = len(set(kept) & set(ksets[str(l)]))
mean_ov = {k: sum(v.values()) / len(v) for k, v in ov.items()}
plan = {
    'schema': 'dynamic-container-plan-v1', 'name': 'E216', 'state': 'FROZEN_PLAN',
    'point': {'bpw': 3.05, 'prune': removed / (N * len(LAYERS)), 'removed_experts': removed, 'retained_experts': K * len(LAYERS),
              'mean_keep': K, 'ranking': 'reap_native_mean_gate_weighted_ean', 'uniform_keep': K, 'layer_floor': K, 'block': None,
              'removal_count_source': f'uniform keep {K} per MoE layer x {len(LAYERS)} layers (SGLang requires a single n_routed_experts)'},
    'base': r216['base'],
    'inputs_sha256': {'observations.json': sha(a.observations), 'routing-counts-r216.json': sha(a.routing),
                      'r216-plan.json': sha(a.r216_plan), 's216-plan.json': sha(a.s216_plan),
                      'u216-plan.json': sha(a.u216_plan), 't216-plan.json': sha(a.t216_plan),
                      'reap_prune_py': sha(f"{a.reap_src}/src/reap/prune.py"),
                      'reap_pruning_metrics_py': sha(f"{a.reap_src}/src/reap/pruning_metrics.py"),
                      'provenance_note': 'REAP-native score per expert = weighted_ean_sum / expert_frequency from the sealed 0xSero observations '
                                         '(schema glm53-original-records-seal-v1, state SEALED_PARTIAL_ORIGINAL_RECORDS: 21,248/23,088 records, '
                                         '32,602,850/37,328,459 tokens, science domain missing, min expert routes 49,419 — sealed partial is the '
                                         'of-record observation set). Matches reap-strict-serialization src/reap/pruning_metrics.py:198 '
                                         '(reap = mean over ROUTED tokens of ean_norm x active_router_weights, top-k router weights renormalized '
                                         'to sum 1) and src/reap/prune.py (prune LOWEST saliency). Routing counts used for DIAGNOSTICS only. '
                                         'MTP layer 45 verbatim; no observation exists for layer 45.'},
    'routing_capture': {'source_model': 'turboderp/GLM-5.3-Flash-exl3 2.05bpw (unpruned) served by exl3_plain on spark-2384',
                        'note': 'DIAGNOSTIC ONLY — the E216 score does not use route counts; totals embedded for comparability with S216/R216',
                        'total_routes': rt.get('total_routes')},
    'allocation_proxy': 'none (uniform); within-layer ranking = REAP-native mean(router_weight x ||expert output||) over routed tokens',
    'allocator': 'uniform top-K of REAP-native score, ties lower id',
    'selection_uses_heldout_quality': False,
    'keep_by_layer': {str(l): K for l in LAYERS},
    'retained_original_ids_by_layer': ids, 'pruned_original_ids_by_layer': pruned,
    'original_id_to_contiguous_id': {str(l): {str(e): i for i, e in enumerate(ids[str(l)])} for l in LAYERS},
    'scores_by_layer': {str(l): reap_score[l] for l in LAYERS},
    'weighted_ean_sum_by_layer': {str(l): wsum[l] for l in LAYERS},
    'expert_frequency_by_layer': {str(l): freq[l] for l in LAYERS},
    'routes_by_layer_diag': {str(l): routes[l] for l in LAYERS},
    'layer_details': detail,
    'overlap_with_r216_keep': ov['r216'], 'overlap_with_s216_keep': ov['s216'],
    'overlap_with_u216_keep': ov['u216'], 'overlap_with_t216_keep': ov['t216'],
    'summary': {'min_keep': K, 'max_keep': K, 'route_mass_retained_diag': tot_routes_kept / tot_routes,
                'mean_overlap_with_r216': mean_ov['r216'], 'mean_overlap_with_s216': mean_ov['s216'],
                'mean_overlap_with_u216': mean_ov['u216'], 'mean_overlap_with_t216': mean_ov['t216'],
                'min_expert_frequency_sealed': minfreq},
    'byte_model': {'expert_payload_bytes_each': a.expert_bytes, 'router_row_bytes_each': a.router_row_bytes,
                   'removed_tensor_bytes': removed * (a.expert_bytes + a.router_row_bytes),
                   'projected_index_total_size': 122286101684 - removed * (a.expert_bytes + a.router_row_bytes)},
    'text_config_patch': {'n_routed_experts': K, 'routed_experts_per_layer': {str(l): K for l in LAYERS}, 'retained_expert_ids_by_layer': ids},
    'mtp_layer': {'layer': 45, 'experts': 288, 'policy': 'kept verbatim (mtp.safetensors byte-identical); no observation exists for layer 45'},
    'protected_policy': 'Every non-target tensor byte-exact, including all MTP tensors and all 288 MTP experts; only target router rows and bias entries follow retained original IDs.',
    'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
body = json.dumps(plan, indent=1, sort_keys=True); plan['plan_sha256'] = hashlib.sha256(body.encode()).hexdigest()
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
if out.exists(): out.chmod(0o644)
out.write_text(json.dumps(plan, indent=1, sort_keys=True) + '\n'); out.chmod(0o444)
print(json.dumps({'plan_sha256': plan['plan_sha256'], 'frozen_utc': plan['frozen_utc'], 'summary': plan['summary'],
                  'score_threshold_first_layer': detail['3']['score_threshold']}, indent=1))
