#!/usr/bin/env python3
"""U216 plan: UNIFORM keep-216 prune of the 3.05bpw base (72 removed per MoE layer, 42 layers).

Within-layer selection is the sealed massmax_domain ranking (top 216 of ranked_experts_high_to_low);
retained ids are sorted ascending and renumbered contiguously.  Uniform count is required by the
SGLang loader (single n_routed_experts).  MTP layer 45 has no sealed saliency and is kept verbatim.
"""
import argparse, hashlib, json, math, time
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--keep', type=int, default=216)
    ap.add_argument('--expert-bytes', type=int, default=9474060)
    ap.add_argument('--router-row-bytes', type=int, default=8196)
    a = ap.parse_args()
    K = a.keep; require(0 < K <= N, 'bad keep')
    sealed = a.root / 'sealed-observations'; inputs = {}
    for n, d in EXPECTED.items():
        require(sha(sealed / n) == d, f'sealed observations changed: {n}'); inputs['sealed-observations/' + n] = d
    obs = read(sealed / 'observations.json'); cand = read(sealed / 'candidates.json')
    rows = cand['selection']['massmax_domain']; require(set(rows) == set(map(str, LAYERS)), 'wrong layer coverage')
    scores = {}; rankings = {}; domain_scores = {}
    for l in LAYERS:
        k = str(l); scores[l] = rows[k]['scores']; rankings[l] = rows[k]['ranked_experts_high_to_low']; domain_scores[l] = {}
        require(len(scores[l]) == N and sorted(rankings[l]) == list(range(N)), 'invalid expert coverage')
        require(rankings[l] == sorted(range(N), key=lambda e: (-scores[l][e], e)), 'ranking differs from massmax criterion')
        for domain, group in obs['domain_observations'].items():
            mass = group[k]['weighted_ean_sum']; denom = sum(mass); require(denom > 0, 'empty domain mass')
            domain_scores[l][domain] = [v / denom for v in mass]
        rec = [max(s[e] for s in domain_scores[l].values()) for e in range(N)]
        require(all(math.isclose(x, y, rel_tol=1e-13, abs_tol=1e-15) for x, y in zip(scores[l], rec)), 'sealed score does not reproduce')
        # cross-check against the sealed keep_map for this count if present
        km = rows[k].get('keep_maps', {}).get(str(K))
        if km is not None:
            kept_from_map = sorted(i for i, v in enumerate(km) if v) if len(km) == N and set(km) <= {0, 1, True, False} else sorted(km)
            require(kept_from_map == sorted(rankings[l][:K]), f'layer {l}: sealed keep_map[{K}] != top-{K} of ranking')
    counts = {l: K for l in LAYERS}
    ids = {str(l): sorted(rankings[l][:K]) for l in LAYERS}
    pruned = {str(l): sorted(rankings[l][K:]) for l in LAYERS}
    removed = len(LAYERS) * (N - K)
    detail = {}; pooled_kept = 0.0; pooled_tot = 0.0
    for l in LAYERS:
        kept = ids[str(l)]; pool = obs['observations'][str(l)]['weighted_ean_sum']; tot = sum(scores[l])
        pooled_kept += sum(pool[e] for e in kept); pooled_tot += sum(pool)
        detail[str(l)] = {'keep': K, 'normalized_proxy_removed': sum(scores[l][e] / tot for e in pruned[str(l)]),
                          'pooled_mass_retention': sum(pool[e] for e in kept) / sum(pool),
                          'minimum_domain_mass_retention': min(sum(s[e] for e in kept) for s in domain_scores[l].values())}
    plan = {
        'schema': 'dynamic-container-plan-v1', 'name': 'U216', 'state': 'FROZEN_PLAN',
        'point': {'bpw': 3.05, 'prune': removed / (N * len(LAYERS)), 'removed_experts': removed, 'retained_experts': K * len(LAYERS),
                  'mean_keep': K, 'ranking': 'massmax_domain', 'uniform_keep': K, 'layer_floor': K, 'block': None,
                  'removal_count_source': f'uniform keep {K} per MoE layer x {len(LAYERS)} layers (SGLang requires a single n_routed_experts)'},
        'base': {'repo': 'turboderp/GLM-5.3-Flash-exl3', 'branch': '3.05bpw', 'revision': '332ab457b709b7ba30dd9a448be5de03b80a7ac9', 'total_bytes': 125263712676},
        'inputs_sha256': inputs, 'observation_model_identity': cand['model_identity'],
        'allocation_proxy': 'none (uniform); within-layer massmax_domain ranking only',
        'allocator': 'uniform top-K of sealed massmax_domain ranked_experts_high_to_low',
        'selection_uses_heldout_quality': False,
        'keep_by_layer': {str(l): K for l in LAYERS},
        'retained_original_ids_by_layer': ids, 'pruned_original_ids_by_layer': pruned,
        'original_id_to_contiguous_id': {str(l): {str(e): i for i, e in enumerate(ids[str(l)])} for l in LAYERS},
        'layer_details': detail,
        'summary': {'min_keep': K, 'max_keep': K,
                    'normalized_proxy_removed_total': sum(d['normalized_proxy_removed'] for d in detail.values()),
                    'pooled_mass_retained': pooled_kept / pooled_tot},
        'byte_model': {'expert_payload_bytes_each': a.expert_bytes, 'router_row_bytes_each': a.router_row_bytes,
                       'removed_tensor_bytes': removed * (a.expert_bytes + a.router_row_bytes),
                       'projected_index_total_size': 122286101684 - removed * (a.expert_bytes + a.router_row_bytes),
                       'projected_total_files_bytes_approx': 125263712676 - removed * (a.expert_bytes + a.router_row_bytes)},
        'text_config_patch': {'n_routed_experts': K, 'routed_experts_per_layer': {str(l): K for l in LAYERS}, 'retained_expert_ids_by_layer': ids},
        'mtp_layer': {'layer': 45, 'experts': 288, 'policy': 'kept verbatim (mtp.safetensors byte-identical); no sealed saliency exists for layer 45'},
        'protected_policy': 'Every non-target tensor byte-exact, including all MTP tensors and all 288 MTP experts (mtp.safetensors copied verbatim); only target router rows and bias entries follow retained original IDs.',
        'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    body = json.dumps(plan, indent=1, sort_keys=True)
    plan['plan_sha256'] = hashlib.sha256(body.encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists(): a.out.chmod(0o644)
    a.out.write_text(json.dumps(plan, indent=1, sort_keys=True) + '\n'); a.out.chmod(0o444)
    print(json.dumps({'plan_sha256': plan['plan_sha256'], 'point': plan['point'], 'summary': plan['summary'], 'byte_model': plan['byte_model']}, indent=1))
if __name__ == '__main__': main()
