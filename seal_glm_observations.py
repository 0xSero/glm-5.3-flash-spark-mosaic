#!/usr/bin/env python3
"""Seal paused original-record observations and propose frequency-aware keep maps.

CPU only. PYTHONPATH must point to the observation checkout's src directory.
No GPU execution, remote operations, pruning, or quality claims occur here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_observations(target, observations):
    for layer, row in observations.items():
        if layer not in target:
            target[layer] = {k: ([0] * len(v) if isinstance(v, list) else 0)
                             for k, v in row.items()
                             if k in ('expert_frequency', 'ean_sum', 'weighted_ean_sum',
                                      'weighted_ean_model_scaled_sum',
                                      'weighted_expert_frequency_sum', 'max_activations', 'total_tokens')}
        for key, current in target[layer].items():
            value = row[key]
            if key == 'max_activations':
                target[layer][key] = [max(a, b) for a, b in zip(current, value)]
            elif isinstance(current, list):
                target[layer][key] = [a + b for a, b in zip(current, value)]
            else:
                target[layer][key] += value


def candidate_maps(total, domains, keeps, seed=42):
    """Domain score is largest fraction of any domain's routed activation mass."""
    results = {metric: {} for metric in ('mass_global', 'massmax_domain',
                                         'conditional_mean_control', 'frequency_control', 'random_control')}
    for layer in sorted(total, key=int):
        row = total[layer]
        n = len(row['expert_frequency'])
        mass = row['weighted_ean_sum']
        domain_scores = []
        for values in domains.values():
            values = values[layer]['weighted_ean_sum']
            denom = sum(values)
            if denom <= 0:
                raise ValueError('domain has no positive activation mass')
            domain_scores.append([v / denom for v in values])
        if not domain_scores:
            raise ValueError('no homogeneous-domain observations available')
        scores = {'mass_global': mass,
                  'massmax_domain': [max(v[e] for v in domain_scores) for e in range(n)],
                  'conditional_mean_control': [v / max(1, c) for v, c in zip(mass, row['expert_frequency'])],
                  'frequency_control': row['expert_frequency']}
        rng = random.Random(seed + int(layer))
        scores['random_control'] = [rng.random() for _ in range(n)]
        for metric, values in scores.items():
            ranked = sorted(range(n), key=lambda e: (-values[e], e))
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('invalid candidate score')
            results[metric][layer] = {'scores': values, 'ranked_experts_high_to_low': ranked,
                'keep_maps': {str(k): {'keep': sorted(ranked[:k]), 'prune': sorted(ranked[k:]),
                    'pooled_mass_retention': sum(mass[e] for e in ranked[:k]) / sum(mass),
                    'minimum_domain_mass_retention': min(sum(d[e] for e in ranked[:k]) for d in domain_scores)}
                    for k in keeps}}
    return results


def seal(args):
    from reap.glm53_record_observation_pipeline import (record_contract, validate_manifest,
                                                       validate_receipt, load_records)
    from reap.glm53_observation_suite import LAYERS, EXPERTS, _validated_observation
    from reap.glm53_observation_sidecar import safe_path

    manifest_path, model_path, root = map(safe_path, (args.token_manifest, args.model_identity, args.run_root))
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = digest_file(manifest_path)
    model = json.loads(model_path.read_text())
    contract = record_contract(model, manifest, manifest_sha)
    shards = validate_manifest(manifest)
    if root.name != contract['run_id']:
        raise ValueError('run directory and immutable contract mismatch')
    # Never seal while the observer holds its production lock. Opening read-only
    # neither creates nor changes the lock file.
    with safe_path(root / 'pipeline.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = safe_path(root / 'record-status.json')
        status = json.loads(status_path.read_text())
        if status.get('state') not in ('PAUSED_AT_RECORD_SHARD', 'RECORD_LOCAL_COMPLETE'):
            raise ValueError('observer has no terminal shard-boundary status')
        ids = status.get('completed_shards')
        if not isinstance(ids, list) or not ids or ids != list(range(len(ids))):
            raise ValueError('committed shards must be a contiguous nonempty prefix')
        if len(ids) > len(shards) or status.get('run_id') != contract['run_id']:
            raise ValueError('invalid paused run identity or range')
        paths = sorted(root.glob('shard-*/receipt.json'))
        if [p.parent.name for p in paths] != [f'shard-{sid:05d}' for sid in ids]:
            raise ValueError('status and actual committed receipts disagree')
        total, domains, sources = {}, {}, []
        domain_counts, domain_tokens = Counter(), Counter()
        homogeneous, mixed = {}, []
        runtime = None
        records = tokens = 0
        for sid, path in zip(ids, paths):
            path = safe_path(path)
            receipt = json.loads(path.read_text())
            if runtime is None:
                runtime = receipt['runtime_identity']
            validate_receipt(contract, shards[sid], receipt, runtime)
            rows = load_records(manifest_path, shards[sid])
            counts = Counter(row.get('domain') or '__missing__' for row in rows)
            dt = Counter()
            for row in rows:
                dt[row.get('domain') or '__missing__'] += len(row['input_ids'])
            domain_counts.update(counts)
            domain_tokens.update(dt)
            add_observations(total, receipt['observations'])
            if len(counts) == 1:
                domain = next(iter(counts))
                add_observations(domains.setdefault(domain, {}), receipt['observations'])
                entry = homogeneous.setdefault(domain, {'shards': [], 'records': 0, 'tokens': 0})
                entry['shards'].append(sid)
                entry['records'] += receipt['sequence_count']
                entry['tokens'] += receipt['tokens']
            else:
                mixed.append({'shard_id': sid, 'domain_records': dict(counts), 'domain_tokens': dict(dt)})
            sources.append({'shard_id': sid, 'receipt_sha256': receipt['sha256'],
                            'receipt_file_sha256': digest_file(path),
                            'token_shard_sha256': receipt['token_shard_sha256']})
            records += receipt['sequence_count']
            tokens += receipt['tokens']
        if (status.get('records'), status.get('tokens')) != (records, tokens):
            raise ValueError('terminal status counters do not match validated receipts')
        for group in [total, *domains.values()]:
            if set(group) != {str(i) for i in LAYERS}:
                raise ValueError('aggregate routed layer coverage mismatch')
            for row in group.values():
                _validated_observation(row, row['total_tokens'])
        if digest_file(manifest_path) != manifest_sha:
            raise ValueError('token manifest changed during sealing')
        keeps = sorted(set(args.keep))
        if any(type(k) is not int or not 8 <= k <= EXPERTS for k in keeps):
            raise ValueError('keep count must preserve at least top-k and not exceed experts')
        complete = len(ids) == len(shards)
        common = {'schema': 'glm53-original-records-seal-v1',
                  'state': 'SEALED_COMPLETE_ORIGINAL_RECORDS' if complete else 'SEALED_PARTIAL_ORIGINAL_RECORDS',
                  'full_suite_pass': False, 'quality_accepted': False, 'contract': contract,
                  'runtime_identity': runtime, 'token_manifest_sha256': manifest_sha,
                  'model_identity_file_sha256': digest_file(model_path),
                  'terminal_status_file_sha256': digest_file(status_path),
                  'sealer_sha256': digest_file(__file__),
                  'records': records, 'tokens': tokens,
                  'planned_records': manifest['sequence_count'], 'planned_tokens': manifest['tokens'],
                  'committed_shards': ids, 'sources': sources,
                  'layers': list(LAYERS), 'experts_per_layer': EXPERTS,
                  'rank_receipt_policy': 'Rank0 receipts already merge both pipeline layer partitions; count each once.',
                  'covered_domain_records': dict(domain_counts), 'covered_domain_tokens': dict(domain_tokens),
                  'planned_domain_records': manifest['domain_counts'],
                  'missing_domains': sorted(set(manifest['domain_counts']) - set(domain_counts)),
                  'homogeneous_domain_coverage': homogeneous, 'mixed_domain_shards': mixed,
                  'domain_policy': 'Only homogeneous shards contribute domain statistics. Mixed shards remain in pooled totals.',
                  'minimum_expert_routes': min(min(row['expert_frequency']) for row in total.values()),
                  'unseen_experts': {l: [e for e, c in enumerate(row['expert_frequency']) if not c] for l, row in total.items()}}
        aggregates = {**common, 'observations': total, 'domain_observations': domains}
        candidates = {'schema': 'glm53-original-records-candidates-v1', 'state': 'CANDIDATES_NOT_QUALITY_ACCEPTED',
                      'source_seal_state': common['state'], 'model_identity': model,
                      'keep_counts': keeps, 'missing_domains': common['missing_domains'],
                      'domain_policy': common['domain_policy'],
                      'metrics': {'mass_global': 'Sum router_weight * expert_output_norm over all observed routes; no frequency division.',
                                  'massmax_domain': 'Maximum over domains of expert weighted_ean_sum / sum_all_experts(weighted_ean_sum).',
                                  'conditional_mean_control': 'Frequency-divided historical REAP score, comparison control only.',
                                  'frequency_control': 'Observed route count, comparison control only.',
                                  'random_control': f'Uniform deterministic random order, seed {args.seed} + layer, control only.'},
                      'selection': candidate_maps(total, domains, keeps, args.seed)}
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=False)
        for name, value in [('observations.json', aggregates), ('candidates.json', candidates)]:
            (out / name).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
        summary = {k: v for k, v in common.items() if k not in ('sources', 'unseen_experts')}
        summary['artifact_sha256'] = {name: digest_file(out / name) for name in ('observations.json', 'candidates.json')}
        (out / 'seal.json').write_text(json.dumps(summary, sort_keys=True, indent=2) + '\n')
        print(json.dumps({'state': common['state'], 'records': records, 'tokens': tokens,
                          'missing_domains': common['missing_domains'], 'minimum_expert_routes': common['minimum_expert_routes'],
                          'artifact_sha256': summary['artifact_sha256']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('token-manifest', 'model-identity', 'run-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--keep', type=int, nargs='+', default=[128, 144, 160, 176, 192, 208, 224, 240, 256, 288])
    parser.add_argument('--seed', type=int, default=42)
    seal(parser.parse_args())
