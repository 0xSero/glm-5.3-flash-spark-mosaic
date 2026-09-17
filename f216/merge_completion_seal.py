#!/usr/bin/env python3
"""Merge F216 completion receipts into the partial original-records seal.

CPU only. Runs where the observer checkout's src directory is on PYTHONPATH.
The original sealer (seal_glm_observations.py, imported here — its aggregation
code is reused verbatim, never reimplemented) cannot do this merge for two
hard reasons, both enforced by its own validators:
  1. It pins ONE runtime identity across all receipts; the completion shards
     332-360 were captured by a different run (different hosts/image), so
     validate_receipt against the original runtime fails by design.
  2. It requires completed_shards == list(range(len(ids))); the completion
     run-root legitimately holds the contiguous TAIL [332..360].
This merger therefore validates each epoch's receipts against that epoch's own
runtime identity (original runtime from the partial seal for shards 0-331,
completion runtime for shards 332-360, single identity enforced within each
epoch), joins the per-shard sources, and records the split provenance
explicitly. No aggregation math changes: add_observations is the sealer's own
function applied to the new receipts over the partial aggregate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seal_glm_observations as sealer


def require(condition, message):
    if not condition:
        raise ValueError("merge-refused: " + message)


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('partial-observations', 'partial-seal', 'model-identity',
                 'token-manifest', 'completion-run-root', 'completion-driver', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--keep', type=int, nargs='+',
                        default=[128, 144, 160, 176, 192, 208, 216, 224, 240, 256, 288])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    from reap.glm53_record_observation_pipeline import (record_contract, validate_manifest,
                                                        validate_receipt, load_records)
    from reap.glm53_observation_suite import LAYERS, EXPERTS, _validated_observation
    from reap.glm53_observation_sidecar import safe_path
    from collections import Counter

    partial_obs_path, partial_seal_path = safe_path(args.partial_observations), safe_path(args.partial_seal)
    partial = load_json(partial_obs_path)
    partial_summary = load_json(partial_seal_path)
    require(partial.get('schema') == 'glm53-original-records-seal-v1', 'partial seal schema mismatch')
    require(partial.get('state') == 'SEALED_PARTIAL_ORIGINAL_RECORDS',
            f"partial seal state is {partial.get('state')!r}, expected the untouched partial seal")
    require(sealer.digest_file(partial_obs_path) ==
            partial_summary.get('artifact_sha256', {}).get('observations.json'),
            'partial observations.json bytes do not match the partial seal receipt')

    manifest_path, model_path = safe_path(args.token_manifest), safe_path(args.model_identity)
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sealer.digest_file(manifest_path)
    model = json.loads(model_path.read_text())
    contract = record_contract(model, manifest, manifest_sha)
    shards = validate_manifest(manifest)
    require(manifest_sha == partial.get('token_manifest_sha256'),
            'token manifest changed since the partial seal')
    require(contract == partial.get('contract'), 'recomputed contract differs from partial seal contract')

    committed = partial.get('committed_shards', [])
    require(committed == list(range(len(committed))) and 0 < len(committed) < len(shards),
            'partial seal committed_shards is not a nonempty proper contiguous prefix')
    require(len(partial.get('sources', [])) == len(committed), 'partial seal sources count mismatch')
    tail_expected = list(range(len(committed), len(shards)))

    completion_root = safe_path(args.completion_run_root) / contract['run_id']
    status_path = safe_path(completion_root / 'record-status.json')
    status = json.loads(status_path.read_text())
    require(status.get('state') == 'RECORD_LOCAL_COMPLETE',
            f"completion run state {status.get('state')!r} — refusing to seal an incomplete tail")
    tail_ids = status.get('completed_shards')
    require(tail_ids == tail_expected,
            f"completion completed_shards is not the full contiguous tail {tail_expected}")
    require(status.get('run_id') == contract['run_id'], 'completion status run_id mismatch')

    paths = [safe_path(completion_root / f'shard-{sid:05d}' / 'receipt.json') for sid in tail_ids]
    completion_runtime = None
    new_sources, new_domain_counts, new_domain_tokens = [], Counter(), Counter()
    homogeneous, mixed = {}, []
    total = {layer: dict(row) for layer, row in partial['observations'].items()}
    domains = {d: {layer: dict(row) for layer, row in rows.items()}
               for d, rows in partial['domain_observations'].items()}
    for sid, path in zip(tail_ids, paths):
        receipt = json.loads(path.read_text())
        if completion_runtime is None:
            completion_runtime = receipt['runtime_identity']
        validate_receipt(contract, shards[sid], receipt, completion_runtime)
        rows = load_records(manifest_path, shards[sid])
        counts = Counter(row.get('domain') or '__missing__' for row in rows)
        dt = Counter()
        for row in rows:
            dt[row.get('domain') or '__missing__'] += len(row['input_ids'])
        new_domain_counts.update(counts)
        new_domain_tokens.update(dt)
        sealer.add_observations(total, receipt['observations'])
        if len(counts) == 1:
            domain = next(iter(counts))
            sealer.add_observations(domains.setdefault(domain, {}), receipt['observations'])
            entry = homogeneous.setdefault(domain, {'shards': [], 'records': 0, 'tokens': 0})
            entry['shards'].append(sid)
            entry['records'] += receipt['sequence_count']
            entry['tokens'] += receipt['tokens']
        else:
            mixed.append({'shard_id': sid, 'domain_records': dict(counts), 'domain_tokens': dict(dt)})
        new_sources.append({'shard_id': sid, 'receipt_sha256': receipt['sha256'],
                            'receipt_file_sha256': sealer.digest_file(path),
                            'token_shard_sha256': receipt['token_shard_sha256'],
                            'runtime_epoch': 'completion'})

    records_new = sum(shards[s]['sequence_count'] for s in tail_ids)
    tokens_new = sum(shards[s]['tokens'] for s in tail_ids)
    require((status.get('records'), status.get('tokens')) == (records_new, tokens_new),
            'completion status counters do not match validated receipts')
    require(partial['records'] + records_new == manifest['sequence_count'] and
            partial['tokens'] + tokens_new == manifest['tokens'],
            'merged coverage would not equal the contract planned records/tokens')

    homogeneity = partial.get('homogeneous_domain_coverage', {})
    domain_counts = dict(partial.get('covered_domain_records', {}))
    domain_tokens = dict(partial.get('covered_domain_tokens', {}))
    for d, n in new_domain_counts.items():
        domain_counts[d] = domain_counts.get(d, 0) + n
    for d, n in new_domain_tokens.items():
        domain_tokens[d] = domain_tokens.get(d, 0) + n

    homogeneity = partial.get('homogeneous_domain_coverage', {})
    for domain, entry in homogeneous.items():
        merged = homogeneity.setdefault(domain, {'shards': [], 'records': 0, 'tokens': 0})
        merged['shards'] = sorted(merged['shards'] + entry['shards'])
        merged['records'] += entry['records']
        merged['tokens'] += entry['tokens']
    mixed_all = list(partial.get('mixed_domain_shards', [])) + mixed

    for group in [total, *domains.values()]:
        require(set(group) == {str(i) for i in LAYERS}, 'merged routed layer coverage mismatch')
        for row in group.values():
            _validated_observation(row, row['total_tokens'])

    sources = [dict(s, runtime_epoch='original') for s in partial.get('sources', [])] + new_sources
    missing_domains = sorted(set(manifest['domain_counts']) - set(domain_counts))
    common = {'schema': 'glm53-original-records-seal-v1',
              'state': 'SEALED_COMPLETE_ORIGINAL_RECORDS',
              'full_suite_pass': False, 'quality_accepted': False, 'contract': contract,
              'runtime_identity': partial.get('runtime_identity'),
              'completion_runtime_identity': completion_runtime,
              'runtime_homogeneity': ('split: shards 0-331 validated against the original observation run '
                                      'runtime identity; shards 332-360 against the F216 completion run '
                                      '(derived capture driver, identical pipeline math). Single-epoch '
                                      'identity enforced within each epoch at merge time.'),
              'token_manifest_sha256': manifest_sha,
              'model_identity_file_sha256': sealer.digest_file(model_path),
              'terminal_status_file_sha256': partial.get('terminal_status_file_sha256'),
              'completion_status_file_sha256': sealer.digest_file(status_path),
              'sealer_sha256': sealer.digest_file(Path(sealer.__file__)),
              'merge_sealer_sha256': sealer.digest_file(Path(__file__)),
              'records': partial['records'] + records_new, 'tokens': partial['tokens'] + tokens_new,
              'planned_records': manifest['sequence_count'], 'planned_tokens': manifest['tokens'],
              'committed_shards': list(range(len(shards))), 'sources': sources,
              'layers': list(LAYERS), 'experts_per_layer': EXPERTS,
              'rank_receipt_policy': 'Rank0 receipts already merge both pipeline layer partitions; count each once.',
              'covered_domain_records': domain_counts, 'covered_domain_tokens': domain_tokens,
              'planned_domain_records': manifest['domain_counts'],
              'missing_domains': missing_domains,
              'homogeneous_domain_coverage': homogeneity, 'mixed_domain_shards': mixed_all,
              'domain_policy': 'Only homogeneous shards contribute domain statistics. Mixed shards remain in pooled totals.',
              'minimum_expert_routes': min(min(row['expert_frequency']) for row in total.values()),
              'unseen_experts': {l: [e for e, c in enumerate(row['expert_frequency']) if not c]
                                 for l, row in total.items()},
              'completion_run': {
                  'run_root': str(args.completion_run_root),
                  'phase': (f"tail shards {tail_expected[0]}-{tail_expected[-1]} only "
                            f"(original run paused at shard {committed[-1]}; capture host pair changed)"),
                  'capture_driver_sha256': sealer.digest_file(safe_path(args.completion_driver)),
                  'records': records_new, 'tokens': tokens_new,
                  'runtime_identity': completion_runtime}}

    aggregates = {**common, 'observations': total, 'domain_observations': domains}
    candidates = {'schema': 'glm53-original-records-candidates-v1', 'state': 'CANDIDATES_NOT_QUALITY_ACCEPTED',
                  'source_seal_state': common['state'], 'model_identity': model,
                  'keep_counts': sorted(set(args.keep)), 'missing_domains': missing_domains,
                  'domain_policy': common['domain_policy'],
                  'metrics': {'mass_global': 'Sum router_weight * expert_output_norm over all observed routes; no frequency division.',
                              'massmax_domain': 'Maximum over domains of expert weighted_ean_sum / sum_all_experts(weighted_ean_sum).',
                              'conditional_mean_control': 'Frequency-divided historical REAP score, comparison control only.',
                              'frequency_control': 'Observed route count, comparison control only.',
                              'random_control': f'Uniform deterministic random order, seed {args.seed} + layer, control only.'},
                  'selection': sealer.candidate_maps(total, domains, sorted(set(args.keep)), args.seed)}

    out = Path(safe_path(args.output))
    out.mkdir(parents=True, exist_ok=False)
    for name, value in [('observations.json', aggregates), ('candidates.json', candidates)]:
        (out / name).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    summary = {k: v for k, v in common.items() if k not in ('sources', 'unseen_experts')}
    summary['artifact_sha256'] = {name: sealer.digest_file(out / name)
                                  for name in ('observations.json', 'candidates.json')}
    (out / 'seal.json').write_text(json.dumps(summary, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'state': common['state'], 'records': common['records'], 'tokens': common['tokens'],
                      'missing_domains': missing_domains,
                      'minimum_expert_routes': common['minimum_expert_routes'],
                      'completion_records': records_new, 'completion_tokens': tokens_new,
                      'artifact_sha256': summary['artifact_sha256']}))


if __name__ == '__main__':
    main()
