#!/usr/bin/env python3
"""Seal completed matched-trunk quality results without allocating any GPU."""
import argparse
import hashlib
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        while data := handle.read(8 << 20):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--inspect', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--early-capture-summary', type=Path)
    parser.add_argument('--disclose-missing-raw-receipts', type=int, nargs='+', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inspect = json.loads(args.inspect.read_text())[0]
    assert inspect['State']['Status'] == 'exited'
    assert inspect['State']['ExitCode'] == 0 and not inspect['State']['OOMKilled']
    history = json.loads(args.history.read_text())
    assert history['container_id'] == inspect['Id']
    missing_receipts = set(range(45)) - set(map(int, history['layers']))
    assert set(map(int, history['layers'])) <= set(range(45))
    if args.disclose_missing_raw_receipts:
        # Missing monitoring records cannot establish per-layer timings. Final
        # normalized rows and evaluator identity are still independently checked.
        assert missing_receipts == set(args.disclose_missing_raw_receipts)
        assert 44 not in missing_receipts, 'Final layer completion must be retained'
        assert args.early_capture_summary is None
    elif missing_receipts:
        # Original Q3 monitoring began after the first rotating layer slot was
        # removed. Preserve that distinction instead of inventing its receipt.
        assert missing_receipts == {0} and args.early_capture_summary is not None
        early = json.loads(args.early_capture_summary.read_text())
        assert early['container_id'] == inspect['Id']
        assert early['image_id'] == inspect['Image']
        assert missing_receipts <= set(early['completed_layer_ids'])
        assert early['rows_per_completed_layer'] == 32
        assert math.isfinite(early['worker_seconds']['0'])
    for layer in history['layers'].values():
        assert layer['state'] == 'COMPLETE'
        assert sum(worker['completed'] + worker['skipped'] for worker in layer['workers']) == 32
    root = args.result
    identity = json.loads((root/'evaluation-identity.json').read_text())
    report = json.loads((root/'quality-report.json').read_text())
    normalized = root/'variant-normalized'
    manifest = json.loads((normalized/'manifest.json').read_text())
    assert report['state'] == 'QUALITY_MEASURED'
    assert report['tokens'] == report['positions'] == 65504
    assert report['variant_normalized_manifest_sha256'] == sha(normalized/'manifest.json')
    assert manifest['state'] == 'COMPLETE' and manifest['rows'] == 32
    assert len(manifest['records']) == 32
    assert {item['row'] for item in manifest['records']} == set(range(32))
    assert len(report['per_row']) == 32
    assert {row['row'] for row in report['per_row']} == set(range(32))
    assert sum(row['tokens'] for row in report['per_row']) == 65504
    for key, value in identity.items():
        if key != 'schema':
            assert report[key] == value, key
    for key in ['kl_bf16_to_variant', 'top1_agreement', 'bf16_perplexity', 'variant_perplexity']:
        assert math.isfinite(report[key])
    for item in manifest['records']:
        row = normalized/f"row-{item['row']:03d}.safetensors"
        assert row.stat().st_size == item['bytes'] and sha(row) == item['sha256']
        with row.open('rb') as handle:
            size = struct.unpack('<Q', handle.read(8))[0]
            header = json.loads(handle.read(size))
        assert header['hidden']['shape'] == [1, 2048, 4096]
        assert header['hidden']['dtype'] == 'BF16'
        assert header['hidden']['data_offsets'] == [0, 1 * 2048 * 4096 * 2]
        assert row.stat().st_size == 8 + size + header['hidden']['data_offsets'][1]
    evidence = {'schema': 'glm53-completed-quality-integrity-seal-v1',
        'state': 'INTEGRITY_SEALED_QUALITY_NOT_ACCEPTED',
        'sealed_at': datetime.now(timezone.utc).isoformat(),
        'container_id': inspect['Id'], 'image_id': inspect['Image'],
        'container_inspect_sha256': sha(args.inspect),
        'history_sha256': sha(args.history),
        'retained_layer_receipt_ids': sorted(map(int, history['layers'])),
        'missing_raw_layer_receipt_ids': sorted(missing_receipts),
        'raw_receipt_coverage_complete': not missing_receipts,
        'missing_receipt_policy': 'No per-layer timing or raw receipt claims for missing IDs; final result integrity checked independently.',
        'early_capture_summary_sha256': sha(args.early_capture_summary) if args.early_capture_summary else None,
        'identity_sha256': sha(root/'evaluation-identity.json'),
        'quality_report_sha256': sha(root/'quality-report.json'),
        'normalized_manifest_sha256': sha(normalized/'manifest.json'),
        'normalized_rows': 32, 'positions': 65504,
        'trunk_layers': 45, 'artifact_mode': identity['artifact_mode'],
        'experts_per_layer': identity['experts_per_layer'],
        'runtime_accepted': False, 'quality_accepted': False}
    assert not args.output.exists(), 'Refusing to overwrite an existing seal'
    args.output.write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
