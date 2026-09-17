#!/usr/bin/env python3
"""CPU-only, lossless physical expert selection from the pinned GLM Flash Q3 or K2.

No tensor arithmetic or requantization. Source files are read only; shard outputs
and receipts are atomic and resumable. A structural seal is not quality or serving
acceptance. The complete native MTP layer remains independent of trunk pruning.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import time

REVISION = '2a30ad09c15f779a44fa62c216f5dbe5fb0c9223'
REPO = '0xSero/GLM-5.3-Flash-EXL3-3.0bpw'
SOURCE_PROFILES = {
    (REPO, REVISION): {
        'bits': 3, 'files': 130, 'file_bytes': 149402871912,
        'routed_bytes': 115490479104,
        'manifest_sha256': '05e0ff9cc6a3f87fbd8e27b46bb679e114579dfea3bc4afcc2d724b58be3d1ee',
        'inventory_sha256': 'a5fb57d1c198d1dcbd58748ed22c1f1cb86d2cee8d04bb5623a7ef1b099350e4'},
    ('0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw', '35b4b580debe2a1510a2eb042d9d6d4cd7fb3a6b'): {
        'bits': 2, 'files': 133, 'file_bytes': 111352026456,
        'routed_bytes': 77439753216,
        'manifest_sha256': '501641b947fa56afc9ed098bdc034e59c2bd229d4fa1a1d7b80e3727f10c8ba3',
        'inventory_sha256': '36b79ee52ace17008eb917588666e3ae7f6ce5f419470df019da8a27576d594b'},
}
PROTECTED_BYTES = 33835039608
LAYERS = tuple(range(3, 45))
EXPERTS = 288
EXPERT = re.compile(r'^(model\.language_model\.layers\.)(\d+)(\.mlp\.experts\.)(\d+)(\..+)$')
ROUTER = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.gate\.(weight|e_score_correction_bias)$')
CHUNK = 8 << 20


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def safe_relative(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError(f'unsafe relative path: {name}')
    return p


def read_header(path):
    size = path.stat().st_size
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if not 2 <= n <= min(size - 8, 64 << 20):
            raise ValueError('invalid safetensors header length')
        header = json.loads(f.read(n))
    cursor = 0
    for _, value in sorted(((k, v) for k, v in header.items() if k != '__metadata__'),
                           key=lambda item: item[1]['data_offsets']):
        lo, hi = value['data_offsets']
        if lo != cursor or hi < lo or hi > size - n - 8:
            raise ValueError('invalid safetensors offsets')
        cursor = hi
    if cursor != size - n - 8:
        raise ValueError('unreferenced safetensors payload')
    return header, n + 8


def selection_maps(candidates, metric, keep, layers=LAYERS, experts=EXPERTS):
    if candidates.get('schema') != 'glm53-original-records-candidates-v1':
        raise ValueError('unexpected candidates schema')
    if candidates.get('state') != 'CANDIDATES_NOT_QUALITY_ACCEPTED':
        raise ValueError('unexpected candidates state')
    if metric not in ('mass_global', 'massmax_domain') or not 8 <= keep <= experts:
        raise ValueError('invalid selection request')
    maps = {}
    for layer in layers:
        entry = candidates['selection'][metric][str(layer)]['keep_maps'][str(keep)]
        chosen = entry['keep']
        if (len(chosen) != keep or chosen != sorted(set(chosen)) or
                any(type(x) is not int or not 0 <= x < experts for x in chosen)):
            raise ValueError(f'invalid keep map at layer {layer}')
        if sorted(chosen + entry['prune']) != list(range(experts)):
            raise ValueError('keep/prune partition is not complete')
        maps[layer] = {old: new for new, old in enumerate(chosen)}
    return maps


def plan_tensor(name, value, maps):
    """Return destination name/header, source-relative byte spans, and class."""
    lo, hi = value['data_offsets']
    out = {'dtype': value['dtype'], 'shape': list(value['shape'])}
    m = EXPERT.match(name)
    if m and int(m[2]) in maps:
        layer, expert = int(m[2]), int(m[4])
        if expert not in maps[layer]:
            return None
        name = f'{m[1]}{layer}{m[3]}{maps[layer][expert]}{m[5]}'
        return name, out, [(lo, hi)], 'routed'
    m = ROUTER.match(name)
    if m and int(m[1]) in maps:
        chosen = sorted(maps[int(m[1])])
        if not out['shape'] or out['shape'][0] != EXPERTS:
            raise ValueError(f'router shape mismatch: {name}')
        if (hi - lo) % EXPERTS:
            raise ValueError('router rows are not byte aligned')
        row_bytes = (hi - lo) // EXPERTS
        out['shape'][0] = len(chosen)
        spans = [(lo + e * row_bytes, lo + (e + 1) * row_bytes) for e in chosen]
        return name, out, spans, 'router_sliced'
    return name, out, [(lo, hi)], 'native_unchanged'


def span_digest(f, base, spans, destination=None):
    h = hashlib.sha256()
    for lo, hi in spans:
        f.seek(base + lo)
        left = hi - lo
        while left:
            b = f.read(min(CHUNK, left))
            if not b:
                raise ValueError('truncated tensor')
            h.update(b)
            if destination is not None:
                destination.write(b)
            left -= len(b)
    return h.hexdigest()


def rewrite_shard(source, destination, maps):
    header, base = read_header(source)
    out_header = {}
    if '__metadata__' in header:
        out_header['__metadata__'] = header['__metadata__']
    entries, counts, cursor = [], Counter(), 0
    for name, value in header.items():
        if name == '__metadata__':
            continue
        planned = plan_tensor(name, value, maps)
        if planned is None:
            continue
        new_name, desc, spans, kind = planned
        if new_name in out_header:
            raise ValueError('duplicate remapped tensor')
        length = sum(hi - lo for lo, hi in spans)
        desc['data_offsets'] = [cursor, cursor + length]
        cursor += length
        out_header[new_name] = desc
        entries.append((name, new_name, spans, kind))
        counts[kind] += 1
    if not entries:
        return {'omitted_empty_shard': True, 'tensor_count': 0, 'tensor_bytes': 0,
                'counts': {}, 'protected_tensors': []}, {}
    raw = json.dumps(out_header, separators=(',', ':')).encode()
    raw += b' ' * ((-len(raw)) % 8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.tmp')
    protected = []
    with source.open('rb') as src, temp.open('wb') as dst:
        dst.write(struct.pack('<Q', len(raw)))
        dst.write(raw)
        for old, new, spans, kind in entries:
            h = span_digest(src, base, spans, dst)
            if kind != 'routed':
                protected.append({'source_name': old, 'output_name': new, 'kind': kind,
                                  'sha256': h, 'dtype': out_header[new]['dtype'],
                                  'shape': out_header[new]['shape']})
        dst.flush()
        os.fsync(dst.fileno())
    # Independent reread proves the unchanged native and sliced router payloads.
    checked_header, checked_base = read_header(temp)
    with temp.open('rb') as f:
        for item in protected:
            value = checked_header[item['output_name']]
            if span_digest(f, checked_base, [value['data_offsets']]) != item['sha256']:
                raise ValueError('protected tensor reread failed')
    receipt = {'omitted_empty_shard': False, 'bytes': temp.stat().st_size,
               'sha256': digest(temp), 'tensor_count': len(entries), 'tensor_bytes': cursor,
               'counts': dict(counts), 'protected_tensors': protected}
    temp.replace(destination)
    return receipt, {new: destination for _, new, _, _ in entries}


def source_profile(inventory, inventory_path=None):
    profile = SOURCE_PROFILES.get((inventory.get('repo'), inventory.get('revision')))
    if (profile is None or inventory.get('manifest_sha256', inventory.get('metadata_sha256', {}).get('EXL3_MANIFEST.json')) != profile['manifest_sha256'] or
            inventory.get('weight_file_count') != profile['files'] or
            inventory.get('weight_file_bytes') != profile['file_bytes']):
        raise ValueError('source inventory pin mismatch')
    if inventory_path is not None and digest(inventory_path) != profile['inventory_sha256']:
        raise ValueError('source inventory content mismatch')
    return profile


def candidate_quantization(config, bits, keep):
    quant = dict(config.get('quantization_config', {}))
    quant.update(quant_method='exl3', bits=bits, rank_stacked_tp=4,
                 archive_tensor_parallel_size=4, runtime_tensor_parallel_size=1,
                 requires_custom_loader=True, native_mtp_n_routed_experts=EXPERTS,
                 tensor_parallel_size=1, codebook='mcg',
                 quantized_scope=f'model.language_model.layers.3..44.mlp.experts.0..{keep-1}.{{gate_proj,up_proj,down_proj}}.weight')
    if bits == 2:
        # Source K2 metadata contains an obsolete224-tail count. Remapped candidate is uniform.
        quant.update(k2_experts_per_layer=keep, k3_experts_per_layer=0,
                     k4_retained_experts_per_layer=0, tail_experts_per_layer=keep,
                     effective_routed_tier_bpw=2.0, target_routed_tier_bpw=2.0,
                     selection='REAP kept routed experts retain original K2 bytes')
    return quant


def validate_source_metadata(source, inventory):
    profile = source_profile(inventory)
    manifest_path = source / 'EXL3_MANIFEST.json'
    if digest(manifest_path) != profile['manifest_sha256']:
        raise ValueError('original source manifest mismatch')
    manifest = json.loads(manifest_path.read_text())
    expected = {x['path']: (x['bytes'], x['sha256']) for x in inventory['files']}
    declared = {x['path']: (x['bytes'], x['sha256']) for x in manifest['files']}
    if (expected != declared or len(expected) != profile['files'] or
            sum(x[0] for x in expected.values()) != profile['file_bytes'] or
            manifest.get('state') != 'STRUCTURAL_PASS' or manifest.get('target_bpw') != profile['bits']):
        raise ValueError('source manifest/file inventory mismatch')
    for name in ('config.json', 'quantization_config.json', 'model.safetensors.index.json'):
        if digest(source / name) != inventory['metadata_sha256'][name]:
            raise ValueError(f'original source metadata mismatch: {name}')
    config = json.loads((source / 'config.json').read_text())
    if config['text_config']['n_routed_experts'] != EXPERTS:
        raise ValueError('source is already pruned')
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    if len(index['weight_map']) != 583090 or set(index['weight_map'].values()) != set(expected):
        raise ValueError('source index closure mismatch')
    return config, index, expected


def build(args):
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('source and output must be disjoint')
    candidates = json.loads(args.candidates.read_text())
    maps = selection_maps(candidates, args.metric, args.keep)
    inventory = json.loads(args.inventory.read_text())
    profile = source_profile(inventory, args.inventory)
    config, source_index, expected = validate_source_metadata(source, inventory)
    contract = {'schema': 'glm53-physical-reap-contract-v1', 'source_repo': inventory['repo'],
                'source_revision': inventory['revision'], 'source_manifest_sha256': profile['manifest_sha256'],
                'source_config_sha256': digest(source / 'config.json'),
                'source_index_sha256': digest(source / 'model.safetensors.index.json'),
                'candidates_sha256': digest(args.candidates), 'metric': args.metric,
                'keep': args.keep, 'native_mtp_n_routed_experts': EXPERTS,
                'expert_maps': {str(k): {str(a): b for a, b in v.items()} for k, v in maps.items()},
                'builder_sha256': digest(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = output / 'BUILD_CONTRACT.json'
        if prior.exists() and json.loads(prior.read_text()) != contract:
            raise ValueError('output belongs to a different build contract')
        if not prior.exists() and any(x.name != '.build.lock' for x in output.iterdir()):
            raise ValueError('unrecognized nonempty output')
        atomic_json(prior, contract)
        receipts, weight_map, protected, source_seen = [], {}, [], set()
        expert_fields, all_counts = Counter(), Counter()
        projected = PROTECTED_BYTES + profile['routed_bytes'] * args.keep // EXPERTS
        existing_bytes = sum(x.stat().st_size for x in output.rglob('*.safetensors'))
        if shutil.disk_usage(output).free < max(0, projected - existing_bytes) + args.free_floor_gib * 2**30:
            raise ValueError('insufficient free disk for projected candidate and safety floor')
        for i, (relative, (size, sha)) in enumerate(sorted(expected.items())):
            relative_path = safe_relative(relative)
            src, dst = source / relative_path, output / relative_path
            if src.stat().st_size != size or digest(src) != sha:
                raise ValueError(f'source hash mismatch: {relative}')
            header, _ = read_header(src)
            source_names = set(header) - {'__metadata__'}
            if source_seen & source_names:
                raise ValueError('duplicate source tensor')
            source_seen.update(source_names)
            for name in source_names:
                if source_index['weight_map'].get(name) != relative:
                    raise ValueError('source header/index mismatch')
                m = EXPERT.match(name)
                if m and int(m[2]) in maps and int(m[4]) in maps[int(m[2])]:
                    expert_fields[(int(m[2]), int(m[4]))] += 1
            receipt_path = output / '.receipts' / (relative + '.json')
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            if receipt is not None:
                if receipt['source_sha256'] != sha:
                    raise ValueError('resume source mismatch')
                if not receipt['omitted_empty_shard']:
                    if not dst.is_file() or dst.stat().st_size != receipt['bytes'] or digest(dst) != receipt['sha256']:
                        raise ValueError('resume output corrupted; preserved for inspection')
            else:
                receipt, _ = rewrite_shard(src, dst, maps)
                receipt.update(path=relative, source_sha256=sha)
                atomic_json(receipt_path, receipt)
            if not receipt['omitted_empty_shard']:
                emitted, _ = read_header(dst)
                for name in set(emitted) - {'__metadata__'}:
                    if name in weight_map:
                        raise ValueError('duplicate output tensor')
                    weight_map[name] = relative
            protected.extend(receipt['protected_tensors'])
            all_counts.update(receipt['counts'])
            receipts.append({k: v for k, v in receipt.items() if k != 'protected_tensors'})
            atomic_json(output / 'BUILD_STATUS.json', {'state': 'BUILDING', 'completed_files': i + 1,
                        'total_files': len(expected), 'updated_unix': time.time()})
            print(json.dumps({'completed_files': i + 1, 'file': relative}), flush=True)
        if source_seen != set(source_index['weight_map']):
            raise ValueError('source tensor closure incomplete')
        if len(expert_fields) != 42 * args.keep or set(expert_fields.values()) != {48}:
            raise ValueError('incomplete expert/rank/projection encoding fields')
        if all_counts['native_unchanged'] != 2398 or all_counts['router_sliced'] != 84:
            raise ValueError('protected tensor closure mismatch')
        if sum('.layers.45.' in x['output_name'] for x in protected) != 889:
            raise ValueError('native MTP closure mismatch')
        shutil.copyfile(source / 'config.json', output / 'source-config.json')
        quant = candidate_quantization(config, profile['bits'], args.keep)
        config['quantization_config'] = quant
        config['text_config']['n_routed_experts'] = args.keep
        config['native_mtp_n_routed_experts'] = EXPERTS
        config['reap'] = {'metric': args.metric, 'keep': args.keep, 'source_experts': EXPERTS,
                          'expert_map_file': 'BUILD_CONTRACT.json', 'native_mtp_unchanged': True}
        for filename in ('tokenizer.json', 'tokenizer_config.json', 'generation_config.json',
                         'processor_config.json', 'preprocessor_config.json', 'chat_template.jinja',
                         'special_tokens_map.json', 'LICENSE', 'LICENSE.txt'):
            if (source / filename).is_file():
                shutil.copyfile(source / filename, output / filename)
        atomic_json(output / 'config.json', config)
        atomic_json(output / 'quantization_config.json', quant)
        tensor_bytes = sum(x['tensor_bytes'] for x in receipts)
        atomic_json(output / 'model.safetensors.index.json',
                    {'metadata': {'total_size': tensor_bytes}, 'weight_map': weight_map})
        atomic_json(output / 'PROTECTED_TENSORS.json', {'state': 'BYTEWISE_VERIFIED',
                    'native_unchanged_count': all_counts['native_unchanged'],
                    'router_sliced_count': all_counts['router_sliced'], 'tensors': protected})
        seal = {'schema': 'glm53-physical-reap-exl3-v1', 'state': 'STRUCTURAL_PASS',
                'source_repo': inventory['repo'], 'source_revision': inventory['revision'], 'keep': args.keep,
                'metric': args.metric, 'target_bpw': float(profile['bits']), 'tensor_bytes': tensor_bytes,
                'tensor_count': len(weight_map), 'tensor_counts': dict(all_counts),
                'files': receipts, 'runtime_claim': False, 'quality_claim': False,
                'native_mtp_n_routed_experts': EXPERTS,
                'metadata_sha256': {n: digest(output / n) for n in
                    ('BUILD_CONTRACT.json', 'PROTECTED_TENSORS.json', 'config.json',
                     'quantization_config.json', 'source-config.json', 'model.safetensors.index.json')}}
        atomic_json(output / 'EXL3_MANIFEST.json', seal)
        atomic_json(output / 'BUILD_STATUS.json', {'state': 'COMPLETE',
                    'manifest_sha256': digest(output / 'EXL3_MANIFEST.json'),
                    'completed_files': len(expected), 'runtime_claim': False, 'quality_claim': False})
        return seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--metric', choices=['mass_global', 'massmax_domain'], required=True)
    parser.add_argument('--keep', type=int, required=True)
    parser.add_argument('--free-floor-gib', type=int, default=20)
    args = parser.parse_args()
    if args.free_floor_gib < 0:
        parser.error('free floor must be nonnegative')
    result = build(args)
    print(json.dumps({k: result[k] for k in ('state', 'keep', 'metric', 'tensor_bytes')}))


if __name__ == '__main__':
    main()
