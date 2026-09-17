"""Freeze unpruned candidates from existing calibration; no fit/quality claim."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'candidate-designs'
BASE = ROOT / 'quality/unpruned-next-candidate'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parent_path = BASE / 'projection14/selection.json'
    assert sha(parent_path) == '57926af4ddf6aa5f5cf0180ec6cf57b867f4ec62457d806ecab1310cddf65c78'
    parent = json.loads(parent_path.read_text())
    tiers = {}
    sources = {}
    for k in (2, 3):
        p = BASE / f'projection/k{k}-projection-errors.json'
        assert sha(p) == parent['projection_input_sha256'][p.name]
        sources[p.name] = sha(p)
        entries = {}
        for part in json.loads(p.read_text())['parts']:
            for expert in part['reports']:
                key = (part['layer'], expert['expert'])
                assert key not in entries
                entries[key] = expert
        assert set(entries) == {(l, e) for l in range(3, 45) for e in range(288)}
        tiers[k] = entries
    scores = {}
    for layer in range(3, 45):
        for expert in range(288):
            a, b = tiers[2][layer, expert], tiers[3][layer, expert]
            assert a['routes'] == b['routes'] and a['routes'] > 0
            assert all(math.isfinite(v) for v in (*a['proxy'].values(), *b['proxy'].values()))
        scores[layer] = sum(tiers[2][layer, e]['proxy']['down_proj'] -
                            tiers[3][layer, e]['proxy']['down_proj'] for e in range(288))
    ranking = sorted(scores, key=lambda l: (-scores[l], l))
    assert set(ranking[:14]) == set(map(int, parent['layer_projection_bits']))
    assert math.isclose(sum(scores[l] for l in ranking[:14]), parent['calibration_proxy_reduction'], rel_tol=1e-12)
    increment = parent['extra_tensor_bytes'] // 14
    assert increment == 301989888
    baseline_bytes = parent['expected_tensor_bytes'] - parent['extra_tensor_bytes']
    OUT.mkdir(exist_ok=True)
    for count in (20, 22):
        result = {
            'state': 'CALIBRATION_FROZEN_NOT_BUILT_NOT_ADMITTED',
            'parent_selection_sha256': sha(parent_path), 'calibration_sha256': sources,
            'selection_uses_heldout_quality': False, 'quality_prediction': None,
            'selection_policy': parent['ranking_policy'],
            'default_bits': 2, 'rank_stacked_tp': 4,
            'experts_per_layer': 288, 'removed_experts': 0,
            'native_protected_tensor_bytes': parent['native_protected_tensor_bytes'],
            'layer_projection_bits': {str(l): {'down_proj': 3} for l in sorted(ranking[:count])},
            'extra_target_bytes_over_k2': count * increment,
            'expected_tensor_bytes': baseline_bytes + count * increment,
            'additional_layers_over_projection14': sorted(set(ranking[:count])-set(ranking[:14])),
            'calibration_proxy_reduction': sum(scores[l] for l in ranking[:count]),
            'prerequisites': ['Verify full-server NVFP4 draft admission and actual memory.',
                              'Verify source weights and protected closure during assembly.',
                              'Run unchanged frozen quality panel before serving promotion.',
                              'Measure262144context,image/video,MTP,graphs and sustained throughput.'],
            'caveat': 'Component proxy is not model fidelity. Tensor bytes are not resident runtime memory. Preserve native protected target weights.',
            'generator_sha256': sha(Path(__file__)),
        }
        path = OUT / f'projection{count}.json'
        raw = json.dumps(result, indent=2, sort_keys=True)+'\n'
        if path.exists():
            assert path.read_text() == raw, 'Refuse to replace a different frozen selection'
        else:
            path.write_text(raw)
        print(json.dumps({'candidate': count, 'extra_GiB': count*increment/2**30,
                          'additional_layers': result['additional_layers_over_projection14'],
                          'sha256': sha(path), 'state': result['state']}))


if __name__ == '__main__':
    main()
