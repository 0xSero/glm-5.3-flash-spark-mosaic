"""CPU layout oracle for one fused launch with different projection precisions."""
from math import prod
import re

PROJECTIONS = ('gate_proj', 'up_proj', 'down_proj')


def resolve_layer_projection_bits(default_bits, overrides, layer):
    if type(default_bits) is not int or default_bits not in range(2, 7):
        raise ValueError('Unsupported EXL3 default precision')
    if not isinstance(overrides, dict) or type(layer) is not int or not 3 <= layer <= 44:
        raise ValueError('Only target routed layers3..44 are in this contract')
    resolved = {}
    for key, values in overrides.items():
        if (not isinstance(key, str) or not re.fullmatch(r'[1-9][0-9]*', key)
                or not 3 <= int(key) <= 44 or not isinstance(values, dict)
                or not values or set(values) - set(PROJECTIONS)):
            raise ValueError('Invalid per-layer projection map')
        if any(type(v) is not int or v not in range(2, 7) for v in values.values()):
            raise ValueError('Unsupported per-projection precision')
        triple = tuple(values.get(p, default_bits) for p in PROJECTIONS)
        if triple[0] != triple[1]:
            raise ValueError('Combined w13 storage requires equal gate/up precision')
        resolved[int(key)] = triple
    return resolved.get(layer, (default_bits,) * 3)


def packed_layout(bits, experts=288, hidden=4096, intermediate=2048, ranks=4):
    if (len(bits) != 3 or any(type(k) is not int or k not in range(2, 7) for k in bits)
            or bits[0] != bits[1]):
        raise ValueError('Initial layout requires valid equal gate/up precision')
    if (experts != 288 or hidden != 4096 or intermediate != 2048 or ranks != 4):
        raise ValueError('Oracle preserves full native expert geometry/rankstack4')
    local = intermediate // ranks
    shapes = {'w13_trellis': (experts, 2, hidden // 16, local // 16, bits[0] * 16),
              'w2_trellis': (experts, local // 16, hidden // 16, bits[2] * 16)}
    return {'shapes_per_rank': shapes,
            'total_trellis_bytes': sum(prod(s) * 2 * ranks for s in shapes.values()),
            'native_dispatch_k': bits[0] if len(set(bits)) == 1 else 0,
            'native_projection_arguments': bits}
