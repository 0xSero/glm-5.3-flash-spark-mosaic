"""Method-local EXL3 precision selection; no tensors or shared mutations."""
from copy import copy
import re


def parse_layer_bits(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError('layer_bits must be a JSON object')
    result = {}
    for layer, bits in value.items():
        if not isinstance(layer, str) or not re.fullmatch(r'[1-9][0-9]*', layer):
            raise ValueError('layer_bits keys must be canonical decimal layer IDs')
        if not 3 <= int(layer) <= 44:
            raise ValueError('Only GLM target routed layers3..44 may be overridden')
        if type(bits) is not int or bits not in (2, 3, 4, 5, 6):
            raise ValueError('Unsupported per-layer EXL3 bit width')
        result[layer] = bits
    return result


def for_routed_prefix(config, prefix):
    if not config.layer_bits:
        return config
    match = re.fullmatch(r'(?:language_model\.)?model\.layers\.([0-9]+)\.mlp\.experts(?:\.routed_experts)?', prefix)
    if match is None or not 3 <= int(match[1]) <= 44:
        raise ValueError('Mixed precision requires an exact target routed-expert prefix')
    bits = config.layer_bits.get(str(int(match[1])), config.bits)
    if bits == config.bits:
        return config
    selected = copy(config)
    selected.bits = bits
    selected.raw_config = config.raw_config.copy()
    selected.layer_bits = {}
    return selected
