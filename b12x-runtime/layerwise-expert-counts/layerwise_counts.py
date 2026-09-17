"""GLM TP1 layerwise expert geometry for physically remapped checkpoints."""
from copy import copy
import re


def plan(config):
    counts = getattr(config, "routed_experts_per_layer", {})
    kept = getattr(config, "retained_expert_ids_by_layer", {})
    if not isinstance(counts, dict) or not isinstance(kept, dict):
        raise ValueError("Layerwise metadata must be JSON objects")
    if counts.keys() != kept.keys():
        raise ValueError("Counts and retained expert ID maps must have identical keys")
    if not counts:
        return {}
    if config.n_routed_experts != 288 or config.num_hidden_layers != 45:
        raise ValueError("Expected original GLM geometry: 45 layers, 288 experts")
    if getattr(config, "n_group", 1) != 1 or getattr(config, "topk_group", 1) != 1:
        raise ValueError("Grouped-router pruning needs a separate mapping contract")
    top_k = config.num_experts_per_token
    result = {}
    for key, count in counts.items():
        if not isinstance(key, str) or not re.fullmatch(r"[1-9][0-9]*", key):
            raise ValueError("Layer IDs must be canonical decimal strings")
        layer = int(key)
        if not 3 <= layer <= 44 or config.mlp_layer_types[layer] != "sparse":
            raise ValueError("Only routed target layers3..44 may be pruned")
        if type(count) is not int or not top_k <= count <= 288:
            raise ValueError("Expert count must preserve top-k and native upper bound")
        ids = kept[key]
        if (not isinstance(ids, list) or len(ids) != count
                or any(type(x) is not int or not 0 <= x < 288 for x in ids)
                or ids != sorted(set(ids))):
            raise ValueError("Retained IDs must be unique, sorted native IDs")
        result[layer] = count
    return result


def validate_parallel(parallel):
    for name in ("tensor_parallel_size", "pipeline_parallel_size",
                 "data_parallel_size", "decode_context_parallel_size",
                 "prefill_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            raise ValueError("Layerwise experts initially support TP1/PP1/DP1/CP1 only")
    for name in ("enable_expert_parallel", "enable_eplb", "use_sequence_parallel_moe"):
        if getattr(parallel, name, False):
            raise ValueError("Layerwise experts require EP/EPLB/sequence parallel off")
    if getattr(parallel.eplb_config, "num_redundant_experts", 0):
        raise ValueError("Redundant experts are unsupported")


def layer_config(config, parallel, layer_idx, is_mtp_layer=False):
    counts = plan(config)
    if is_mtp_layer:
        if layer_idx != 45 or config.n_routed_experts != 288:
            raise ValueError("Native MTP must retain layer45 and all288 experts")
        return config
    if not counts:
        return config
    validate_parallel(parallel)
    if layer_idx not in counts:
        return config
    selected = copy(config)
    if "text_config" in vars(config):
        selected.text_config = copy(config.text_config)
    selected.n_routed_experts = counts[layer_idx]
    return selected


def validate_loaded_tensor(config, name, shape, counts=None):
    counts = plan(config) if counts is None else counts
    if not counts:
        return
    match = re.search(r"(?:^|\.)layers\.([0-9]+)\.mlp\.(.*)$", name)
    if match is None:
        return
    layer, tail = int(match[1]), match[2]
    if layer >= 45:
        return  # The target loader independently skips native MTP tensors.
    count = counts.get(layer, 288)
    if tail in ("gate.weight", "gate.e_score_correction_bias"):
        expected = (count, config.hidden_size) if tail == "gate.weight" else (count,)
        if tuple(shape) != expected:
            raise ValueError(f"Router shape mismatch at layer{layer}: {shape} != {expected}")
    elif tail.startswith("experts."):
        expert = re.match(r"experts\.([0-9]+)\.", tail)
        if expert is None or not 3 <= layer <= 44 or int(expert[1]) >= count:
            raise ValueError("Expected physically remapped contiguous target expert IDs")
