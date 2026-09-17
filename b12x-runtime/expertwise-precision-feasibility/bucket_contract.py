"""CPU-only oracle for preserving global routes across homogeneous K buckets.

Not a serving implementation. A GPU implementation must use fixed-shape tensor
lookups and sum FP32 bucket outputs before the single final dtype conversion.
"""


def banks(upgraded_ids, native_experts=288):
    if native_experts != 288:
        raise ValueError("This unpruned contract preserves all288 native experts")
    if (not isinstance(upgraded_ids, list)
            or any(type(i) is not int or not 0 <= i < native_experts
                   for i in upgraded_ids)
            or upgraded_ids != sorted(set(upgraded_ids))):
        raise ValueError("K3 IDs must be unique sorted native expert IDs")
    upgraded = set(upgraded_ids)
    ids = {2: [i for i in range(native_experts) if i not in upgraded],
           3: upgraded_ids.copy()}
    return {k: {"original_ids": group,
                "original_to_local": {e: i for i, e in enumerate(group)}}
            for k, group in ids.items() if group}


def expanded_routes(bank, global_ids, weights, archive_ranks=4):
    if archive_ranks != 4 or len(global_ids) != len(weights):
        raise ValueError("Expected four archive ranks and aligned route weights")
    if any(type(i) is not int or not 0 <= i < 288 for i in global_ids):
        raise ValueError("Router IDs must refer to the native288 experts")
    count = len(bank["original_ids"])
    sentinel = count * archive_ranks
    result = []
    for expert, weight in zip(global_ids, weights):
        local = bank["original_to_local"].get(expert)
        for rank in range(archive_ranks):
            result.append((sentinel if local is None else local * archive_ranks + rank,
                           0.0 if local is None else weight))
    return result
