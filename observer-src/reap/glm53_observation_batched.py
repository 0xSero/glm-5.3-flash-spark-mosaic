"""Optional attention-chunked, expert-batched decoder; not the default runner.

Attention, MHC, norms, router and shared MLP retain their original chunk shapes.
Only routed expert execution is concatenated, preserving the exact router inputs.
Native GEMM shape changes require numerical qualification, not bitwise claims.
"""
from __future__ import annotations

import torch


def validate_config(config):
    if (config.num_hidden_layers != 45 or config.hidden_size != 4096
            or config.hc_mult != 4 or config.num_experts_per_tok != 8
            or config.n_routed_experts != 288 or list(config.indexer_types) != ["full"] * 45):
        raise ValueError("batched observer requires original Flash geometry and all-full indexers")


@torch.inference_mode()
def forward_layer_batched_experts(layer, hidden, ids, previous, config, chunk_size=512, *, cache_factory=None):
    validate_config(config)
    if (type(chunk_size) is not int or chunk_size < 64 or chunk_size % 64
            or ids.ndim != 2 or ids.shape[1] < 64 or ids.shape[1] % 64
            or tuple(hidden.shape) != (*ids.shape, 4, 4096) or hidden.dtype != torch.bfloat16):
        raise ValueError("invalid batched observer hidden/token/chunk geometry")
    if previous is not None or layer.training:
        raise ValueError("batched observer requires eval mode and no shared index state")
    if layer.block_type not in ("linear_attention", "deepseek_sparse_attention"):
        raise ValueError("unsupported attention block")
    index = layer.self_attn.layer_idx
    if type(index) is not int or not 0 <= index < 45 or config.layer_types[index] != layer.block_type:
        raise ValueError("layer index/type differs from configuration")
    sparse = config.mlp_layer_types[index] == "sparse"
    if config.mlp_layer_types[index] not in ("dense", "sparse"):
        raise ValueError("unsupported MLP block")
    if cache_factory is None:
        from transformers.cache_utils import DynamicCache
        cache_factory = DynamicCache
    cache = cache_factory(config=config)
    dense_outputs, inputs, indices, weights, pending = [], [], [], [], []
    dtype = hidden.dtype
    for start in range(0, ids.shape[1], chunk_size):
        chunk_ids = ids[:, start:start + chunk_size]
        residual = hidden[:, start:start + chunk_size]
        if not sparse:
            result, topk = layer(residual, attention_mask=torch.ones_like(chunk_ids, dtype=torch.bool),
                                 position_ids=torch.arange(start, start + chunk_ids.shape[1], device=ids.device).unsqueeze(0),
                                 past_key_values=cache, use_cache=True, prev_topk_indices=None, input_ids=chunk_ids)
            if topk is not None:
                raise ValueError("unexpected cross-layer index state")
            dense_outputs.append(result)
            continue
        post, comb, value = layer.attn_hc(residual)
        value = layer.input_layernorm(value)
        if layer.block_type == "linear_attention":
            value = layer.self_attn(hidden_states=value, cache_params=cache,
                                   attention_mask=torch.ones_like(chunk_ids, dtype=torch.bool), input_ids=chunk_ids)
        else:
            value, _, topk = layer.self_attn(
                hidden_states=value, attention_mask=torch.ones_like(chunk_ids, dtype=torch.bool),
                position_ids=torch.arange(start, start + chunk_ids.shape[1], device=ids.device).unsqueeze(0),
                past_key_values=cache, use_cache=True, position_embeddings=None,
                prev_topk_indices=None, input_ids=chunk_ids)
            if topk is not None:
                raise ValueError("unexpected cross-layer index state")
        value = post.to(dtype).unsqueeze(-1) * value.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual)
        residual = value
        post, comb, value = layer.ffn_hc(value)
        value = layer.post_attention_layernorm(value)
        _, route_weights, route_indices = layer.mlp.gate(value)
        flat = value.reshape(-1, 4096)
        if tuple(route_indices.shape) != (len(flat), 8) or route_weights.shape != route_indices.shape:
            raise ValueError("unexpected routed geometry")
        inputs.append(flat)
        indices.append(route_indices)
        weights.append(route_weights)
        # Shared experts remain token-local at the original chunk GEMM shape.
        shared = layer.mlp.shared_experts(value)
        pending.append((residual, post, comb, shared))
    del cache  # Expanded DSA KV is no longer needed during full expert work.
    if not sparse:
        return torch.cat(dense_outputs, dim=1), None
    full_input, full_indices, full_weights = torch.cat(inputs), torch.cat(indices), torch.cat(weights)
    inputs.clear()
    indices.clear()
    weights.clear()
    expert_output = layer.mlp.experts(full_input, full_indices, full_weights)
    if expert_output.shape != full_input.shape or expert_output.dtype != dtype:
        raise ValueError("expert output violated native observer shape/dtype contract")
    outputs, offset = [], 0
    for residual, post, comb, shared in pending:
        rows = shared.shape[0] * shared.shape[1]
        value = expert_output[offset:offset + rows].view_as(shared) + shared
        outputs.append(post.to(dtype).unsqueeze(-1) * value.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual))
        offset += rows
    return torch.cat(outputs, dim=1), None
