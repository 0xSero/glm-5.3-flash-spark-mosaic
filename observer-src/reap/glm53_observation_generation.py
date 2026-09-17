"""Two-stage pinned original-Q3/Q4 generation smoke, separate from observations.

Run on both allocated workers with torchrun after staging completes. This is a
bounded text-generation execution gate, never an automatic coherence verdict.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

from reap.glm53_observation_worker import atomic_json, file_sha256, read_json

PROMPTS = (
    "Answer briefly: What is 2 + 2?",
    "Complete this sentence: The capital of France is",
)
HEAD_KEY = "lm_head.weight"
NORM_KEY = "model.language_model.norm.weight"
ORIGINAL_MODEL_IDENTITIES = (
    {"name": "0xSero/GLM-5.3-Flash-EXL3-3.0bpw",
     "revision": "2a30ad09c15f779a44fa62c216f5dbe5fb0c9223",
     "sha256": "0fd35de9b0d5fc9428a45d3b311dc757ea891e4cec7788050b75089593ad3215"},
    {"name": "0xSero/GLM-5.3-Flash-EXL3-Q4",
     "revision": "d0b9301a10da765df1d76571107577041009f28d",
     "sha256": "6a6357fd0b6268fb9f7133b5713846cfcf0f6d36b7a3f50c30b0d24ccf59c471"},
)


def validate_generation_plan(plan):
    """Reject unpinned or altered models before any source loading or GPU use."""
    from reap.glm53_observation_suite import validate_plan
    if not isinstance(plan, dict) or plan.get("model_identity") not in ORIGINAL_MODEL_IDENTITIES:
        raise ValueError("generation requires an exact pinned original Q3/Q4 model identity")
    validate_plan(plan)


def prepare_public_prompts(tokenizer, vocab_size):
    """Prepare every bounded public prompt without padding, truncation, or coercion."""
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("invalid model vocabulary size for public prompts")
    prepared = []
    for prompt in PROMPTS:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_dict=False, padding=False, truncation=False)
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 512
                or any(type(token) is not int or not 0 <= token < vocab_size for token in ids)):
            raise ValueError("public smoke prompt must be a flat list of 1..512 in-vocabulary integer IDs")
        prepared.append({"prompt": prompt, "input_ids": ids})
    return prepared


def validate_prompt_peers(statuses):
    if (len(statuses) != 2 or any(not isinstance(status, dict) or status.get("ok") is not True
                                 for status in statuses)):
        raise ValueError(f"generation peer public-prompt preparation failed: {statuses}")
    if statuses[0]["prompts"] != statuses[1]["prompts"]:
        raise ValueError("generation peers disagree on exact public-prompt token IDs")


def head_source_files(mapping, receipt):
    files = {entry["file"] for entry in receipt["selected_files"]}
    required = {key: mapping[key] for key in (HEAD_KEY, NORM_KEY)}
    if receipt.get("stage_id") != 0 or any(name not in files for name in required.values()):
        raise ValueError("verified rank-zero receipt does not cover original norm/head source files")
    return required


def _load_head(stage, model_root):
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from reap.glm53_exl3_observe import IndexedTensors
    from reap.glm53_observation_stage import _ensure_room
    receipt = read_json(model_root / "observation-stage-receipt.json")
    indexed = IndexedTensors(model_root)
    required = head_source_files(indexed.weight_map, receipt)
    norm, head = indexed.get(NORM_KEY), indexed.get(HEAD_KEY)
    if norm.shape != (4096,) or not norm.is_floating_point():
        raise ValueError("original final norm has unexpected geometry/dtype")
    if head.shape != (stage.model.config.vocab_size, 4096) or head.dtype != torch.bfloat16:
        raise ValueError("original BF16 language-model head geometry/dtype mismatch")
    _ensure_room(stage.device, head.numel() * head.element_size() + norm.numel() * norm.element_size(),
                 label="original final norm and BF16 head")
    set_module_tensor_to_device(stage.model, "norm.weight", stage.device, value=norm, dtype=norm.dtype)
    head = head.to(stage.device)
    return head, {"source_files": required, "head_dtype": str(head.dtype), "norm_dtype": str(norm.dtype)}


class GenerationSession:
    """Persistent global-indexed cache for exactly one prompt on one stage."""

    def __init__(self, stage, cache_factory=None):
        if cache_factory is None:
            from transformers.cache_utils import DynamicCache
            cache_factory = DynamicCache
        self.stage = stage
        self.cache = cache_factory(config=stage.model.config)
        self.seen = 0

    def forward(self, ids, hidden=None):
        import torch
        stage = self.stage
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or ids.dtype != torch.int64:
            raise ValueError("generation requires int64 [1,nonempty sequence] ids")
        ids = ids.to(stage.device)
        if hidden is None:
            if stage.stage_id != 0:
                raise ValueError("rank one requires incoming hidden streams")
            with torch.inference_mode():
                hidden = stage.model.embed_tokens(ids).unsqueeze(2).expand(-1, -1, 4, -1)
        if tuple(hidden.shape) != (*ids.shape, 4, 4096) or hidden.dtype != torch.bfloat16:
            raise ValueError("generation hidden streams must be source BF16 [1,sequence,4,4096]")
        hidden = hidden.to(stage.device)
        with torch.inference_mode():
            for layer_index in stage.owned_layers:
                parts = []
                for start in range(0, ids.shape[1], stage.prefill_chunk_size):
                    end = min(ids.shape[1], start + stage.prefill_chunk_size)
                    local_ids = ids[:, start:end]
                    result, topk = stage.model.layers[layer_index](
                        hidden[:, start:end], attention_mask=torch.ones_like(local_ids, dtype=torch.bool),
                        position_ids=torch.arange(self.seen + start, self.seen + end, device=ids.device).unsqueeze(0),
                        past_key_values=self.cache, use_cache=True, prev_topk_indices=None, input_ids=local_ids)
                    if topk is not None:
                        raise ValueError("unexpected shared DSA indices in all-full original config")
                    if result.shape != hidden[:, start:end].shape or result.dtype != torch.bfloat16:
                        raise ValueError("generation layer changed hidden shape/dtype")
                    parts.append(result)
                hidden = torch.cat(parts, dim=1)
        self.seen += ids.shape[1]
        return hidden


def final_logits(model, head, hidden):
    import torch
    with torch.inference_mode():
        collapsed = model.hc_head(hidden[:, -1:])
        normalized = model.norm(collapsed)
        if normalized.dtype != head.dtype:
            raise ValueError("original final norm/head dtypes disagree; refusing an unqualified cast")
        logits = torch.nn.functional.linear(normalized, head)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("generation produced nonfinite logits")
        return logits


def run(args):
    from reap.glm53_observation_pipeline import verify_stage_source
    rank = int(os.environ.get("RANK", "-1"))
    if int(os.environ.get("WORLD_SIZE", "0")) != 2 or rank not in (0, 1):
        raise ValueError("generation requires exactly two torchrun ranks")
    if args.steps not in (8, 16):
        raise ValueError("generation smoke uses exactly an 8- or 16-step ceiling")
    plan = read_json(args.plan)
    validate_generation_plan(plan)
    tokenizer, prepared_prompts = None, None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_root, local_files_only=True)
        config = read_json(args.model_root / "config.json")
        prepared_prompts = prepare_public_prompts(tokenizer, config["text_config"]["vocab_size"])
        prompt_status = {"ok": True, "prompts": prepared_prompts}
    except Exception as exc:
        prompt_status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    import torch
    import torch.distributed as dist
    # CPU/Gloo lets both peers report tokenizer failures before source rehash or
    # CUDA allocation; no rank can strand its peer in the later prompt broadcast.
    dist.init_process_group("gloo", timeout=timedelta(minutes=60))
    try:
        prompt_statuses = [None, None]
        dist.all_gather_object(prompt_statuses, prompt_status)
        validate_prompt_peers(prompt_statuses)
        source = verify_stage_source(plan, args.model_root, rank)
        from reap import glm53_exl3_observe as observer
        from reap.glm53_observation_kernels import selected_kda_identity
        from transformers.models.glm5_next.modeling_glm5_next import recurrent_kimi_delta_attention
        from reap.glm53_observation_stage import load_stage
        from reap.glm53_exl3_native import native_runtime_identity
        # Decode must use the actual recurrent FLA closure, not just prefill.
        recurrent_identity = selected_kda_identity(recurrent_kimi_delta_attention, "fla")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        device = torch.device("cuda", torch.cuda.current_device())
        data_group = dist.new_group(backend="nccl", timeout=timedelta(minutes=60))
        identities = [None, None]
        local_identity = {"model": plan["model_identity"], "script_sha256": file_sha256(Path(__file__)),
                          "stage_sha256": file_sha256(Path(__file__).with_name("glm53_observation_stage.py")),
                          "expert": native_runtime_identity(), "kda": observer._KDA_IDENTITY,
                          "recurrent_kda": recurrent_identity, "public_prompts": prepared_prompts}
        dist.all_gather_object(identities, local_identity)
        if identities[0] != identities[1]:
            raise ValueError("generation peers disagree on model/runtime/code")
        sources = [None, None]
        dist.all_gather_object(sources, source)
        if sources[0]["config_sha256"] != sources[1]["config_sha256"]:
            raise ValueError("generation peer configs differ")
        stage = load_stage(args.model_root, rank, prefill_chunk_size=512)
        head, head_info = _load_head(stage, args.model_root) if rank == 0 else (None, None)
        dist.barrier()
        outputs = []
        for prepared in prepared_prompts:
            prompt = prepared["prompt"]
            ids = torch.tensor([prepared["input_ids"]], dtype=torch.int64, device=device)
            prompt_length = ids.shape[1]
            session = GenerationSession(stage)
            generated, started, ended = [], time.monotonic(), False
            for step in range(args.steps):
                if rank == 0:
                    hidden = session.forward(ids)
                    dist.send(hidden.contiguous(), dst=1, group=data_group)
                    tail = torch.empty((1, 1, 4, 4096), dtype=torch.bfloat16, device=device)
                    dist.recv(tail, src=1, group=data_group)
                    token = int(final_logits(stage.model, head, tail)[0, 0].argmax())
                    generated.append(token)
                    eos = stage.model.config.eos_token_id
                    ended = token in (eos if isinstance(eos, (list, tuple)) else [eos])
                    decision = [token, ended]
                else:
                    hidden = torch.empty((1, ids.shape[1], 4, 4096), dtype=torch.bfloat16, device=device)
                    dist.recv(hidden, src=0, group=data_group)
                    hidden = session.forward(ids, hidden)
                    dist.send(hidden[:, -1:].contiguous(), dst=0, group=data_group)
                    decision = [None, None]
                dist.broadcast_object_list(decision, src=0)
                if decision[1]:
                    break
                ids = torch.tensor([[decision[0]]], dtype=torch.int64, device=device)
            cache_lengths = [None, None]
            dist.all_gather_object(cache_lengths, session.seen)
            if cache_lengths != [prompt_length + step, prompt_length + step]:
                raise ValueError("generation stages disagree on consumed prefix length")
            if rank == 0:
                outputs.append({"prompt": prompt, "prompt_tokens": prompt_length,
                                "generated_ids": generated, "decoded": tokenizer.decode(generated, skip_special_tokens=False),
                                "eos_seen": ended, "seconds": time.monotonic() - started,
                                "cache_seen_tokens": session.seen, "manual_coherence_review": "PENDING"})
                atomic_json(args.output.with_name(args.output.stem + "-outputs.json"), outputs)
            del session
            stage.reset_observations()  # generation stats never enter observation receipts
        if rank == 0:
            report = {"schema": "glm53-resident-generation-smoke-v1", "state": "EXECUTION_COMPLETE_REVIEW_REQUIRED",
                      "runtime": local_identity, "stage_sources": sources, "head": head_info,
                      "prompts_completed": len(outputs), "step_ceiling": args.steps,
                      "outputs_file": args.output.with_name(args.output.stem + "-outputs.json").name,
                      "coherence_pass": None, "full_suite_pass": False, "uploaded": False}
            atomic_json(args.output, report)
            print(json.dumps(report, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "model-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(8, 16), default=16)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
