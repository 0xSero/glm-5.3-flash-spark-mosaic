"""Resumable layerwise REAP observation for GLM-5.3 selective EXL3.

This runner keeps the published BF16 trunk resident on one large ROCm GPU,
decodes one packed routed layer at a time, and replays fixed token sequences
through that layer.  Checkpoints are written after every decoder layer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterator

import httpx
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from reap.glm53_observation_kernels import prepare_kda_import, selected_kda_identity

_KDA_REQUEST = prepare_kda_import()

from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextModel,
    chunk_kimi_delta_attention,
)

_KDA_IDENTITY = selected_kda_identity(chunk_kimi_delta_attention, _KDA_REQUEST)

from reap.glm53_exl3_rocm import EXL3TensorIndex, LayerObservation, RocmEXL3Experts


MODEL_PREFIX = "model.language_model."
ROUTED_LAYERS = range(3, 45)
DEFAULT_LANGUAGES = (
    "ar",
    "zh",
    "de",
    "es",
    "fr",
    "hi",
    "ja",
    "ko",
    "pl",
    "pt",
    "ru",
    "tr",
)


def _expert_backend():
    backend = os.environ.get("GLM53_OBSERVER_EXPERTS", "portable")
    if backend not in ("portable", "native"):
        raise ValueError("GLM53_OBSERVER_EXPERTS must be portable or native")
    if backend == "native":
        from reap.glm53_exl3_native import NativeEXL3Experts
        return NativeEXL3Experts
    return RocmEXL3Experts


def _expert_identity():
    backend = _expert_backend()
    if backend is RocmEXL3Experts:
        return {"backend": "portable",
                "adapter_sha256": _sha256(Path(__file__).with_name("glm53_exl3_rocm.py"))}
    from reap.glm53_exl3_native import native_runtime_identity
    return {"backend": "native", **native_runtime_identity(),
            "reference_sha256": _sha256(Path(__file__).with_name("glm53_exl3_rocm.py")),
            "adapter_sha256": _sha256(Path(__file__).with_name("glm53_exl3_native.py"))}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.chmod(0o644)
    temporary.replace(path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file({key: value.contiguous() for key, value in tensors.items()}, temporary)
    temporary.replace(path)


def _upload_and_verify(
    local_path: Path,
    *,
    repo_id: str,
    path_in_repo: str,
) -> str:
    api = HfApi()
    normalized_path = path_in_repo.strip("/")

    def retry_transient(operation: Any, stage: str) -> Any:
        attempts = 5
        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except (httpx.TransportError, TimeoutError) as error:
                if attempt == attempts:
                    raise
                delay = min(2 ** (attempt - 1), 16)
                print(
                    json.dumps(
                        {
                            "event": "hub_retry",
                            "stage": stage,
                            "path_in_repo": normalized_path,
                            "attempt": attempt,
                            "delay_seconds": delay,
                            "error": type(error).__name__,
                        }
                    ),
                    flush=True,
                )
                time.sleep(delay)

    commit = retry_transient(
        lambda: api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=normalized_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Checkpoint {normalized_path}",
        ),
        "upload",
    )
    verified_path = Path(
        retry_transient(
            lambda: hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=normalized_path,
                revision=commit.oid,
                force_download=True,
            ),
            "verify_download",
        )
    )
    if _sha256(verified_path) != _sha256(local_path):
        raise RuntimeError(f"uploaded artifact digest mismatch: {path_in_repo}")
    print(
        json.dumps(
            {
                "event": "upload_verified",
                "path_in_repo": normalized_path,
                "commit": commit.oid,
            }
        ),
        flush=True,
    )
    return commit.oid


class IndexedTensors:
    def __init__(self, model_root: Path):
        self.model_root = model_root.resolve()
        payload = json.loads(
            (self.model_root / "model.safetensors.index.json").read_text()
        )
        self.weight_map: dict[str, str] = payload["weight_map"]

    def get(self, key: str) -> torch.Tensor:
        relative = self.weight_map.get(key)
        if relative is None:
            raise KeyError(f"tensor is not indexed: {key}")
        path = (self.model_root / relative).resolve()
        if not path.is_relative_to(self.model_root) or not path.is_file():
            raise FileNotFoundError(path)
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


def _source_key(target_name: str) -> str:
    aliases = {
        ".attn_hc.fn": ".hc_attn_fn",
        ".attn_hc.base": ".hc_attn_base",
        ".attn_hc.scale": ".hc_attn_scale",
        ".ffn_hc.fn": ".hc_ffn_fn",
        ".ffn_hc.base": ".hc_ffn_base",
        ".ffn_hc.scale": ".hc_ffn_scale",
        ".self_attn.forget_gate.dt_bias": ".self_attn.dt_bias",
        ".self_attn.forget_gate.A_log": ".self_attn.A_log",
        ".self_attn.forget_gate.f_a_proj.weight": ".self_attn.f_a_proj.weight",
        ".self_attn.forget_gate.f_b_proj.weight": ".self_attn.f_b_proj.weight",
    }
    for suffix, replacement in aliases.items():
        if target_name.endswith(suffix):
            return MODEL_PREFIX + target_name[: -len(suffix)] + replacement
    return MODEL_PREFIX + target_name


def _load_text_trunk(model_root: Path, device: torch.device):
    config = AutoConfig.from_pretrained(model_root, local_files_only=True).text_config
    config._attn_implementation = "sdpa"
    with init_empty_weights():
        model = Glm5NextTextModel(config)
    accumulators = {layer: LayerObservation.empty() for layer in ROUTED_LAYERS}
    tensor_index = EXL3TensorIndex(model_root)
    for layer_idx in ROUTED_LAYERS:
        sink = lambda _layer, obs, idx=layer_idx: accumulators[idx].merge_(obs)
        model.layers[layer_idx].mlp.experts = _expert_backend()(
            model_root, layer_idx, observation_sink=sink, tensor_index=tensor_index
        )

    indexed = IndexedTensors(model_root)
    loaded = 0
    skipped = []
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if ".mlp.experts." in name:
            continue
        if name.endswith(".self_attn.conv1d.weight"):
            continue
        key = _source_key(name)
        try:
            source = indexed.get(key)
        except KeyError:
            skipped.append(name)
            continue
        set_module_tensor_to_device(
            model, name, device, value=source, dtype=source.dtype
        )
        loaded += source.numel() * source.element_size()
    for layer_idx, layer in enumerate(model.layers):
        if layer.block_type != "linear_attention":
            continue
        parts = [
            indexed.get(
                f"{MODEL_PREFIX}layers.{layer_idx}.self_attn.{axis}_conv1d.weight"
            )
            for axis in ("q", "k", "v")
        ]
        convolution = torch.cat(parts, dim=0)
        set_module_tensor_to_device(
            model,
            f"layers.{layer_idx}.self_attn.conv1d.weight",
            device,
            value=convolution,
            dtype=convolution.dtype,
        )
        loaded += convolution.numel() * convolution.element_size()
    if skipped:
        raise RuntimeError(f"unresolved retained text tensors: {skipped}")
    model.eval()
    return model, accumulators, loaded


def _text_from_row(row: dict[str, Any]) -> str | None:
    for key in ("text", "content", "document", "prompt", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in row.values():
        if isinstance(value, str) and len(value.strip()) >= 80:
            return value
    return None


def _ours_rows(dataset_revision: str) -> Iterator[tuple[str, str]]:
    dataset = load_dataset(
        "0xSero/reap-calibration-data-v1",
        data_files="calibration-v1.jsonl",
        split="train",
        streaming=True,
        revision=dataset_revision,
    )
    seen: set[str] = set()
    for row in dataset:
        text = _text_from_row(row)
        if text is not None:
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_hash in seen:
                continue
            seen.add(text_hash)
            yield "ours", text


def prepare_tokens(
    *,
    model_root: Path,
    corpus: str,
    output: Path,
    sequence_count: int,
    sequence_length: int,
    dataset_revision: str,
    languages: tuple[str, ...],
) -> Path:
    token_path = output / "token_rows.safetensors"
    manifest_path = output / "token_rows.json"
    if token_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("sequence_count") == sequence_count
            and manifest.get("sequence_length") == sequence_length
            and manifest.get("corpus") == corpus
            and manifest.get("sha256") == _sha256(token_path)
        ):
            return token_path

    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    sequences = []
    source_documents = 0
    source_counts: dict[str, int] = {}
    if corpus == "ours":
        pending: list[int] = []
        for source, text in _ours_rows(dataset_revision):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue
            pending.extend(token_ids)
            source_documents += 1
            source_counts[source] = source_counts.get(source, 0) + 1
            while len(pending) >= sequence_length and len(sequences) < sequence_count:
                sequences.append(pending[:sequence_length])
                del pending[:sequence_length]
            if len(sequences) == sequence_count:
                break
    else:
        if not languages:
            raise ValueError("Wikipedia observation requires at least one language")
        # Build each language independently so long articles in the first
        # stream cannot crowd the remaining languages out of the token budget.
        for position, language in enumerate(languages):
            quota = (sequence_count + len(languages) - 1 - position) // len(languages)
            if quota == 0:
                continue
            dataset = load_dataset(
                "wikimedia/wikipedia",
                f"20231101.{language}",
                split="train",
                streaming=True,
            )
            pending = []
            made = 0
            documents = 0
            for row in dataset:
                text = _text_from_row(row)
                if text is None:
                    continue
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if not token_ids:
                    continue
                pending.extend(token_ids)
                documents += 1
                while len(pending) >= sequence_length and made < quota:
                    sequences.append(pending[:sequence_length])
                    del pending[:sequence_length]
                    made += 1
                if made == quota:
                    break
            if made != quota:
                raise RuntimeError(
                    f"Wikipedia {language} exhausted after {made}/{quota} sequences"
                )
            source_documents += documents
            source_counts[language] = documents
    if len(sequences) != sequence_count:
        raise RuntimeError(
            f"corpus exhausted after {len(sequences)}/{sequence_count} sequences"
        )
    input_ids = torch.tensor(sequences, dtype=torch.int64)
    _atomic_safetensors(token_path, {"input_ids": input_ids})
    _atomic_json(
        manifest_path,
        {
            "schema": "glm53-exl3-reap-token-rows-v1",
            "corpus": corpus,
            "sequence_count": sequence_count,
            "sequence_length": sequence_length,
            "tokens": int(input_ids.numel()),
            "source_documents": source_documents,
            "source_counts": source_counts,
            "dataset_revision": dataset_revision if corpus == "ours" else None,
            "wikipedia_snapshot": "20231101" if corpus == "wikipedia" else None,
            "languages": list(languages) if corpus == "wikipedia" else None,
            "sha256": _sha256(token_path),
            "created_at": _now(),
        },
    )
    return token_path


def _slot_file(slot: Path, group_idx: int) -> Path:
    return slot / f"group-{group_idx:04d}.safetensors"


def _initialize_states(
    model,
    token_path: Path,
    output: Path,
    group_size: int,
) -> tuple[Path, int]:
    progress_path = output / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        slot = output / progress["slot"]
        if slot.is_dir():
            return slot, int(progress["layer"])
        raise RuntimeError(f"progress points to missing slot: {slot}")

    input_ids = load_file(token_path)["input_ids"]
    slot = output / "state-a"
    slot.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for group_idx, start in enumerate(range(0, input_ids.shape[0], group_size)):
            ids = input_ids[start : start + group_size].to(
                model.embed_tokens.weight.device
            )
            hidden = model.embed_tokens(ids).unsqueeze(2).expand(-1, -1, 4, -1)
            _atomic_safetensors(
                _slot_file(slot, group_idx),
                {"input_ids": ids.cpu(), "hidden_streams": hidden.cpu()},
            )
    _atomic_json(
        progress_path,
        {"state": "RUNNING", "layer": -1, "slot": slot.name, "updated_at": _now()},
    )
    return slot, -1


def _replay_groups(
    group_files: list[Path], target_rows: int
) -> Iterator[dict[str, torch.Tensor]]:
    pending: list[dict[str, torch.Tensor]] = []
    rows = 0
    for position, group_file in enumerate(group_files):
        state = load_file(group_file)
        pending.append(state)
        rows += int(state["input_ids"].shape[0])
        if rows < target_rows and position + 1 < len(group_files):
            continue
        merged = {
            "input_ids": torch.cat([item["input_ids"] for item in pending]),
            "hidden_streams": torch.cat([item["hidden_streams"] for item in pending]),
        }
        previous = [item.get("prev_topk_indices") for item in pending]
        if all(value is not None for value in previous):
            merged["prev_topk_indices"] = torch.cat(previous)  # type: ignore[arg-type]
        elif any(value is not None for value in previous):
            raise RuntimeError("replay groups disagree about previous top-k state")
        yield merged
        pending = []
        rows = 0


def _forward_layer_chunked(layer, hidden, ids, previous, config, chunk_size=512):
    """Replay one layer with continuous attention state across bounded queries.

    Cache lifetime is exactly one layer/group. Previous-layer DSA indices retain
    their absolute key positions; only their query dimension is sliced. GLM's
    fallback recurrent kernel requires multiples of 64 (no padding is observed).
    """
    length = ids.shape[1]
    if type(chunk_size) is not int or chunk_size < 64 or chunk_size % 64 or length < 64 or length % 64:
        raise ValueError("sequence length and prefill chunk size must be positive multiples of 64")
    if tuple(hidden.shape[:2]) != tuple(ids.shape):
        raise ValueError("hidden/input sequence geometry mismatch")
    if previous is not None and tuple(previous.shape[:2]) != tuple(ids.shape):
        raise ValueError("previous top-k query geometry mismatch")
    cache = DynamicCache(config=config)
    outputs, routes = [], []
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        chunk_ids = ids[:, start:end]
        result, topk = layer(
            hidden[:, start:end],
            attention_mask=torch.ones_like(chunk_ids, dtype=torch.bool),
            position_ids=torch.arange(start, end, device=ids.device).unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
            prev_topk_indices=None if previous is None else previous[:, start:end],
            input_ids=chunk_ids,
        )
        if tuple(result.shape) != tuple(hidden[:, start:end].shape):
            raise ValueError("layer changed chunk hidden geometry")
        if topk is not None and tuple(topk.shape[:2]) != tuple(chunk_ids.shape):
            raise ValueError("layer changed chunk top-k query geometry")
        outputs.append(result)
        routes.append(topk)
    if any(route is None for route in routes) and not all(route is None for route in routes):
        raise ValueError("layer inconsistently returned top-k state across chunks")
    return torch.cat(outputs, dim=1), (None if routes[0] is None else torch.cat(routes, dim=1))


@torch.inference_mode()
def observe(
    *,
    model_root: Path,
    corpus: str,
    output: Path,
    sequence_count: int,
    sequence_length: int,
    group_size: int,
    dataset_revision: str,
    languages: tuple[str, ...],
    model_revision: str,
    sidecar_repo: str | None = None,
    sidecar_prefix: str | None = None,
    stop_after_layer: int | None = None,
    prepared_tokens: Path | None = None,
    prefill_chunk_size: int = 512,
) -> None:
    if prefill_chunk_size < 64 or prefill_chunk_size % 64 or sequence_length < 64 or sequence_length % 64:
        raise ValueError("sequence length and prefill chunk size must be positive multiples of 64")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm GPU is unavailable to PyTorch")
    device = torch.device("cuda:0")
    output.mkdir(parents=True, exist_ok=True)
    if prepared_tokens is not None:
        token_path = prepared_tokens.resolve()
        with safe_open(token_path, framework="pt", device="cpu") as handle:
            shape = handle.get_slice("input_ids").get_shape()
            if shape != [sequence_count, sequence_length]:
                raise ValueError(
                    f"prepared token shape {shape} does not match requested geometry"
                )
        token_identity = _sha256(token_path)
    else:
        token_identity = None
    identity = {
        "schema": "glm53-observation-run-identity-v1",
        "model_revision": model_revision,
        "corpus": corpus,
        "sequence_count": sequence_count,
        "sequence_length": sequence_length,
        "group_size": group_size,
        "prefill_chunk_size": prefill_chunk_size,
        "prefill_policy": "per-layer-group-dynamic-cache-query-chunks-v1",
        "kda_implementation": _KDA_IDENTITY,
        "expert_implementation": _expert_identity(),
        "dataset_revision": dataset_revision if corpus == "ours" else None,
        "languages": list(languages) if corpus == "wikipedia" else None,
        "prepared_token_sha256": token_identity,
        "corpus_policy": "explicit-calibration-v1-text-dedup-v1",
        "tokenizer_sha256": _sha256(model_root / "tokenizer.json"),
    }
    identity_path = output / "run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError(
                "output belongs to a different observation run; choose a new output directory"
            )
    elif (output / "progress.json").exists() or (output / "token_rows.json").exists():
        raise ValueError(
            "legacy progress has no sealed run identity; choose a new output directory"
        )
    else:
        _atomic_json(identity_path, identity)
    # Two BF16 hidden-stream slots plus token/top-k state and a fixed margin.
    # Fail before initializing a multi-terabyte replay rather than filling disk.
    replay_bytes = sequence_count * sequence_length * (4 * 4096 * 2 + 8 + 8 * 4)
    required_free = replay_bytes * 2 + 8 * 1024**3
    if shutil.disk_usage(output).free < required_free:
        raise RuntimeError(
            f"bounded replay needs {required_free} free bytes; reduce shard size"
        )
    if prepared_tokens is not None:
        _atomic_json(
            output / "token_rows.json",
            {
                **identity,
                "sha256": token_identity,
                "tokens": sequence_count * sequence_length,
            },
        )
    else:
        token_path = prepare_tokens(
            model_root=model_root,
            corpus=corpus,
            output=output,
            sequence_count=sequence_count,
            sequence_length=sequence_length,
            dataset_revision=dataset_revision,
            languages=languages,
        )
    print(
        json.dumps(
            {"event": "tokens_ready", "corpus": corpus, "path": str(token_path)}
        ),
        flush=True,
    )
    if (sidecar_repo is None) != (sidecar_prefix is None):
        raise ValueError("sidecar repo and prefix must be supplied together")
    if sidecar_repo is not None and sidecar_prefix is not None:
        _upload_and_verify(
            output / "token_rows.json",
            repo_id=sidecar_repo,
            path_in_repo=f"{sidecar_prefix}/token_rows.json",
        )
        for completed_observation in sorted(
            (output / "observations").glob("layer-*.json")
        ):
            _upload_and_verify(
                completed_observation,
                repo_id=sidecar_repo,
                path_in_repo=(
                    f"{sidecar_prefix}/observations/{completed_observation.name}"
                ),
            )
    model, accumulators, trunk_bytes = _load_text_trunk(model_root, device)
    print(
        json.dumps({"event": "trunk_ready", "bytes": trunk_bytes}),
        flush=True,
    )
    input_slot, completed_layer = _initialize_states(
        model, token_path, output, group_size
    )
    group_files = sorted(input_slot.glob("group-*.safetensors"))
    if not group_files:
        raise RuntimeError("no replay groups were initialized")

    for layer_idx in range(completed_layer + 1, model.config.num_hidden_layers):
        layer = model.layers[layer_idx]
        experts = layer.mlp.experts if layer_idx in ROUTED_LAYERS else None
        print(json.dumps({"event": "layer_start", "layer": layer_idx}), flush=True)
        if experts is not None:
            experts.materialize_cache(
                device,
                progress=lambda completed, total, idx=layer_idx: (
                    print(
                        json.dumps(
                            {
                                "event": "cache_progress",
                                "layer": idx,
                                "completed": completed,
                                "total": total,
                            }
                        ),
                        flush=True,
                    )
                    if completed % 288 == 0 or completed == total
                    else None
                ),
            )
        output_slot = output / (
            "state-b" if input_slot.name == "state-a" else "state-a"
        )
        if output_slot.exists():
            shutil.rmtree(output_slot)
        output_slot.mkdir(parents=True)
        for group_idx, state in enumerate(_replay_groups(group_files, group_size)):
            ids = state["input_ids"].to(device)
            hidden = state["hidden_streams"].to(device)
            previous = state.get("prev_topk_indices")
            if previous is not None:
                previous = previous.to(device)
            hidden, next_topk = _forward_layer_chunked(
                layer, hidden, ids, previous, model.config,
                chunk_size=prefill_chunk_size,
            )
            tensors = {
                "input_ids": ids.cpu(),
                "hidden_streams": hidden.to("cpu", torch.bfloat16),
            }
            if next_topk is not None:
                tensors["prev_topk_indices"] = next_topk.to("cpu", torch.int32)
            _atomic_safetensors(_slot_file(output_slot, group_idx), tensors)
            del state, ids, hidden, previous, tensors
        if experts is not None:
            experts.clear_cache()
            observation_path = output / "observations" / f"layer-{layer_idx:02d}.json"
            _atomic_json(observation_path, accumulators[layer_idx].as_dict())
            if sidecar_repo is not None and sidecar_prefix is not None:
                _upload_and_verify(
                    observation_path,
                    repo_id=sidecar_repo,
                    path_in_repo=(
                        f"{sidecar_prefix}/observations/{observation_path.name}"
                    ),
                )
        _atomic_json(
            output / "progress.json",
            {
                "state": "RUNNING",
                "layer": layer_idx,
                "slot": output_slot.name,
                "updated_at": _now(),
            },
        )
        shutil.rmtree(input_slot)
        input_slot = output_slot
        group_files = sorted(input_slot.glob("group-*.safetensors"))
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"event": "layer_complete", "layer": layer_idx}), flush=True)
        if stop_after_layer is not None and layer_idx >= stop_after_layer:
            return

    manifest = {
        "state": "COMPLETE",
        "schema": "glm53-selective-exl3-reap-observations-v1",
        "corpus": corpus,
        "model_root_name": model_root.name,
        "model_revision": model_revision,
        "sequence_count": sequence_count,
        "sequence_length": sequence_length,
        "tokens": sequence_count * sequence_length,
        "layers": list(ROUTED_LAYERS),
        "trunk_bytes_loaded": trunk_bytes,
        "device": torch.cuda.get_device_name(device),
        "completed_at": _now(),
    }
    _atomic_json(output / "manifest.json", manifest)
    if sidecar_repo is not None and sidecar_prefix is not None:
        _upload_and_verify(
            output / "manifest.json",
            repo_id=sidecar_repo,
            path_in_repo=f"{sidecar_prefix}/manifest.json",
        )
    _atomic_json(
        output / "progress.json",
        {
            "state": "COMPLETE",
            "layer": 44,
            "slot": input_slot.name,
            "updated_at": _now(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--corpus", choices=("ours", "wikipedia"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-count", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--sidecar-repo")
    parser.add_argument("--sidecar-prefix")
    parser.add_argument(
        "--dataset-revision",
        default="115e754ab8835025e5b59df4b0f30735d8a40ce8",
    )
    parser.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES))
    parser.add_argument("--stop-after-layer", type=int)
    parser.add_argument("--prepared-tokens", type=Path)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    args = parser.parse_args()
    if min(args.sequence_count, args.sequence_length, args.group_size) < 1:
        parser.error("sequence count, length, and group size must be positive")
    observe(
        model_root=args.model_root.resolve(),
        corpus=args.corpus,
        output=args.output.resolve(),
        sequence_count=args.sequence_count,
        sequence_length=args.sequence_length,
        group_size=args.group_size,
        dataset_revision=args.dataset_revision,
        languages=tuple(value for value in args.languages.split(",") if value),
        model_revision=args.model_revision,
        sidecar_repo=args.sidecar_repo,
        sidecar_prefix=args.sidecar_prefix,
        stop_after_layer=args.stop_after_layer,
        prepared_tokens=args.prepared_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
