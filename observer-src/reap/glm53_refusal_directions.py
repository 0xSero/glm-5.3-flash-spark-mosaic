"""Stream GLM-5.3 refusal-direction measurements through selective EXL3.

The full model cannot reside on one MI300X.  This runner reuses the validated
GLM-5.3 layerwise replay path, records contrastive residual means at the final
prompt token, and seals one resumable direction artifact per decoder layer.
Prompt text is never persisted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import torch
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

HARMFUL_DATASET = "mlabonne/harmful_behaviors"
HARMFUL_REVISION = "01cead01398926d81f7c52bdb790ee8cf77ebba7"
HARMLESS_DATASET = "mlabonne/harmless_alpaca"
HARMLESS_REVISION = "02c6a92cfcf11bb0c387334f8146d149d65b587f"


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
    temporary.chmod(0o644)
    temporary.replace(path)


def _row_text(row: dict[str, Any]) -> str | None:
    for key in ("text", "goal", "instruction", "prompt", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _deterministic_prompts(
    rows: Iterable[dict[str, Any]], *, count: int, pool_size: int, salt: str
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for row in rows:
        text = _row_text(row)
        if text is None or text in seen:
            continue
        seen.add(text)
        candidates.append(text)
        if len(candidates) >= pool_size:
            break
    if len(candidates) < count:
        raise RuntimeError(f"prompt dataset produced only {len(candidates)}/{count} rows")
    return sorted(
        candidates,
        key=lambda text: hashlib.sha256(f"{salt}\0{text}".encode()).digest(),
    )[:count]


def _prompt_digest(prompts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def prepare_prompt_rows(
    *,
    model_root: Path,
    output: Path,
    prompts_per_class: int,
    sequence_length: int,
    pool_size: int,
    harmful_data: Path | None = None,
    harmless_data: Path | None = None,
) -> Path:
    rows_path = output / "prompt_rows.safetensors"
    manifest_path = output / "prompt_rows.json"
    if rows_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("prompts_per_class") == prompts_per_class
            and manifest.get("sequence_length") == sequence_length
            and manifest.get("sha256") == _sha256(rows_path)
            and manifest.get("harmful_revision") == HARMFUL_REVISION
            and manifest.get("harmless_revision") == HARMLESS_REVISION
        ):
            return rows_path

    if (harmful_data is None) != (harmless_data is None):
        raise ValueError("harmful and harmless local data must be supplied together")
    if harmful_data is None:
        harmful_rows = load_dataset(
            HARMFUL_DATASET,
            split="train",
            streaming=True,
            revision=HARMFUL_REVISION,
        )
        harmless_rows = load_dataset(
            HARMLESS_DATASET,
            split="train",
            streaming=True,
            revision=HARMLESS_REVISION,
        )
        harmful_source_sha256 = None
        harmless_source_sha256 = None
    else:
        assert harmless_data is not None
        harmful_data = harmful_data.resolve()
        harmless_data = harmless_data.resolve()
        if not harmful_data.is_file() or not harmless_data.is_file():
            raise FileNotFoundError("local refusal prompt parquet is missing")
        harmful_rows = load_dataset(
            "parquet", data_files=str(harmful_data), split="train", streaming=True
        )
        harmless_rows = load_dataset(
            "parquet", data_files=str(harmless_data), split="train", streaming=True
        )
        harmful_source_sha256 = _sha256(harmful_data)
        harmless_source_sha256 = _sha256(harmless_data)
    harmful = _deterministic_prompts(
        harmful_rows,
        count=prompts_per_class,
        pool_size=pool_size,
        salt="glm53-harmful-v1",
    )
    harmless = _deterministic_prompts(
        harmless_rows,
        count=prompts_per_class,
        pool_size=pool_size,
        salt="glm53-harmless-v1",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    encoded = []
    labels = []
    for label, prompts in ((1, harmful), (0, harmless)):
        for prompt in prompts:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            tokens = rendered.get("input_ids") if hasattr(rendered, "get") else rendered
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.flatten().tolist()
            if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
                if len(tokens) != 1:
                    raise RuntimeError("chat template returned multiple prompt rows")
                tokens = tokens[0]
            if not isinstance(tokens, list) or not tokens:
                raise RuntimeError("chat template produced no prompt tokens")
            encoded.append(tokens[-sequence_length:])
            labels.append(label)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if isinstance(pad_id, list):
        pad_id = pad_id[0]
    if not isinstance(pad_id, int):
        raise RuntimeError("tokenizer does not define a usable pad token")
    input_ids = torch.full(
        (len(encoded), sequence_length), pad_id, dtype=torch.int64
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, tokens in enumerate(encoded):
        input_ids[row, -len(tokens) :] = torch.tensor(tokens, dtype=torch.int64)
        attention_mask[row, -len(tokens) :] = True
    class_labels = torch.tensor(labels, dtype=torch.int8)
    _atomic_safetensors(
        rows_path,
        {
            "attention_mask": attention_mask,
            "class_labels": class_labels,
            "input_ids": input_ids,
        },
    )
    _atomic_json(
        manifest_path,
        {
            "schema": "glm53-refusal-prompt-rows-v1",
            "state": "COMPLETE",
            "prompts_per_class": prompts_per_class,
            "sequence_length": sequence_length,
            "pool_size": pool_size,
            "harmful_dataset": HARMFUL_DATASET,
            "harmful_revision": HARMFUL_REVISION,
            "harmful_prompt_digest": _prompt_digest(harmful),
            "harmful_source_sha256": harmful_source_sha256,
            "harmless_dataset": HARMLESS_DATASET,
            "harmless_revision": HARMLESS_REVISION,
            "harmless_prompt_digest": _prompt_digest(harmless),
            "harmless_source_sha256": harmless_source_sha256,
            "sha256": _sha256(rows_path),
            "created_at": _now(),
        },
    )
    return rows_path


def _slot_file(slot: Path, group: int) -> Path:
    return slot / f"group-{group:04d}.safetensors"


def _initialize_states(model, rows_path: Path, output: Path, group_size: int) -> tuple[Path, int]:
    progress_path = output / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        slot = output / progress["slot"]
        if slot.is_dir():
            return slot, int(progress["layer"])
        raise RuntimeError(f"progress points to a missing state slot: {slot}")

    rows = load_file(rows_path)
    slot = output / "state-a"
    slot.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for group, start in enumerate(range(0, rows["input_ids"].shape[0], group_size)):
            ids = rows["input_ids"][start : start + group_size].to(
                model.embed_tokens.weight.device
            )
            hidden = model.embed_tokens(ids).unsqueeze(2).expand(-1, -1, 4, -1)
            _atomic_safetensors(
                _slot_file(slot, group),
                {
                    "attention_mask": rows["attention_mask"][start : start + group_size],
                    "class_labels": rows["class_labels"][start : start + group_size],
                    "hidden_streams": hidden.to("cpu", torch.bfloat16),
                    "input_ids": ids.cpu(),
                },
            )
    _atomic_json(
        progress_path,
        {"state": "RUNNING", "layer": -1, "slot": slot.name, "updated_at": _now()},
    )
    return slot, -1


class ContrastAccumulator:
    def __init__(self, hidden_size: int) -> None:
        self.sum = {
            0: torch.zeros(hidden_size, dtype=torch.float64),
            1: torch.zeros(hidden_size, dtype=torch.float64),
        }
        self.square_norm_sum = {0: 0.0, 1: 0.0}
        self.count = {0: 0, 1: 0}

    def update(self, hidden: torch.Tensor, labels: torch.Tensor) -> None:
        if hidden.ndim != 4:
            raise ValueError("expected [batch,sequence,streams,hidden] residual state")
        final = hidden[:, -1].float().mean(dim=1).cpu().to(torch.float64)
        for label in (0, 1):
            selected = final[labels.cpu() == label]
            if selected.numel() == 0:
                continue
            self.sum[label] += selected.sum(dim=0)
            self.square_norm_sum[label] += float(selected.square().sum().item())
            self.count[label] += int(selected.shape[0])

    def result(self) -> tuple[torch.Tensor, dict[str, Any]]:
        if min(self.count.values()) < 1:
            raise RuntimeError("both prompt classes must contribute residual states")
        means = {label: self.sum[label] / self.count[label] for label in (0, 1)}
        difference = means[1] - means[0]
        norm = torch.linalg.vector_norm(difference)
        if not torch.isfinite(norm):
            raise RuntimeError("refusal direction has a non-finite norm")
        variances = {}
        for label in (0, 1):
            second_moment = self.square_norm_sum[label] / self.count[label]
            variances[label] = max(0.0, second_moment - float(means[label].square().sum()))
        pooled = ((variances[0] + variances[1]) / 2.0) ** 0.5
        valid = bool(norm > 0)
        direction = (
            (difference / norm).to(torch.float32)
            if valid
            else torch.zeros_like(difference, dtype=torch.float32)
        )
        return direction, {
            "valid": valid,
            "harmful_count": self.count[1],
            "harmless_count": self.count[0],
            "difference_norm": float(norm),
            "pooled_within_class_l2_std": pooled,
            "separation": float(norm) / max(pooled, 1e-12) if valid else 0.0,
            "direction_abs_mean": float(direction.abs().mean()),
            "direction_max_abs": float(direction.abs().max()),
        }


@torch.inference_mode()
def measure(
    *,
    model_root: Path,
    output: Path,
    model_revision: str,
    prompts_per_class: int,
    sequence_length: int,
    group_size: int,
    pool_size: int,
    harmful_data: Path | None = None,
    harmless_data: Path | None = None,
    stop_after_layer: int | None = None,
) -> None:
    from reap.glm53_exl3_observe import _load_text_trunk

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is unavailable to PyTorch")
    output.mkdir(parents=True, exist_ok=True)
    rows_path = prepare_prompt_rows(
        model_root=model_root,
        output=output,
        prompts_per_class=prompts_per_class,
        sequence_length=sequence_length,
        pool_size=pool_size,
        harmful_data=harmful_data,
        harmless_data=harmless_data,
    )
    model, _observations, trunk_bytes = _load_text_trunk(model_root, torch.device("cuda:0"))
    input_slot, completed_layer = _initialize_states(model, rows_path, output, group_size)
    group_files = sorted(input_slot.glob("group-*.safetensors"))
    if not group_files:
        raise RuntimeError("no refusal-direction replay groups were initialized")

    for layer_idx in range(completed_layer + 1, model.config.num_hidden_layers):
        layer = model.layers[layer_idx]
        experts = layer.mlp.experts if layer_idx in range(3, 45) else None
        print(json.dumps({"event": "layer_start", "layer": layer_idx}), flush=True)
        if experts is not None:
            experts.materialize_cache(
                "cuda:0",
                progress=lambda completed, total, idx=layer_idx: print(
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
                else None,
            )
        output_slot = output / ("state-b" if input_slot.name == "state-a" else "state-a")
        if output_slot.exists():
            shutil.rmtree(output_slot)
        output_slot.mkdir(parents=True)
        pre = ContrastAccumulator(model.config.hidden_size)
        post = ContrastAccumulator(model.config.hidden_size)
        for group, group_file in enumerate(group_files):
            state = load_file(group_file)
            labels = state["class_labels"]
            hidden = state["hidden_streams"].to("cuda:0")
            pre.update(hidden, labels)
            ids = state["input_ids"].to("cuda:0")
            attention_mask = state["attention_mask"].to("cuda:0")
            previous = state.get("prev_topk_indices")
            if previous is not None:
                previous = previous.to("cuda:0")
            hidden, next_topk = layer(
                hidden,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                prev_topk_indices=previous,
                input_ids=ids,
            )
            post.update(hidden, labels)
            tensors = {
                "attention_mask": state["attention_mask"],
                "class_labels": labels,
                "hidden_streams": hidden.to("cpu", torch.bfloat16),
                "input_ids": state["input_ids"],
            }
            if next_topk is not None:
                tensors["prev_topk_indices"] = next_topk.to("cpu", torch.int32)
            _atomic_safetensors(_slot_file(output_slot, group), tensors)
        if experts is not None:
            experts.clear_cache()
        pre_direction, pre_stats = pre.result()
        post_direction, post_stats = post.result()
        direction_path = output / "directions" / f"layer-{layer_idx:02d}.safetensors"
        _atomic_safetensors(
            direction_path,
            {"resid_post": post_direction, "resid_pre": pre_direction},
        )
        _atomic_json(
            output / "directions" / f"layer-{layer_idx:02d}.json",
            {
                "state": "COMPLETE",
                "layer": layer_idx,
                "resid_pre": pre_stats,
                "resid_post": post_stats,
                "sha256": _sha256(direction_path),
                "completed_at": _now(),
            },
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
        "schema": "glm53-refusal-directions-v1",
        "state": "COMPLETE",
        "model_root_name": model_root.name,
        "model_revision": model_revision,
        "layers": list(range(model.config.num_hidden_layers)),
        "prompts_per_class": prompts_per_class,
        "sequence_length": sequence_length,
        "harmful_dataset": HARMFUL_DATASET,
        "harmful_revision": HARMFUL_REVISION,
        "harmless_dataset": HARMLESS_DATASET,
        "harmless_revision": HARMLESS_REVISION,
        "prompt_rows_sha256": _sha256(rows_path),
        "trunk_bytes_loaded": trunk_bytes,
        "completed_at": _now(),
    }
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(
        output / "progress.json",
        {"state": "COMPLETE", "layer": 44, "slot": input_slot.name, "updated_at": _now()},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--prompts-per-class", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--harmful-data", type=Path)
    parser.add_argument("--harmless-data", type=Path)
    parser.add_argument("--stop-after-layer", type=int)
    args = parser.parse_args()
    if min(
        args.prompts_per_class,
        args.sequence_length,
        args.group_size,
        args.pool_size,
    ) < 1:
        parser.error("prompt, sequence, group, and pool sizes must be positive")
    if args.pool_size < args.prompts_per_class:
        parser.error("pool size must cover prompts per class")
    measure(
        model_root=args.model_root.resolve(),
        output=args.output.resolve(),
        model_revision=args.model_revision,
        prompts_per_class=args.prompts_per_class,
        sequence_length=args.sequence_length,
        group_size=args.group_size,
        pool_size=args.pool_size,
        harmful_data=args.harmful_data,
        harmless_data=args.harmless_data,
        stop_after_layer=args.stop_after_layer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
