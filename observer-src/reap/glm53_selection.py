"""Validate GLM-5.3 observation lanes and build pruning candidate sweeps."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROUTED_LAYERS = tuple(range(3, 45))
NUM_EXPERTS = 288
DEFAULT_VARIANTS = ("exl3-3bpw", "exl3-q4")
DEFAULT_CORPORA = ("ours", "wikipedia")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing required artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_vector(value: Any, *, path: Path, name: str, experts: int) -> list[float]:
    if not isinstance(value, list) or len(value) != experts:
        raise ValueError(f"{path}: {name} must contain {experts} entries")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) or item < 0 for item in result):
        raise ValueError(f"{path}: {name} contains invalid values")
    return result


def _rank(values: list[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda expert: (values[expert], expert))


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _candidate_count(ratio: float, experts: int) -> int:
    if not 0 < ratio < 1:
        raise ValueError(f"prune ratio must be between zero and one: {ratio}")
    return max(1, int(experts * ratio))


def build_candidate_sweep(
    root: Path,
    *,
    ratios: Iterable[float],
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    corpora: tuple[str, ...] = DEFAULT_CORPORA,
    layers: tuple[int, ...] = ROUTED_LAYERS,
    experts: int = NUM_EXPERTS,
    selected_ratio: float | None = None,
) -> dict[str, Any]:
    """Build corpus-balanced and cross-bitrate candidate maps from sealed runs."""

    ratios = tuple(sorted(set(float(ratio) for ratio in ratios)))
    if not ratios:
        raise ValueError("at least one prune ratio is required")
    for ratio in ratios:
        _candidate_count(ratio, experts)
    if selected_ratio is not None and selected_ratio not in ratios:
        raise ValueError("selected ratio must also be present in the ratio sweep")

    provenance: dict[str, Any] = {}
    observations: dict[str, dict[str, dict[int, dict[str, list[float]]]]] = {}
    expected_tokens: int | None = None
    for variant in variants:
        observations[variant] = {}
        provenance[variant] = {}
        for corpus in corpora:
            lane = root / variant / corpus
            manifest_path = lane / "manifest.json"
            manifest = _read_json(manifest_path)
            if manifest.get("state") != "COMPLETE":
                raise ValueError(f"lane is not complete: {lane}")
            if tuple(manifest.get("layers", ())) != layers:
                raise ValueError(f"unexpected routed-layer coverage: {manifest_path}")
            tokens = manifest.get("tokens")
            if not isinstance(tokens, int) or tokens <= 0:
                raise ValueError(f"invalid token count: {manifest_path}")
            if expected_tokens is None:
                expected_tokens = tokens
            elif tokens != expected_tokens:
                raise ValueError("all corpora and variants must use equal token counts")
            revision = manifest.get("model_revision")
            if not isinstance(revision, str) or not revision:
                raise ValueError(f"missing model revision: {manifest_path}")
            provenance[variant][corpus] = {
                "model_revision": revision,
                "tokens": tokens,
                "manifest_sha256": _sha256(manifest_path),
            }
            observations[variant][corpus] = {}
            for layer in layers:
                path = lane / "observations" / f"layer-{layer:02d}.json"
                payload = _read_json(path)
                observations[variant][corpus][layer] = {
                    "reap_score": _validate_vector(
                        payload.get("reap_score"), path=path, name="reap_score", experts=experts
                    ),
                    "weighted_ean_sum": _validate_vector(
                        payload.get("weighted_ean_sum"),
                        path=path,
                        name="weighted_ean_sum",
                        experts=experts,
                    ),
                    "expert_frequency": _validate_vector(
                        payload.get("expert_frequency"),
                        path=path,
                        name="expert_frequency",
                        experts=experts,
                    ),
                }

    variant_results: dict[str, Any] = {}
    balanced_ranks: dict[str, dict[int, list[int]]] = {}
    for variant in variants:
        variant_results[variant] = {"layers": {}}
        balanced_ranks[variant] = {}
        for layer in layers:
            corpus_scores = {
                corpus: observations[variant][corpus][layer]["reap_score"]
                for corpus in corpora
            }
            balanced = [
                sum(corpus_scores[corpus][expert] for corpus in corpora) / len(corpora)
                for expert in range(experts)
            ]
            pooled = []
            for expert in range(experts):
                numerator = sum(
                    observations[variant][corpus][layer]["weighted_ean_sum"][expert]
                    for corpus in corpora
                )
                denominator = sum(
                    observations[variant][corpus][layer]["expert_frequency"][expert]
                    for corpus in corpora
                )
                pooled.append(numerator / denominator if denominator else 0.0)
            balanced_rank = _rank(balanced)
            balanced_ranks[variant][layer] = balanced_rank
            corpus_ranks = {corpus: _rank(corpus_scores[corpus]) for corpus in corpora}
            ratio_rows: dict[str, Any] = {}
            for ratio in ratios:
                count = _candidate_count(ratio, experts)
                corpus_sets = {corpus: set(corpus_ranks[corpus][:count]) for corpus in corpora}
                pairwise = [
                    _jaccard(corpus_sets[left], corpus_sets[right])
                    for index, left in enumerate(corpora)
                    for right in corpora[index + 1 :]
                ]
                ratio_rows[str(ratio)] = {
                    "count": count,
                    "experts": balanced_rank[:count],
                    "corpus_jaccard_mean": sum(pairwise) / len(pairwise) if pairwise else 1.0,
                }
            variant_results[variant]["layers"][str(layer)] = {
                "balanced_reap_score": balanced,
                "pooled_reap_score": pooled,
                "ranked_experts_low_to_high": balanced_rank,
                "candidate_sweep": ratio_rows,
            }

    consensus_layers: dict[str, Any] = {}
    for layer in layers:
        positions = {
            variant: {expert: position for position, expert in enumerate(balanced_ranks[variant][layer])}
            for variant in variants
        }
        consensus_score = [
            sum(positions[variant][expert] / max(1, experts - 1) for variant in variants)
            / len(variants)
            for expert in range(experts)
        ]
        consensus_rank = _rank(consensus_score)
        ratio_rows = {}
        for ratio in ratios:
            count = _candidate_count(ratio, experts)
            variant_sets = {
                variant: set(balanced_ranks[variant][layer][:count]) for variant in variants
            }
            pairwise = [
                _jaccard(variant_sets[left], variant_sets[right])
                for index, left in enumerate(variants)
                for right in variants[index + 1 :]
            ]
            ratio_rows[str(ratio)] = {
                "count": count,
                "experts": consensus_rank[:count],
                "variant_jaccard_mean": sum(pairwise) / len(pairwise) if pairwise else 1.0,
            }
        consensus_layers[str(layer)] = {
            "mean_rank_percentile": consensus_score,
            "ranked_experts_low_to_high": consensus_rank,
            "candidate_sweep": ratio_rows,
        }

    result: dict[str, Any] = {
        "schema": "glm53-selective-exl3-pruning-candidates-v1",
        "state": "SELECTED" if selected_ratio is not None else "CANDIDATE_SWEEP",
        "score": "equal-corpus mean of paper-scale REAP; cross-variant mean rank percentile",
        "variants": list(variants),
        "corpora": list(corpora),
        "layers": list(layers),
        "experts_per_layer": experts,
        "prune_ratios": list(ratios),
        "selected_ratio": selected_ratio,
        "provenance": provenance,
        "variant_results": variant_results,
        "consensus": {"layers": consensus_layers},
    }
    if selected_ratio is not None:
        key = str(selected_ratio)
        result["selected_experts"] = {
            layer: consensus_layers[layer]["candidate_sweep"][key]["experts"]
            for layer in consensus_layers
        }
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# GLM-5.3 selective-EXL3 pruning candidates",
        "",
        f"State: `{result['state']}`",
        "",
        "No ratio is an accepted pruning decision unless `selected_ratio` is populated.",
        "",
        "| Ratio | Experts/layer | Mean corpus agreement | Mean bitrate agreement |",
        "|---:|---:|---:|---:|",
    ]
    for ratio in result["prune_ratios"]:
        key = str(ratio)
        corpus_values = []
        variant_values = []
        for variant in result["variants"]:
            for layer in result["layers"]:
                corpus_values.append(
                    result["variant_results"][variant]["layers"][str(layer)]["candidate_sweep"][key][
                        "corpus_jaccard_mean"
                    ]
                )
        for layer in result["layers"]:
            variant_values.append(
                result["consensus"]["layers"][str(layer)]["candidate_sweep"][key][
                    "variant_jaccard_mean"
                ]
            )
        count = result["consensus"]["layers"][str(result["layers"][0])]["candidate_sweep"][key]["count"]
        lines.append(
            f"| {ratio:.3f} | {count} | {sum(corpus_values) / len(corpus_values):.4f} | "
            f"{sum(variant_values) / len(variant_values):.4f} |"
        )
    lines.extend(["", "## Provenance", ""])
    for variant in result["variants"]:
        for corpus in result["corpora"]:
            row = result["provenance"][variant][corpus]
            lines.append(
                f"- `{variant}/{corpus}`: revision `{row['model_revision']}`, "
                f"manifest SHA-256 `{row['manifest_sha256']}`, {row['tokens']} tokens"
            )
    return "\n".join(lines) + "\n"
