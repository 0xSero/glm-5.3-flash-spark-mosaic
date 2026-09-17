"""Select a cross-bitrate GLM-5.3 refusal direction.

The selector accepts two completed observer roots, verifies every candidate
digest, and ranks only directions that agree in sign across both bitrates.
It emits a normalized consensus vector plus a complete candidate report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from reap.glm53_refusal_directions import _atomic_json, _atomic_safetensors, _sha256


def _manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing completed observer manifest: {path}")
    payload = json.loads(path.read_text())
    if payload.get("state") != "COMPLETE":
        raise RuntimeError(f"observer root is not complete: {root}")
    layers = payload.get("layers")
    if not isinstance(layers, list) or not layers:
        raise RuntimeError(f"observer manifest has no layers: {path}")
    return payload


def _candidate(root: Path, layer: int, position: str) -> tuple[torch.Tensor, dict]:
    tensor_path = root / "directions" / f"layer-{layer:02d}.safetensors"
    metadata_path = root / "directions" / f"layer-{layer:02d}.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("state") != "COMPLETE" or metadata.get("layer") != layer:
        raise RuntimeError(f"invalid direction metadata: {metadata_path}")
    actual_sha = _sha256(tensor_path)
    if actual_sha != metadata.get("sha256"):
        raise RuntimeError(f"direction digest mismatch: {tensor_path}")
    stats = metadata.get(position)
    if not isinstance(stats, dict):
        raise RuntimeError(f"missing {position} statistics: {metadata_path}")
    tensors = load_file(tensor_path)
    direction = tensors[position].to(torch.float64)
    return direction, {**stats, "sha256": actual_sha}


def _harmonic_mean(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return 2.0 * left * right / (left + right)


def select(
    *,
    left_root: Path,
    right_root: Path,
    output: Path,
    minimum_cosine: float = 0.8,
    minimum_layer: int = 0,
    maximum_layer: int = 44,
) -> dict[str, Any]:
    if not -1.0 <= minimum_cosine <= 1.0:
        raise ValueError("minimum cosine must be between -1 and 1")
    left_manifest = _manifest(left_root)
    right_manifest = _manifest(right_root)
    if left_manifest.get("prompt_rows_sha256") != right_manifest.get(
        "prompt_rows_sha256"
    ):
        raise RuntimeError("observer roots do not share identical sealed prompt rows")
    left_layers = set(left_manifest["layers"])
    right_layers = set(right_manifest["layers"])
    layers = sorted(
        layer
        for layer in left_layers & right_layers
        if minimum_layer <= layer <= maximum_layer
    )
    if not layers:
        raise RuntimeError("no shared layers fall inside the requested range")

    candidates: list[dict[str, Any]] = []
    vectors: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in layers:
        for position in ("resid_pre", "resid_post"):
            left, left_stats = _candidate(left_root, layer, position)
            right, right_stats = _candidate(right_root, layer, position)
            if left.shape != right.shape or left.ndim != 1:
                raise RuntimeError(f"direction shape mismatch at layer {layer} {position}")
            valid = bool(left_stats.get("valid")) and bool(right_stats.get("valid"))
            cosine = (
                float(torch.nn.functional.cosine_similarity(left, right, dim=0))
                if valid
                else 0.0
            )
            separation = _harmonic_mean(
                float(left_stats.get("separation", 0.0)),
                float(right_stats.get("separation", 0.0)),
            )
            eligible = valid and math.isfinite(cosine) and cosine >= minimum_cosine
            score = cosine * separation if eligible else None
            candidates.append(
                {
                    "layer": layer,
                    "position": position,
                    "eligible": eligible,
                    "cosine": cosine,
                    "left_separation": float(left_stats.get("separation", 0.0)),
                    "right_separation": float(right_stats.get("separation", 0.0)),
                    "harmonic_separation": separation,
                    "score": score,
                    "left_sha256": left_stats["sha256"],
                    "right_sha256": right_stats["sha256"],
                }
            )
            vectors[(layer, position)] = (left, right)

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise RuntimeError("no valid cross-bitrate direction meets minimum cosine")
    winner = max(eligible, key=lambda candidate: candidate["score"])
    left, right = vectors[(winner["layer"], winner["position"])]
    consensus = left + right
    consensus /= torch.linalg.vector_norm(consensus)
    consensus = consensus.to(torch.float32)

    output.mkdir(parents=True, exist_ok=True)
    tensor_path = output / "refusal_direction.safetensors"
    _atomic_safetensors(tensor_path, {"direction": consensus})
    report = {
        "schema": "glm53-cross-bitrate-refusal-direction-v1",
        "state": "COMPLETE",
        "method": "maximum cosine times harmonic separation; unflipped mean",
        "minimum_cosine": minimum_cosine,
        "minimum_layer": minimum_layer,
        "maximum_layer": maximum_layer,
        "prompt_rows_sha256": left_manifest["prompt_rows_sha256"],
        "left_model_revision": left_manifest.get("model_revision"),
        "right_model_revision": right_manifest.get("model_revision"),
        "selected": winner,
        "direction_sha256": _sha256(tensor_path),
        "candidates": candidates,
    }
    _atomic_json(output / "selection.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-root", type=Path, required=True)
    parser.add_argument("--right-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-cosine", type=float, default=0.8)
    parser.add_argument("--minimum-layer", type=int, default=0)
    parser.add_argument("--maximum-layer", type=int, default=44)
    args = parser.parse_args()
    report = select(**vars(args))
    print(json.dumps({"selected": report["selected"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
