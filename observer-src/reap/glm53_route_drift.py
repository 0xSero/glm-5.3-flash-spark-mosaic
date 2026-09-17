"""Fail-closed cross-accelerator routing-drift validation for GLM-5.3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


MAXIMUM_ASSIGNMENT_CHANGE_FRACTION = 0.005
MAXIMUM_PER_EXPERT_RELATIVE_DELTA = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts_sha256(counts: list[int]) -> str:
    encoded = json.dumps(counts, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_route_counts(
    *,
    actual: list[int],
    sealed: list[int],
    route_floor: int,
    maximum_assignment_change_fraction: float = MAXIMUM_ASSIGNMENT_CHANGE_FRACTION,
    maximum_per_expert_relative_delta: float = MAXIMUM_PER_EXPERT_RELATIVE_DELTA,
) -> dict[str, Any]:
    if not (
        len(actual) == len(sealed) > 0
        and route_floor > 0
        and 0 <= maximum_assignment_change_fraction < 1
        and 0 <= maximum_per_expert_relative_delta < 1
        and all(isinstance(value, int) and value >= 0 for value in actual + sealed)
    ):
        raise RuntimeError("route-drift inputs are malformed")
    actual_total = sum(actual)
    sealed_total = sum(sealed)
    if actual_total != sealed_total or actual_total <= 0:
        raise RuntimeError("route total differs from sealed coverage")
    if min(actual) < route_floor:
        raise RuntimeError("actual route coverage falls below the sealed floor")
    absolute_deltas = [abs(left - right) for left, right in zip(actual, sealed)]
    l1_delta = sum(absolute_deltas)
    if l1_delta % 2:
        raise RuntimeError("route-count L1 delta cannot represent reassignment")
    assignment_changes = l1_delta // 2
    assignment_change_fraction = assignment_changes / actual_total
    relative_deltas = [
        delta / expected if expected else (0.0 if delta == 0 else float("inf"))
        for delta, expected in zip(absolute_deltas, sealed)
    ]
    maximum_relative_delta = max(relative_deltas)
    if assignment_change_fraction > maximum_assignment_change_fraction:
        raise RuntimeError("aggregate route drift exceeds the sealed tolerance")
    if maximum_relative_delta > maximum_per_expert_relative_delta:
        raise RuntimeError("per-expert route drift exceeds the sealed tolerance")
    return {
        "schema": "glm53-cross-accelerator-route-drift-v1",
        "state": "COMPLETE",
        "actual_route_counts_sha256": _counts_sha256(actual),
        "sealed_route_counts_sha256": _counts_sha256(sealed),
        "route_total": actual_total,
        "route_floor": route_floor,
        "minimum_actual_route_count": min(actual),
        "changed_expert_count": sum(delta != 0 for delta in absolute_deltas),
        "l1_route_count_delta": l1_delta,
        "assignment_changes": assignment_changes,
        "assignment_change_fraction": assignment_change_fraction,
        "maximum_absolute_expert_delta": max(absolute_deltas),
        "maximum_relative_expert_delta": maximum_relative_delta,
        "maximum_assignment_change_fraction": maximum_assignment_change_fraction,
        "maximum_per_expert_relative_delta": maximum_per_expert_relative_delta,
    }


def install_bounded_route_capture(
    controller: Any,
    *,
    evidence_root: Path | None = None,
) -> Callable[..., dict[str, Any]]:
    """Wrap a pinned controller without weakening any non-routing invariant."""

    original = controller.forward_and_capture

    def bounded_forward_and_capture(
        layer_idx: int,
        input_slot: Path,
        output_slot: Path,
        capture_dir: Path,
        rows: int,
        capture_activations: bool,
        controller_pid: int,
    ) -> dict[str, Any]:
        try:
            manifest = original(
                layer_idx,
                input_slot,
                output_slot,
                capture_dir,
                rows,
                capture_activations,
                controller_pid,
            )
        except RuntimeError as error:
            if not (
                str(error)
                == f"layer {layer_idx} route counts differ from sealed coverage"
                and capture_activations
                and rows == controller.ROWS
            ):
                raise
            workers = [
                json.loads(path.read_text())
                for path in sorted(capture_dir.glob("worker-*.json"))
            ]
            if not (
                len(workers) == len(controller.GPU_IDS)
                and all(item.get("state") == "COMPLETE" for item in workers)
                and all(
                    controller.verified_state(output_slot, row)
                    for row in range(rows)
                )
            ):
                raise RuntimeError(
                    "bounded route capture worker closure differs"
                ) from error
            manifest = {
                "state": "COMPLETE",
                "layer": layer_idx,
                "rows": rows,
                "columns": controller.COLUMNS,
                "capture_activations": True,
                "workers": sorted(workers, key=lambda item: item["gpu"]),
                "completed_at": controller.now(),
            }

        if not (capture_activations and rows == controller.ROWS):
            return manifest
        workers = manifest["workers"]
        actual = [
            sum(item["route_counts"][expert] for item in workers)
            for expert in range(controller.EXPERTS)
        ]
        sealed_payload = json.loads(
            (controller.COVERAGE / "layers" / f"layer-{layer_idx:02d}.json").read_text()
        )
        coverage = json.loads(
            (controller.COVERAGE / "coverage-manifest.json").read_text()
        )
        drift = validate_route_counts(
            actual=actual,
            sealed=sealed_payload["route_counts"],
            route_floor=int(coverage["route_floor"]),
        )
        expected_x = rows * controller.COLUMNS * controller.HIDDEN * 2
        expected_ids = rows * controller.COLUMNS * controller.TOP_K * 2
        if sum(item["x_bytes"] for item in workers) != expected_x:
            raise RuntimeError("captured x payload size mismatch")
        if sum(item["ids_bytes"] for item in workers) != expected_ids:
            raise RuntimeError("captured id payload size mismatch")
        manifest.update(
            {
                "route_counts": actual,
                "minimum_route_count": min(actual),
                "maximum_route_count": max(actual),
                "total_routes": sum(actual),
                "route_drift_vs_sealed": drift,
            }
        )
        manifest_path = capture_dir / "manifest.json"
        controller.atomic_json(manifest_path, manifest)
        if evidence_root is not None:
            evidence_root.mkdir(parents=True, exist_ok=True)
            worker_evidence = []
            for item in sorted(workers, key=lambda worker: worker["gpu"]):
                worker_path = capture_dir / f"worker-{item['gpu']}.json"
                worker_evidence.append(
                    {
                        key: item[key]
                        for key in (
                            "state",
                            "gpu",
                            "layer",
                            "elapsed_seconds",
                            "loaded_bytes",
                            "x_bytes",
                            "x_sha256",
                            "ids_bytes",
                            "ids_sha256",
                        )
                        if key in item
                    }
                    | {
                        "worker_manifest_sha256": _sha256(worker_path),
                    }
                )
            controller.atomic_json(
                evidence_root / f"layer-{layer_idx:02d}.json",
                {
                    "schema": "glm53-durable-route-drift-evidence-v1",
                    "state": "COMPLETE",
                    "layer": layer_idx,
                    "rows": rows,
                    "columns": controller.COLUMNS,
                    "completed_at": manifest["completed_at"],
                    "capture_manifest_sha256": _sha256(manifest_path),
                    "route_counts": actual,
                    "minimum_route_count": min(actual),
                    "maximum_route_count": max(actual),
                    "total_routes": sum(actual),
                    "route_drift_vs_sealed": drift,
                    "workers": worker_evidence,
                },
            )
        return manifest

    controller.forward_and_capture = bounded_forward_and_capture
    return bounded_forward_and_capture
