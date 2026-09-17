"""Assemble verified two-worker selective EXL3 artifacts from ablated GLM-5.3."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from safetensors import safe_open


LAYERS = list(range(3, 45))
EXPERTS = 288
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
RANKS = range(4)
SUFFIXES = ("suh", "svh", "trellis", "mcg")
EXPECTED_QUANTIZED_TENSORS = 42 * 288 * 3 * 4 * 4
EXPECTED_RETAINED_TENSORS = 2_482
EXPECTED_ROUTE_TOTAL = 600 * 2048 * 8
PACKED = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.rank(\d+)\.(suh|svh|trellis|mcg)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.chmod(0o644)
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.chmod(0o644)
    temporary.replace(path)


def hardlink_verified(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"assembly source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file():
            raise RuntimeError(f"assembly destination is not a file: {destination}")
        source_stat = source.stat()
        destination_stat = destination.stat()
        if (source_stat.st_dev, source_stat.st_ino) != (
            destination_stat.st_dev,
            destination_stat.st_ino,
        ) and (
            source_stat.st_size != destination_stat.st_size
            or sha256(source) != sha256(destination)
        ):
            raise RuntimeError(
                "assembly destination is neither the expected hardlink nor "
                f"an identical filesystem-emulated link: {destination}"
            )
        return
    os.link(source, destination)
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (source_stat.st_dev, source_stat.st_ino) != (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ) and (
        source_stat.st_size != destination_stat.st_size
        or sha256(source) != sha256(destination)
    ):
        raise RuntimeError(
            f"filesystem-emulated assembly link changed content: {destination}"
        )


def expected_geometry(
    projection: str, suffix: str, integer_k: int
) -> tuple[list[int], str]:
    if projection in {"gate_proj", "up_proj"}:
        shapes = {
            "suh": [4096],
            "svh": [512],
            "trellis": [256, 32, 16 * integer_k],
            "mcg": [],
        }
    else:
        shapes = {
            "suh": [512],
            "svh": [4096],
            "trellis": [32, 256, 16 * integer_k],
            "mcg": [],
        }
    return shapes[suffix], {"suh": "F16", "svh": "F16", "trellis": "I16", "mcg": "I32"}[
        suffix
    ]


def expected_assignments(
    gpu_ids: list[int], experts: int = EXPERTS
) -> dict[int, list[int]]:
    return {
        gpu: list(range(position, experts, len(gpu_ids)))
        for position, gpu in enumerate(gpu_ids)
    }


def _validate_part(
    *,
    quant_root: Path,
    artifact: Path,
    layer: int,
    part: dict[str, Any],
    expected_experts: list[int],
    integer_k: int,
    weight_map: dict[str, str],
) -> dict[str, Any]:
    gpu = int(part["gpu"])
    name = f"layer-{layer:02d}-part-{gpu}.safetensors"
    path = quant_root / "layers" / name
    sidecar_path = quant_root / "layers" / f"layer-{layer:02d}-part-{gpu}.json"
    sidecar = json.loads(sidecar_path.read_text())
    if not (
        sidecar.get("state") == "COMPLETE"
        and sidecar.get("layer") == layer
        and sidecar.get("gpu") == gpu
        and sidecar.get("experts") == expected_experts
        and sidecar.get("bytes") == path.stat().st_size == int(part["bytes"])
        and sidecar.get("sha256") == part["sha256"] == sha256(path)
        and sidecar.get("tensor_count") == len(expected_experts) * 3 * 4 * 4
    ):
        raise RuntimeError(f"quantized part sidecar mismatch: {name}")
    reports = sidecar.get("expert_reports")
    if (
        not isinstance(reports, list)
        or [row.get("expert") for row in reports] != expected_experts
    ):
        raise RuntimeError(f"expert-report coverage mismatch: {name}")
    maximum_nmse = 0.0
    maximum_proxy = 0.0
    minimum_routes = None
    for report in reports:
        hessian = report.get("hessian", {})
        routes = int(hessian.get("selected_natural_routes", 0))
        if routes < 1024:
            raise RuntimeError(f"Hessian route floor failed: {name}")
        minimum_routes = (
            routes if minimum_routes is None else min(minimum_routes, routes)
        )
        slices = report.get("slices", [])
        expected_units = {
            (projection, rank) for projection in PROJECTIONS for rank in RANKS
        }
        if {
            (row.get("projection"), row.get("rank")) for row in slices
        } != expected_units:
            raise RuntimeError(f"projection/rank report coverage mismatch: {name}")
        for row in slices:
            values = [
                float(row.get("proxy_error", 0.0)),
                float(row.get("packed_nmse_vs_bf16", 0.0)),
                float(row.get("packed_roundtrip_relative_l2", 0.0)),
            ]
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise RuntimeError(f"non-finite packed metric: {name}")
            if row.get("q_fallback") not in {None, False}:
                raise RuntimeError(f"quantizer fallback occurred: {name}")
            maximum_proxy = max(maximum_proxy, values[0])
            maximum_nmse = max(maximum_nmse, values[1])
    tensor_bytes = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if len(keys) != len(expected_experts) * 3 * 4 * 4:
            raise RuntimeError(f"packed tensor count mismatch: {name}")
        for key in keys:
            match = PACKED.fullmatch(key)
            if match is None:
                raise RuntimeError(f"packed tensor key is malformed: {key}")
            key_layer, expert, projection, rank, suffix = match.groups()
            if (
                int(key_layer) != layer
                or int(expert) not in expected_experts
                or int(rank) not in RANKS
            ):
                raise RuntimeError(f"packed tensor is outside part scope: {key}")
            shape, dtype = expected_geometry(projection, suffix, integer_k)
            tensor_slice = handle.get_slice(key)
            if (
                list(tensor_slice.get_shape()) != shape
                or tensor_slice.get_dtype() != dtype
            ):
                raise RuntimeError(f"packed tensor geometry mismatch: {key}")
            if key in weight_map:
                raise RuntimeError(f"duplicate packed tensor: {key}")
            relative = f"layers/{name}"
            weight_map[key] = relative
            elements = math.prod(shape) if shape else 1
            tensor_bytes += elements * {"F16": 2, "I16": 2, "I32": 4}[dtype]
    hardlink_verified(path, artifact / "layers" / name)
    hardlink_verified(sidecar_path, artifact / "layers" / sidecar_path.name)
    return {
        "name": name,
        "sha256": part["sha256"],
        "bytes": path.stat().st_size,
        "tensor_count": len(expected_experts) * 3 * 4 * 4,
        "tensor_bytes": tensor_bytes,
        "minimum_routes": minimum_routes,
        "maximum_proxy_error": maximum_proxy,
        "maximum_packed_nmse_vs_bf16": maximum_nmse,
    }


def _validate_retained(
    *,
    retained_root: Path,
    artifact: Path,
    weight_map: dict[str, str],
) -> tuple[list[dict[str, Any]], int, str]:
    manifest_path = retained_root / "retained-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not (
        manifest.get("state") == "COMPLETE"
        and manifest.get("retained_tensor_count") == EXPECTED_RETAINED_TENSORS
        and len(manifest.get("weight_map", {})) == EXPECTED_RETAINED_TENSORS
    ):
        raise RuntimeError("retained manifest scope mismatch")
    records = []
    observed = set()
    for row in manifest["completed_shards"]:
        name = row["output_name"]
        path = retained_root / "shards" / name
        sidecar_path = path.with_name(path.name + ".json")
        sidecar = json.loads(sidecar_path.read_text())
        if not (
            sidecar.get("state") == "COMPLETE"
            and sidecar.get("sha256") == row["sha256"] == sha256(path)
            and sidecar.get("bytes") == row["bytes"] == path.stat().st_size
            and sidecar.get("keys") == row["keys"]
        ):
            raise RuntimeError(f"retained shard sidecar mismatch: {name}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        if sorted(keys) != row["keys"]:
            raise RuntimeError(f"retained shard key mismatch: {name}")
        for key in keys:
            if key in weight_map or key in observed:
                raise RuntimeError(f"duplicate retained tensor: {key}")
            if manifest["weight_map"].get(key) != f"retained/{name}":
                raise RuntimeError(f"retained weight-map mismatch: {key}")
            observed.add(key)
            weight_map[key] = f"retained/{name}"
        hardlink_verified(path, artifact / "retained" / name)
        hardlink_verified(sidecar_path, artifact / "retained" / sidecar_path.name)
        records.append(
            {
                "name": name,
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "tensor_count": row["tensor_count"],
                "tensor_bytes": row["tensor_bytes"],
            }
        )
    if observed != set(manifest["weight_map"]):
        raise RuntimeError("retained tensor closure mismatch")
    shutil.copy2(manifest_path, artifact / "retained" / manifest_path.name)
    return records, int(manifest["retained_tensor_bytes"]), sha256(manifest_path)


def _model_card(
    repo_id: str,
    revision: str,
    bits: float,
    integer_k: int,
    suite_repo: str,
    variant_repos: list[dict[str, Any]],
) -> str:
    sibling_lines = "\n".join(
        f"- [{row['bpw']:.1f}bpw](https://huggingface.co/{row['repo_id']})"
        for row in variant_repos
    )
    return f"""---
base_model: {repo_id}
base_model_relation: quantized
library_name: exllamav3
license: mit
tags:
  - exl3
  - mixture-of-experts
  - abliterated
---

# GLM-5.3 Flash abliterated selective EXL3 {bits:.1f}bpw

This artifact is derived from full-precision GLM-5.3 Flash weights at
`{revision}`. A cross-bitrate refusal direction was selected from sealed Q4
and 3.0bpw observations, projected from the full-precision hidden-state
writers, independently verified, and only then re-quantized.

Only routed expert gate/up/down projections in language layers 3 through 44
are EXL3 K{integer_k}; the verified ablated trunk remains in source precision.
The packed experts contain four tensor-parallel rank slices and require the
included selective EXL3 loader. Runtime and quality claims are recorded only
in their separate verified reports.

## Release suite

- [Suite index](https://huggingface.co/{suite_repo})
{sibling_lines}
"""


def assemble(
    *,
    variant: str,
    quant_root: Path,
    retained_root: Path,
    source: Path,
    artifact: Path,
    campaign_contract_path: Path,
    abliteration_manifest_path: Path,
    abliteration_verification_path: Path,
    reference_abliteration_verification_path: Path,
    source_fingerprint_path: Path,
    reference_fingerprint_path: Path,
    virtual_descriptor_path: Path,
    coverage_manifest_path: Path,
    coverage_verification_path: Path,
    refusal_selection_path: Path,
    runtime_loader_path: Path,
    gpu_ids: list[int],
    chat_template_path: Path | None = None,
    chat_template_provenance_path: Path | None = None,
) -> dict[str, Any]:
    if variant not in {"q3", "q4"}:
        raise RuntimeError("variant must be q3 or q4")
    integer_k = 3 if variant == "q3" else 4
    bits = float(integer_k)
    if gpu_ids != [0, 1]:
        raise RuntimeError("the sealed two-worker assembly requires GPU IDs 0 and 1")
    artifact = artifact.resolve()
    if any(
        artifact == root.resolve()
        or artifact.is_relative_to(root.resolve())
        or root.resolve().is_relative_to(artifact)
        for root in (quant_root, retained_root, source)
    ):
        raise RuntimeError("artifact output must be disjoint from all weight inputs")
    contract = json.loads(campaign_contract_path.read_text())
    derivation_inputs = {
        "campaign_contract_sha256": sha256(campaign_contract_path),
        "canonical_tensor_tree_sha256": contract["derivation"].get(
            "canonical_tensor_tree_sha256"
        ),
        "abliteration_manifest_sha256": sha256(abliteration_manifest_path),
        "abliteration_verification_sha256": sha256(abliteration_verification_path),
        "reference_verification_sha256": sha256(
            reference_abliteration_verification_path
        ),
        "source_fingerprint_sha256": sha256(source_fingerprint_path),
        "reference_fingerprint_sha256": sha256(reference_fingerprint_path),
        "virtual_abliteration_descriptor_sha256": sha256(
            virtual_descriptor_path
        ),
        "route_coverage_manifest_sha256": sha256(coverage_manifest_path),
        "route_coverage_verification_sha256": sha256(coverage_verification_path),
        "refusal_selection_sha256": sha256(refusal_selection_path),
        "runtime_loader_sha256": sha256(runtime_loader_path),
    }
    if (
        contract.get("schema_version") != 1
        or bits not in contract.get("variants", [])
        or contract["derivation"].get("abliteration_manifest_sha256")
        != derivation_inputs["abliteration_manifest_sha256"]
        or contract["derivation"].get("verification_sha256")
        != derivation_inputs["abliteration_verification_sha256"]
        or contract["derivation"].get("reference_verification_sha256")
        != derivation_inputs["reference_verification_sha256"]
        or contract["derivation"].get("source_fingerprint_sha256")
        != derivation_inputs["source_fingerprint_sha256"]
        or contract["derivation"].get("reference_fingerprint_sha256")
        != derivation_inputs["reference_fingerprint_sha256"]
        or contract["derivation"].get(
            "virtual_abliteration_descriptor_sha256"
        )
        != derivation_inputs["virtual_abliteration_descriptor_sha256"]
        or contract["derivation"].get("selection_sha256")
        != derivation_inputs["refusal_selection_sha256"]
        or contract["calibration"].get("route_coverage_manifest_sha256")
        != derivation_inputs["route_coverage_manifest_sha256"]
        or contract["calibration"].get("route_coverage_verification_sha256")
        != derivation_inputs["route_coverage_verification_sha256"]
    ):
        raise RuntimeError("assembly inputs differ from the campaign contract")
    source_fingerprint = json.loads(source_fingerprint_path.read_text())
    reference_fingerprint = json.loads(reference_fingerprint_path.read_text())
    canonical_root = contract["derivation"].get("canonical_tensor_tree_sha256")
    if (
        source_fingerprint.get("state") != "COMPLETE"
        or reference_fingerprint.get("state") != "COMPLETE"
        or source_fingerprint.get("canonical_tree_sha256") != canonical_root
        or reference_fingerprint.get("canonical_tree_sha256") != canonical_root
    ):
        raise RuntimeError("assembly BF16 tensor fingerprints differ from the contract")
    status = json.loads((quant_root / "status.json").read_text())
    if not (
        status.get("state") == "ENCODE_COMPLETE"
        and status.get("quantization_variant") == variant
        and status.get("campaign_contract_sha256")
        == derivation_inputs["campaign_contract_sha256"]
        and status.get("last_layer") == 44
        and status.get("expert_limit") == 288
        and status.get("rows") == 600
    ):
        raise RuntimeError("quantization status is not a derivation-bound release")
    assembly_contract = {
        "schema": "glm53-abliterated-exl3-assembly-contract-v1",
        "state": "COMPLETE",
        "variant": variant,
        "gpu_ids": gpu_ids,
        **derivation_inputs,
    }
    contract_path = artifact / "assembly-contract.json"
    if artifact.exists() and not contract_path.is_file() and any(artifact.iterdir()):
        raise RuntimeError("nonempty artifact output has no assembly contract")
    artifact.mkdir(parents=True, exist_ok=True)
    if (
        contract_path.is_file()
        and json.loads(contract_path.read_text()) != assembly_contract
    ):
        raise RuntimeError("existing assembly contract differs from requested inputs")
    atomic_json(contract_path, assembly_contract)
    atomic_json(artifact / "assembly-status.json", {"state": "RUNNING"})

    assignments = expected_assignments(gpu_ids)
    weight_map: dict[str, str] = {}
    quant_records = []
    maximum_nmse = 0.0
    maximum_proxy = 0.0
    minimum_routes = None
    quant_tensor_bytes = 0
    for layer in LAYERS:
        layer_path = quant_root / "layers" / f"layer-{layer:02d}.json"
        layer_record = json.loads(layer_path.read_text())
        routes = layer_record.get("capture_route_counts")
        if not (
            layer_record.get("state") == "ENCODED"
            and layer_record.get("layer") == layer
            and isinstance(routes, list)
            and len(routes) == EXPERTS
            and min(routes) >= 1024
            and sum(routes) == EXPECTED_ROUTE_TOTAL
        ):
            raise RuntimeError(f"encoded layer route closure mismatch: {layer}")
        parts = layer_record.get("parts")
        if not isinstance(parts, list):
            raise RuntimeError(f"conversion-worker part closure mismatch: {layer}")
        parts = sorted(parts, key=lambda row: int(row["gpu"]))
        if [row.get("gpu") for row in parts] != gpu_ids:
            raise RuntimeError(f"conversion-worker part closure mismatch: {layer}")
        hardlink_verified(layer_path, artifact / "layers" / layer_path.name)
        local_marker = quant_root / "retained-local" / f"layer-{layer:02d}.json"
        marker = json.loads(local_marker.read_text())
        if marker.get("state") != "VERIFIED_LOCAL" or marker.get("layer") != layer:
            raise RuntimeError(f"local durability marker mismatch: {layer}")
        hardlink_verified(local_marker, artifact / "retained-local" / local_marker.name)
        for part in parts:
            record = _validate_part(
                quant_root=quant_root,
                artifact=artifact,
                layer=layer,
                part=part,
                expected_experts=assignments[int(part["gpu"])],
                integer_k=integer_k,
                weight_map=weight_map,
            )
            quant_records.append(record)
            quant_tensor_bytes += record["tensor_bytes"]
            maximum_nmse = max(maximum_nmse, record["maximum_packed_nmse_vs_bf16"])
            maximum_proxy = max(maximum_proxy, record["maximum_proxy_error"])
            part_minimum = int(record["minimum_routes"])
            minimum_routes = (
                part_minimum
                if minimum_routes is None
                else min(minimum_routes, part_minimum)
            )
    if len(weight_map) != EXPECTED_QUANTIZED_TENSORS:
        raise RuntimeError("quantized tensor closure mismatch")
    retained_records, retained_tensor_bytes, retained_manifest_sha = _validate_retained(
        retained_root=retained_root, artifact=artifact, weight_map=weight_map
    )
    if len(weight_map) != EXPECTED_QUANTIZED_TENSORS + EXPECTED_RETAINED_TENSORS:
        raise RuntimeError("assembled tensor closure mismatch")

    metadata_names = (
        ".gitattributes",
        "LICENSE",
        "generation_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    for name in metadata_names:
        path = source / name
        if path.is_file():
            shutil.copy2(path, artifact / name)
    (artifact / "evidence").mkdir(parents=True, exist_ok=True)
    if chat_template_path is not None:
        shutil.copy2(chat_template_path, artifact / "chat_template.jinja")
        if chat_template_provenance_path is None:
            raise RuntimeError("chat-template provenance is required with an override")
        shutil.copy2(
            chat_template_provenance_path,
            artifact / "evidence" / "chat-template-provenance.json",
        )
    elif (source / "chat_template.jinja").is_file():
        shutil.copy2(source / "chat_template.jinja", artifact / "chat_template.jinja")
    shutil.copy2(runtime_loader_path, artifact / runtime_loader_path.name)
    evidence_paths = {
        "campaign-contract.json": campaign_contract_path,
        "abliteration-manifest.json": abliteration_manifest_path,
        "abliteration-verification.json": abliteration_verification_path,
        "reference-abliteration-verification.json": reference_abliteration_verification_path,
        "source-tensor-fingerprint.json": source_fingerprint_path,
        "reference-tensor-fingerprint.json": reference_fingerprint_path,
        "virtual-abliteration-descriptor.json": virtual_descriptor_path,
        "route-coverage-manifest.json": coverage_manifest_path,
        "route-coverage-verification.json": coverage_verification_path,
        "refusal-direction-selection.json": refusal_selection_path,
    }
    for name, path in evidence_paths.items():
        shutil.copy2(path, artifact / "evidence" / name)

    source_config = json.loads((source / "config.json").read_text())
    source_config["quantization_config"] = {
        "quant_method": "exl3_selective_tp4",
        "format": "glm53-abliterated-selective-exl3-tp4-v1",
        "bits_per_weight": bits,
        "trellis_k": integer_k,
        "mcg": True,
        "tensor_parallel_size": 4,
        "conversion_worker_count": len(gpu_ids),
        "quantized_scope": "model.language_model.layers.3..44.mlp.experts.0..287.{gate_proj,up_proj,down_proj}.weight",
        "retained_dtype": "abliterated_source_precision",
        "requires_custom_loader": True,
    }
    atomic_json(artifact / "config.json", source_config)
    atomic_json(
        artifact / "quantization_config.json",
        source_config["quantization_config"],
    )
    shutil.copy2(coverage_manifest_path, artifact / "CALIBRATION_COVERAGE.json")
    source_info = contract["source"]
    publication = contract["publication"]
    atomic_text(
        artifact / "README.md",
        _model_card(
            source_info["repo_id"],
            source_info["revision"],
            bits,
            integer_k,
            publication["suite_repo"],
            publication["variant_repos"],
        ),
    )
    index = {
        "metadata": {
            "format": "glm53-abliterated-selective-exl3-tp4-v1",
            "source_revision": source_info["revision"],
            "tensor_parallel_size": 4,
            "conversion_worker_count": len(gpu_ids),
            "quantized_tensor_count": EXPECTED_QUANTIZED_TENSORS,
            "retained_tensor_count": EXPECTED_RETAINED_TENSORS,
            "quantized_tensor_bytes": quant_tensor_bytes,
            "retained_tensor_bytes": retained_tensor_bytes,
            "total_size": quant_tensor_bytes + retained_tensor_bytes,
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    atomic_json(artifact / "model.safetensors.index.json", index)
    release_files = sorted(
        [
            {
                "path": f"layers/{row['name']}",
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in quant_records
        ]
        + [
            {
                "path": f"retained/{row['name']}",
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in retained_records
        ],
        key=lambda row: row["path"],
    )
    manifest = {
        "schema": "glm53-abliterated-selective-exl3-tp4-v1",
        "state": "COMPLETE",
        "variant": variant,
        "bits_per_weight": bits,
        "integer_k": integer_k,
        "source": source_info,
        "derivation": derivation_inputs,
        "routed_layers": [3, 44],
        "experts_per_layer": EXPERTS,
        "tensor_parallel_size": 4,
        "conversion_gpu_ids": gpu_ids,
        "quantized_tensor_count": EXPECTED_QUANTIZED_TENSORS,
        "retained_tensor_count": EXPECTED_RETAINED_TENSORS,
        "indexed_tensor_count": len(weight_map),
        "quantized_tensor_bytes": quant_tensor_bytes,
        "retained_tensor_bytes": retained_tensor_bytes,
        "minimum_natural_routes": minimum_routes,
        "maximum_proxy_error": maximum_proxy,
        "maximum_packed_nmse_vs_bf16": maximum_nmse,
        "quantized_shards": quant_records,
        "retained_shards": retained_records,
        "retained_manifest_sha256": retained_manifest_sha,
        "calibration": {
            "tokens": contract["calibration"]["tokens"],
            "axes": contract["calibration"]["axes"],
            "natural_topk_routing": True,
            "forced_expert_activation": False,
            "route_floor": contract["calibration"]["route_floor"],
            "minimum_route_count": minimum_routes,
            "coverage_file": "CALIBRATION_COVERAGE.json",
            "coverage_sha256": sha256(artifact / "CALIBRATION_COVERAGE.json"),
        },
        "runtime": {
            "status": "pending",
            "requires_custom_loader": True,
            "loader": runtime_loader_path.name,
            "cuda_graphs_required": True,
            "eager_execution_forbidden": True,
        },
        "files": release_files,
        "model_index_sha256": sha256(artifact / "model.safetensors.index.json"),
        "config_sha256": sha256(artifact / "config.json"),
        "quantization_config_sha256": sha256(artifact / "quantization_config.json"),
        "runtime_loader_sha256": sha256(artifact / runtime_loader_path.name),
        "assembled_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime_claim": False,
        "quality_claim": False,
    }
    atomic_json(artifact / "EXL3_MANIFEST.json", manifest)
    atomic_json(
        artifact / "assembly-status.json",
        {
            "state": "COMPLETE",
            "manifest_sha256": sha256(artifact / "EXL3_MANIFEST.json"),
            "indexed_tensor_count": len(weight_map),
            "runtime_claim": False,
            "quality_claim": False,
        },
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("q3", "q4"), required=True)
    parser.add_argument("--quant-root", type=Path, required=True)
    parser.add_argument("--retained-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--campaign-contract", type=Path, required=True)
    parser.add_argument("--abliteration-manifest", type=Path, required=True)
    parser.add_argument("--abliteration-verification", type=Path, required=True)
    parser.add_argument(
        "--reference-abliteration-verification", type=Path, required=True
    )
    parser.add_argument("--source-fingerprint", type=Path, required=True)
    parser.add_argument("--reference-fingerprint", type=Path, required=True)
    parser.add_argument(
        "--virtual-abliteration-descriptor", type=Path, required=True
    )
    parser.add_argument("--coverage-manifest", type=Path, required=True)
    parser.add_argument("--coverage-verification", type=Path, required=True)
    parser.add_argument("--refusal-selection", type=Path, required=True)
    parser.add_argument("--runtime-loader", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=int, nargs="+", required=True)
    parser.add_argument("--chat-template", type=Path)
    parser.add_argument("--chat-template-provenance", type=Path)
    args = parser.parse_args()
    manifest = assemble(
        variant=args.variant,
        quant_root=args.quant_root,
        retained_root=args.retained_root,
        source=args.source,
        artifact=args.artifact,
        campaign_contract_path=args.campaign_contract,
        abliteration_manifest_path=args.abliteration_manifest,
        abliteration_verification_path=args.abliteration_verification,
        reference_abliteration_verification_path=args.reference_abliteration_verification,
        source_fingerprint_path=args.source_fingerprint,
        reference_fingerprint_path=args.reference_fingerprint,
        virtual_descriptor_path=args.virtual_abliteration_descriptor,
        coverage_manifest_path=args.coverage_manifest,
        coverage_verification_path=args.coverage_verification,
        refusal_selection_path=args.refusal_selection,
        runtime_loader_path=args.runtime_loader,
        gpu_ids=args.gpu_ids,
        chat_template_path=args.chat_template,
        chat_template_provenance_path=args.chat_template_provenance,
    )
    print(json.dumps({"state": manifest["state"], "variant": manifest["variant"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
