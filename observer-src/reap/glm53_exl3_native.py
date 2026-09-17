"""Packed CUDA EXL3 projections with the reference observer's unchanged routing.

No reconstruction or portable fallback: native dependencies are resolved only
when this backend is requested. This supports original Q3/Q4 rank-sliced weights.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import importlib
import importlib.metadata
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open

from reap.glm53_exl3_rocm import (
    HIDDEN_SIZE, LOCAL_INTERMEDIATE_SIZE, NUM_EXPERTS, TP_SIZE, RocmEXL3Experts,
)


def _native_modules():
    extension = importlib.import_module("exllamav3_ext")
    if not callable(getattr(extension, "exl3_gemm", None)):
        raise RuntimeError("native EXL3 extension lacks exl3_gemm")
    wrapper = importlib.import_module("vllm.model_executor.layers.quantization.exl3")
    if not callable(getattr(wrapper, "make_linear_exl3", None)):
        raise RuntimeError("native EXL3 wrapper lacks make_linear_exl3")
    return wrapper, extension


def _native_factory():
    return _native_modules()[0].make_linear_exl3


def _module_digest(module) -> str:
    path = Path(module.__file__)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_runtime_identity() -> dict[str, Any]:
    """Bind wrapper/compiled binary bytes without exposing installation paths."""
    wrapper, extension = _native_modules()
    versions = {}
    for package in ("vllm", "exllamav3"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"schema": "glm53-native-exl3-runtime-v1",
            "wrapper_module": wrapper.__name__, "wrapper_sha256": _module_digest(wrapper),
            "extension_module": extension.__name__, "extension_sha256": _module_digest(extension),
            "package_versions": versions}


def _validate_packed(tensors: dict[str, torch.Tensor], projection: str) -> int:
    if projection not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError("unsupported native projection")
    input_width, output_width = ((LOCAL_INTERMEDIATE_SIZE, HIDDEN_SIZE)
                                 if projection == "down_proj" else (HIDDEN_SIZE, LOCAL_INTERMEDIATE_SIZE))
    trellis = tensors["trellis"]
    if (trellis.dtype != torch.int16 or trellis.ndim != 3 or trellis.shape[-1] not in (48, 64)
            or tuple(trellis.shape[:2]) != (input_width // 16, output_width // 16)):
        raise ValueError("native EXL3 trellis must match original Q3/Q4 projection geometry")
    if (tensors["suh"].numel(), tensors["svh"].numel()) != (input_width, output_width):
        raise ValueError("native EXL3 scale geometry mismatch")
    if not tensors["suh"].is_floating_point() or not tensors["svh"].is_floating_point():
        raise ValueError("native EXL3 scales must retain floating source dtype")
    if tensors["mcg"].numel() not in (0, 1):
        raise ValueError("native EXL3 marker geometry mismatch")
    return trellis.shape[-1] // 16


class NativeEXL3Experts(RocmEXL3Experts):
    """Keep the parent's expert loop/statistics; replace projection kernels only."""

    def materialize_cache(self, device: torch.device | str, *,
                          progress: Callable[[int, int], None] | None = None) -> None:
        if self._projection_cache:
            raise RuntimeError("projection cache is already materialized")
        if self.compute_dtype != torch.float16:
            raise ValueError("qualified native EXL3 observer requires float16 compute dtype")
        factory = _native_factory()  # fail before materializing any weights
        total, completed, layer_bits = NUM_EXPERTS * 3 * TP_SIZE, 0, None
        suffixes = ("trellis", "suh", "svh", "mcg")
        # Resolve every suffix independently; scales need not share a shard.
        paths = {self.index.file_for(f"{self._prefix(expert, projection, rank)}.{suffix}")
                 for expert in range(NUM_EXPERTS) for projection in ("gate_proj", "up_proj", "down_proj")
                 for rank in range(TP_SIZE) for suffix in suffixes}
        try:
            with ExitStack() as stack:
                handles = {path: stack.enter_context(safe_open(path, framework="pt", device="cpu")) for path in paths}
                for expert in range(NUM_EXPERTS):
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        for rank in range(TP_SIZE):
                            prefix = self._prefix(expert, projection, rank)
                            tensors = {suffix: handles[self.index.file_for(f"{prefix}.{suffix}")].get_tensor(f"{prefix}.{suffix}")
                                       for suffix in suffixes}
                            bits = _validate_packed(tensors, projection)
                            if layer_bits is not None and bits != layer_bits:
                                raise ValueError("mixed native Q3/Q4 bitrate in one routed layer")
                            layer_bits = bits
                            packed = {suffix: value.to(device=device) for suffix, value in tensors.items()}
                            inner = factory(**packed, out_dtype=self.compute_dtype)
                            if not callable(getattr(inner, "forward", None)):
                                raise RuntimeError("native EXL3 factory did not return a forward implementation")
                            self._projection_cache[prefix] = inner
                            completed += 1
                            if progress is not None:
                                progress(completed, total)
        except Exception:
            self.clear_cache()
            raise

    def _projection(self, value: torch.Tensor, *, expert: int, projection: str,
                    rank: int, handles=None) -> torch.Tensor:
        prefix = self._prefix(expert, projection, rank)
        inner = self._projection_cache.get(prefix)
        if inner is None:
            raise RuntimeError(f"native projection is not materialized: {prefix}")
        output = inner.forward(value.contiguous().half(), {}, out_dtype=self.compute_dtype)
        width = HIDDEN_SIZE if projection == "down_proj" else LOCAL_INTERMEDIATE_SIZE
        if output.shape != (value.shape[0], width) or output.dtype != self.compute_dtype or output.device != value.device:
            raise RuntimeError("native projection returned unexpected shape, dtype or device")
        return output
