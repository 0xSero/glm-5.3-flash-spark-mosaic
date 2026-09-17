"""A storage-bounded, content-addressed view of the GLM-5.3 abliteration.

The base BF16 checkpoint remains immutable on disk.  Callers load one tensor
at a time. This module either projects tensors in the sealed abliteration scope
or replaces them with exact tensors streamed from an immutable Hub revision.
A descriptor binds the base checkpoint, Hot Aisle evidence, implementation,
and canonical tensor root. This avoids materializing a second 643 GB checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import fcntl
import time
from typing import Any
from contextlib import suppress

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file

from reap.glm53_abliterate_bf16 import project_tensor, target_kind


DESCRIPTOR_ENV = "GLM53_VIRTUAL_ABLITERATION_DESCRIPTOR"
TENSOR_SEMANTICS_SHA256 = (
    "22164a6f968c6cc37a5e5f1cec63f111f3a05781cb6fe6dd5cd5d0a59dd21eac"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_sha256() -> str:
    """Return the sealed tensor-semantics revision.

    Cache concurrency, locking, and eviction may change without invalidating
    retained tensor bytes. Bump this value only when tensor selection,
    projection, or exact-overlay decoding semantics change.
    """
    return TENSOR_SEMANTICS_SHA256


class VirtualAbliteration:
    """Project sealed source tensors lazily and fail closed on provenance drift."""

    def __init__(
        self,
        *,
        descriptor_path: Path,
        source_root: Path,
        device: torch.device | None = None,
    ) -> None:
        descriptor_path = descriptor_path.resolve()
        source_root = source_root.resolve()
        payload: dict[str, Any] = json.loads(descriptor_path.read_text())
        index_path = source_root / "model.safetensors.index.json"
        if not (
            payload.get("schema") == "glm53-virtual-abliteration-v1"
            and payload.get("state") == "COMPLETE"
            and payload.get("base_source_root") == str(source_root)
            and payload.get("base_source_index_sha256") == sha256(index_path)
            and payload.get("projection_implementation_sha256")
            == implementation_sha256()
            and payload.get("canonical_tree_sha256")
            == "23eb5e77b9fdab6a5cc439121c73babc36eb4a55ec928ceffa52733c31ea64c3"
        ):
            raise RuntimeError("virtual abliteration descriptor is outside its seal")
        self.mode = payload.get("mode", "project")
        direction = None
        if self.mode == "project":
            direction_path = Path(payload.get("direction_path", "")).resolve()
            if payload.get("direction_sha256") != sha256(direction_path):
                raise RuntimeError("virtual abliteration direction is outside its seal")
            direction = load_file(direction_path).get("direction")
            if direction is None or direction.ndim != 1:
                raise RuntimeError("virtual abliteration direction is malformed")
        elif self.mode != "immutable_hub_exact_overlay":
            raise RuntimeError("virtual abliteration mode is not supported")
        if device is None:
            device = (
                torch.device(f"cuda:{torch.cuda.current_device()}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.descriptor_path = descriptor_path
        self.payload = payload
        self.direction = direction
        self.device = device
        self.maximum_residual = float(payload.get("maximum_residual", 5e-4))
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        self.hub_shards = {
            row["name"]: row for row in payload.get("hub_shards", [])
        }
        self.cache_root = Path(
            os.environ.get(
                "GLM53_ABLITERATION_CACHE_ROOT",
                payload.get("cache_root", "/dev/shm/glm53-abliteration-overlay"),
            )
        ).resolve()
        self.maximum_cached_shards = int(
            os.environ.get(
                "GLM53_ABLITERATION_MAXIMUM_CACHED_SHARDS",
                payload.get("maximum_cached_shards", 3),
            )
        )
        if self.mode == "immutable_hub_exact_overlay" and (
            len(self.hub_shards) != 120 or self.maximum_cached_shards < 1
        ):
            raise RuntimeError("Hub overlay shard closure is malformed")

    @classmethod
    def from_environment(
        cls, *, source_root: Path, device: torch.device | None = None
    ) -> VirtualAbliteration | None:
        value = os.environ.get(DESCRIPTOR_ENV)
        if not value:
            return None
        return cls(
            descriptor_path=Path(value), source_root=source_root, device=device
        )

    def get(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        kind = target_kind(name)
        if kind is None:
            return tensor
        if self.mode == "immutable_hub_exact_overlay":
            return self._get_hub_tensor(name)
        assert self.direction is not None
        projected, metrics = project_tensor(
            tensor,
            self.direction,
            kind,
            device=self.device,
        )
        if metrics["relative_projection_residual"] > self.maximum_residual:
            raise RuntimeError(f"virtual projection residual exceeds limit: {name}")
        return projected

    def _get_hub_tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map.get(name)
        expected = self.hub_shards.get(shard or "")
        if expected is None:
            raise RuntimeError(f"Hub overlay has no sealed shard for tensor: {name}")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        consumer_lock_path = self.cache_root / f".{shard}.consumer.lock"
        lock_path = self.cache_root / ".overlay.lock"
        # Hugging Face takes its own per-shard download lock. Never take the
        # global overlay lock before that lock: two workers loading adjacent
        # layers can otherwise acquire those locks in opposite order and
        # deadlock. The consumer lock keeps eviction away from this shard while
        # it is downloaded, validated, and read.
        with consumer_lock_path.open("a+b") as consumer_lock:
            fcntl.flock(consumer_lock, fcntl.LOCK_EX)
            path = Path(
                hf_hub_download(
                    repo_id=self.payload["hub_repo_id"],
                    revision=self.payload["hub_revision"],
                    filename=shard,
                    local_dir=self.cache_root,
                )
            )
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                marker_path = self.cache_root / f".{shard}.validated.json"
                stat = path.stat()
                marker = {}
                if marker_path.is_file():
                    try:
                        marker = json.loads(marker_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        marker = {}
                if not (
                    stat.st_size == expected["bytes"]
                    and marker.get("sha256") == expected["sha256"]
                    and marker.get("bytes") == stat.st_size
                    and marker.get("mtime_ns") == stat.st_mtime_ns
                ):
                    if (
                        stat.st_size != expected["bytes"]
                        or sha256(path) != expected["sha256"]
                    ):
                        raise RuntimeError(
                            f"downloaded Hub overlay shard differs: {shard}"
                        )
                    marker = {
                        "sha256": expected["sha256"],
                        "bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "accessed_ns": time.time_ns(),
                    }
                else:
                    marker["accessed_ns"] = time.time_ns()
                marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n")
                marker_path.chmod(0o644)
                with safe_open(path, framework="pt", device="cpu") as handle:
                    if name not in set(handle.keys()):
                        raise RuntimeError(f"Hub overlay tensor is missing: {name}")
                    result = handle.get_tensor(name).contiguous()
                self._evict_hub_cache(keep=shard)
                return result

    def _evict_hub_cache(self, *, keep: str) -> None:
        cached = []
        for shard, expected in self.hub_shards.items():
            path = self.cache_root / shard
            marker = self.cache_root / f".{shard}.validated.json"
            if not path.is_file():
                continue
            accessed = 0
            with suppress(OSError, ValueError, json.JSONDecodeError):
                accessed = int(json.loads(marker.read_text()).get("accessed_ns", 0))
            cached.append((accessed, shard, path, marker, expected))
        cached.sort(reverse=True)
        protected = {keep, *(row[1] for row in cached[: self.maximum_cached_shards])}
        for _, shard, path, marker, _ in cached:
            if shard in protected:
                continue
            consumer_lock_path = self.cache_root / f".{shard}.consumer.lock"
            with consumer_lock_path.open("a+b") as consumer_lock:
                try:
                    fcntl.flock(
                        consumer_lock,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError:
                    continue
                path.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
