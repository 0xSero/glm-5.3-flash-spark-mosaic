"""Prepare genuine full-length REAP token shards without padding or repetition."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Iterator

LANGUAGES = ("ar", "zh", "de", "es", "fr", "hi", "ja", "ko", "pl", "pt", "ru", "tr")
TOKENIZER_REPO = "zai-org/GLM-5.3-Flash-BF16"
TOKENIZER_REVISION = "a6c167b62691b2bac901344b65cb651a70f53e43"
WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def packed_rows(documents: Iterable[list[int]], *, count: int, length: int) -> Iterator[list[int]]:
    if min(count, length) <= 0:
        raise ValueError("count and length must be positive")
    pending: list[int] = []
    produced = 0
    for tokens in documents:
        pending.extend(tokens)
        offset = 0
        while len(pending) - offset >= length:
            yield pending[offset:offset + length]
            offset += length
            produced += 1
            if produced == count:
                return
        if offset:
            del pending[:offset]
    raise ValueError(f"corpus exhausted: {produced}/{count} full rows, {len(pending)} leftover tokens")


def prepare(output: Path, *, shard_size: int = 64) -> dict:
    import fcntl
    import numpy as np
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer

    if shard_size not in (64, 128):
        raise ValueError("shard size must be 64 or 128")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tokenizer_path = Path(hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REVISION))
        identity = {"schema": "glm53-full-token-identity-v1", "corpus": "wikipedia",
                    "sequence_count": 24000, "sequence_length": 16384, "shard_size": shard_size,
                    "source_repo": "wikimedia/wikipedia", "source_revision": WIKI_REVISION,
                    "snapshot": "20231101", "languages": list(LANGUAGES),
                    "tokenizer_repo": TOKENIZER_REPO, "tokenizer_revision": TOKENIZER_REVISION,
                    "tokenizer_sha256": sha256(tokenizer_path), "packing": "full-no-padding-no-repeat",
                    "text_policy": "article-title-newline-body-exact-text-dedup-per-language-v1"}
        identity_path = output / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("output identity mismatch")
        atomic_json(identity_path, identity)
        complete = output / "manifest.json"
        if complete.exists():
            manifest = json.loads(complete.read_text())
            if manifest.get("identity") != identity or manifest.get("state") != "COMPLETE":
                raise ValueError("invalid completed token manifest")
            for shard in manifest["shards"]:
                if sha256(output / shard["path"]) != shard["sha256"]:
                    raise ValueError("completed token shard checksum mismatch")
            return manifest
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        shards, rows, source_counts = [], [], {}
        total = 0

        def flush() -> None:
            nonlocal total, rows
            shard_id = len(shards)
            path = output / f"tokens-{shard_id:05d}.safetensors"
            temporary = path.with_suffix(".tmp")
            save_file({"input_ids": np.asarray(rows, dtype=np.int64)}, str(temporary))
            digest = sha256(temporary)
            if path.exists():
                if sha256(path) != digest:
                    raise ValueError("replayed token shard differs from sealed bytes")
                temporary.unlink()
            else:
                temporary.replace(path)
            shards.append({"shard_id": shard_id, "start": total, "end": total + len(rows),
                           "path": path.name, "sha256": digest})
            total += len(rows)
            rows = []
            progress = {"state": "PREPARING", "sequence_count": total, "target": 24000,
                        "sequence_length": 16384, "tokens": total * 16384,
                        "sealed_shards": len(shards)}
            atomic_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)

        for language in LANGUAGES:
            seen: set[str] = set()
            source_counts[language] = 0

            def documents() -> Iterator[list[int]]:
                dataset = load_dataset("wikimedia/wikipedia", f"20231101.{language}",
                                       split="train", streaming=True, revision=WIKI_REVISION)
                for article in dataset:
                    body = article.get("text", "")
                    if not isinstance(body, str) or not body.strip():
                        continue
                    text = str(article.get("title", "")) + "\n" + body
                    key = hashlib.sha256(text.encode()).hexdigest()
                    if key in seen:
                        continue
                    seen.add(key)
                    source_counts[language] += 1
                    yield tokenizer.encode(text, add_special_tokens=False).ids

            for row in packed_rows(documents(), count=2000, length=16384):
                rows.append(row)
                if len(rows) == shard_size:
                    flush()
        if rows:
            flush()
        if total != 24000:
            raise ValueError("full token budget not met")
        manifest = {"schema": "glm53-full-token-manifest-v1", "state": "COMPLETE",
                    "corpus": "wikipedia", "tokenizer_sha256": identity["tokenizer_sha256"],
                    "sequence_count": total, "sequence_length": 16384, "tokens": total * 16384,
                    "identity": identity, "source_documents": source_counts,
                    "samples_per_language": {language: 2000 for language in LANGUAGES},
                    "shards": shards}
        atomic_json(complete, manifest)
        atomic_json(output / "progress.json", {"state": "COMPLETE", "sequence_count": total,
                                               "tokens": total * 16384, "sealed_shards": len(shards)})
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=64)
    args = parser.parse_args()
    manifest = prepare(args.output.resolve(), shard_size=args.shard_size)
    print(json.dumps({"state": manifest["state"], "tokens": manifest["tokens"]}), flush=True)


if __name__ == "__main__":
    main()
