"""Prepare pinned original calibration records at natural lengths, capped at 16K.

Local inputs only. No packing, padding, text deduplication, source repetition,
downloads, or GPU imports. Wikipedia and the full-length token format are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

SOURCE_REPO = "0xSero/reap-calibration-data-v1"
SOURCE_REVISION = "115e754ab8835025e5b59df4b0f30735d8a40ce8"
SOURCE_FILE = "calibration-v1.jsonl"
SOURCE_SHA256 = "421d8c14f280012f9fdbe76349e3c0b9a5de6d11dcfaef73adcf4bbdaf5c0805"
TOKENIZER_REPO = "zai-org/GLM-5.3-Flash-BF16"
TOKENIZER_REVISION = "a6c167b62691b2bac901344b65cb651a70f53e43"
TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
LENGTH = 16384
SHARD_SIZE = 64


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sealed_json(path, value):
    """Atomically create, or verify exact replay bytes; never overwrite a conflict."""
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")
    if path.is_symlink():
        raise ValueError(f"refusing symbolic-link output: {path.name}")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"sealed output conflicts with deterministic replay: {path.name}")
        return hashlib.sha256(data).hexdigest()
    fd, temporary = tempfile.mkstemp(prefix=".record-token-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)  # exclusive publication, including competing writers
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise ValueError(f"sealed output conflict: {path.name}") from None
    finally:
        os.unlink(temporary)
    return hashlib.sha256(data).hexdigest()


def record_identity(source_sha, tokenizer_sha):
    return {"schema": "glm53-record-token-identity-v1", "corpus": "ours",
            "source_repo": SOURCE_REPO, "source_revision": SOURCE_REVISION,
            "source_file": SOURCE_FILE, "source_sha256": source_sha,
            "tokenizer_repo": TOKENIZER_REPO, "tokenizer_revision": TOKENIZER_REVISION,
            "tokenizer_sha256": tokenizer_sha, "sequence_length": LENGTH,
            "shard_size": SHARD_SIZE, "packing": "none", "padding": "none",
            "text_policy": "original-text-verbatim-preserve-records-prefix-truncate-16384-v1",
            "add_special_tokens": False, "deduplicate": False}


def prepare_records(source, tokenizer, output, identity):
    """Deterministic CPU engine; caller supplies an already verified tokenizer."""
    import fcntl
    source, output = Path(source), Path(output)
    if identity != record_identity(file_sha256(source), identity.get("tokenizer_sha256")):
        raise ValueError("source bytes or preparation policy differ from identity")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sealed_json(output / "identity.json", identity)
        tokenizer.no_padding()
        tokenizer.no_truncation()
        shards, records = [], []
        counts = Counter(original_records=0, kept_records=0, empty_records=0,
                         truncated_records=0, original_tokens=0, kept_tokens=0)
        domains, original_domains = Counter(), Counter()

        def flush():
            if not records:
                return
            sid = len(shards)
            name = f"records-{sid:05d}.json"
            lengths = [len(row["input_ids"]) for row in records]
            start = sum(item["sequence_count"] for item in shards)
            checksum = sealed_json(output / name, {"schema": "glm53-record-token-shard-v1",
                                                   "records": records})
            shards.append({"shard_id": sid, "path": name, "sha256": checksum,
                           "start": start, "end": start + len(records),
                           "sequence_count": len(records), "tokens": sum(lengths),
                           "lengths": lengths, "record_ids": [row["id"] for row in records],
                           "source_rows": [row["source_row"] for row in records]})
            records.clear()

        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                    raise ValueError(f"source line {line_number} lacks a string text field")
                source_row = counts["original_records"]
                counts["original_records"] += 1
                domain = row.get("domain")
                domain_key = domain if isinstance(domain, str) else "__missing__"
                original_domains[domain_key] += 1
                if row["text"] == "":
                    counts["empty_records"] += 1
                    continue
                ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
                if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
                    raise ValueError(f"source line {line_number} produced invalid or empty token IDs")
                original_length = len(ids)
                kept_ids = ids[:LENGTH]
                records.append({"input_ids": kept_ids, "source_row": source_row,
                                "source_line": line_number, "id": row.get("id"),
                                "domain": domain, "repo_id": row.get("repo_id"),
                                "subset": row.get("subset"),
                                "original_tokens": original_length, "truncated": original_length > LENGTH})
                counts.update(kept_records=1, original_tokens=original_length,
                              kept_tokens=len(kept_ids), truncated_records=int(original_length > LENGTH))
                domains[domain_key] += 1
                if len(records) == SHARD_SIZE:
                    flush()
        flush()
        if counts["kept_records"] == 0:
            raise ValueError("source has no usable original records")
        # Re-read the source checksum to reject concurrent source modification.
        if file_sha256(source) != identity["source_sha256"]:
            raise ValueError("source changed during preparation")
        manifest = {"schema": "glm53-record-token-manifest-v1", "state": "COMPLETE",
                    "corpus": "ours", "identity": identity, "sequence_length": LENGTH,
                    "sequence_length_policy": "natural-up-to-cap", "sequence_count": counts["kept_records"],
                    "tokens": counts["kept_tokens"], "counts": dict(counts),
                    "domain_counts": dict(domains), "original_domain_counts": dict(original_domains),
                    "shards": shards, "full_suite_pass": False}
        sealed_json(output / "manifest.json", manifest)
        return manifest


def prepare(source, tokenizer_path, output):
    source, tokenizer_path = Path(source), Path(tokenizer_path)
    if source.name != SOURCE_FILE or file_sha256(source) != SOURCE_SHA256:
        raise ValueError("requires the exact pinned calibration-v1.jsonl, not filtered_v2 or another source")
    if file_sha256(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("requires the exact original pinned tokenizer JSON")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return prepare_records(source, tokenizer, output, record_identity(SOURCE_SHA256, TOKENIZER_SHA256))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.source, args.tokenizer, args.output)
    print(json.dumps({key: manifest[key] for key in ("state", "sequence_count", "tokens", "counts")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
