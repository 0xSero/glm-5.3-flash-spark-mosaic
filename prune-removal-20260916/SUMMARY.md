# Prune-artifact removal — 2026-09-16 (user order: "remove the prunes off the sparks")

5 pruned artifact dirs removed; **482,966,693,370 B (≈483 GB)** of artifact bytes; df-confirmed
482,967,396,352 B freed. 557f: 99% → 89% full (39 GB → 427 GB free). 2822: 55% → 53%.

| Host | Path | Bytes | Files | rm_rc | gone |
|---|---|---:|---:|---:|---|
| 2822 | /home/sero/models/glm53-3p05-pruned-f216 | 96,593,390,114 | 31 | 0 | yes |
| 557f | /home/valentine/models/glm53-3p05-pruned-e216 | 96,593,388,003 | 31 | 0 | yes |
| 557f | /home/valentine/models/glm53-3p05-pruned-f216 | 96,593,390,114 | 31 | 0 | yes |
| 557f | /home/valentine/models/glm53-3p05-pruned-s216 | 96,593,256,540 | 31 | 0 | yes |
| 557f | /home/valentine/models/glm53-3p05-pruned-t216 | 96,593,268,599 | 31 | 0 | yes |

Files here:
- `prune-receipt.sh` — the receipt tool (shipped to both hosts, sha-able, read-only unless you rm).
- `prune-removal-{2822,557f}-pre.json` — pre-deletion receipts: path, du -sb bytes, file count, sha256 of
  every *.json identity file inside each artifact.
- `prune-removal-{2822,557f}-listing.txt` — full per-file size listings (32 and 128 lines).
- `post-deletion-receipt.json` — df before/after per host, rm_rc, gone flags, preserved list, rebuild path.

Preserved and untouched: sealed turboderp base (3.05bpw, both hosts), 2.05bpw quant copy (557f), Q4
observation model (2822), a0 release artifact (2384), all plans, tooling, sealed observations,
calibration corpora, teacher rows, and every receipt — plus 557f's exllamav3 v1.4.9 quantizer source.
Rebuild path: sealed base + frozen plan (sha in receipts) + packer on 2822:/home/sero/work/f216-build/
(pack measured 80.3 s).
