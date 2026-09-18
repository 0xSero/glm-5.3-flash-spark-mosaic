# M288-12L mosaic — model card

The canonical, rendered version of this card lives with the weights:
**https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-M288-Mosaic-12L**  
Revision this repo's results were measured from: `2642851741fc833764e77d03039117be559dc83e`.
Measured numbers with receipt paths: [`MEASURED.md`](MEASURED.md).
Artifact provenance and per-shard hashes: [`PROVENANCE.json`](PROVENANCE.json).

---

# GLM-5.3-Flash EXL3 — M288 Mosaic 12L

Mixed-precision EXL3 artifact of GLM-5.3-Flash for a single NVIDIA DGX Spark (GB10, arm64, 128 GB
unified memory). 12 of the 45 routed-expert MoE layers are upgraded from 3.05 bpw to higher precision
byte-for-byte; everything else — tokenizer, dense/attention dressing, router rows, and the MTP layer 45
— is carried over untouched from the 2.05 bpw base.

This is a **quality** artifact, not a speed one. Read the MTP limitation below before deploying.

## What this is exactly

| | |
|---|---|
| Base | [`turboderp/GLM-5.3-Flash-exl3`](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) @ `51058cd551c7e570d87bd32a4adee720edce2349` (2.05 bpw) |
| Upgraded layers | **3, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 44** — all 288 experts of each, at 3.05 bpw |
| Untouched | all remaining layers, all dressing, **MTP layer 45 verbatim** |
| Codebook | `mul1` (inherited from the base) |
| Size on disk | 96,105,137,024 B (96.1 GB) — 12 shards |
| Delta over base | 10,871,635,968 B planned; measured delta 10,871,646,674 B (+10,706 B header drift, 5 shards rewritten) |
| Overall rate | ≈2.31 bpw (derived: 96.105/85.233 × 2.05) |
| Experts / top-k | 288 routed, 1 shared, top-8 |
| License | MIT, © 2026 Z.AI Co., Ltd (carried from the base) |

Every upgraded expert tensor is byte-equal to its 3.05 bpw source, and every kept tensor is byte-equal to
the 2.05 bpw base; the checks are per-tensor, recorded in `PROVENANCE.json`. The packer's whole-file
delta-equality assertion reports a FAIL on this artifact — that check compares total bytes against a
prediction and trips on the +10,706 B header drift above. **It is a false positive; the per-tensor
checks are the load-bearing ones.** All 288 experts are present in every layer.

## Measured quality

Full G4 panel, 65,504 positions across 32 pre-registered rows, same harness / same image / same teacher
rows for both sides, teacher reproduction exact on both. a0 was re-measured the same day rather than
taken from a leaderboard.

| Metric | base 2.05 bpw | **M288-12L mosaic** | Δ |
|---|---:|---:|---|
| top-1 agreement | 0.78902 | **0.80842** | **+1.94 pp** |
| KL lower-bound mean | 0.38378 | **0.32522** | **−15.3 %** |
| KL p50 / p95 | 0.1044 / 1.797 | **0.0811 / 1.540** | better |
| KL p99 / max | 3.849 / 13.73 | 3.424 / 14.23 | ≈ |
| candidate PPL | 4.28692 | **4.03563** | **−5.9 %** |

**The mosaic wins 32/32 rows on top-1, 32/32 on KL and 32/32 on NLL — no row regresses.**

Task benchmarks (quick MMLU, 57 subjects × limit 20, seed 1234; GPQA diamond zeroshot MC, 198 docs):

| Benchmark | base 2.05 bpw | **M288-12L mosaic** |
|---|---:|---:|
| MMLU (quick) | 0.8342 ± 0.0105 | **0.8360 ± 0.0104** |
| GPQA diamond (MC) | 0.4343 ± 0.0353 | **0.4394 ± 0.0354** |

Within this program, this is the only artifact that beats the base on the panel *and* regresses on
neither task benchmark — the e216 and f216 mosaics sit 4–6 pp below the base on both.

## Measured serving behaviour and the MTP limitation

Served on one DGX Spark with SGLang (`glm53-flash-sglang-exl3-plain:serve4`, sha256 `c65c840f1908…`):
262,144-token context, vision on, fp8 KV, `mem-fraction-static 0.95` giving **KV 595,200 tokens**,
`max-running-requests 1`, no CUDA graphs.

**Decode: 9–10 tok/s.** Roughly 11.2–11.5 tok/s with decode graphs on the 10-layer sibling.

**There is no MTP on this artifact.** Both MTP-capable runtimes in this programme reject the `mul1`
codebook at config validation, before any weight is read:

```
ValueError: this overlay only implements codebook=mcg; got 'mul1'
```

So this runs at roughly half the decode rate of the `mcg`-codebook staging line (18.7–19.2 tok/s), which
is faster but scores below the base on quality. Unlocking MTP here means implementing the `mul1` decode
path in the MTP overlay, or re-quantizing to `mcg`. If you need speed over fidelity today, this is the
wrong artifact.

## Usage

Serving requires the SGLang EXL3 runtime and its overlay — plain `transformers` will not execute these
weights. See the recipe repository for the launcher, the census preflight that refuses to start on a
layout mismatch, and the exact flag set:

- recipe + docs: https://github.com/0xSero/glm-5.3-flash-spark-mosaic
- sibling 1×-Spark EXL3 + MTP work: https://github.com/0xSero/GLM-5.3-Flash-EXL3-2x-DGX-Sparks

## Provenance and verification

`PROVENANCE.json` records the base pin, the layer plan (`plan_sha256
7938939c3a5b1cbaaa120ff1fdb51e7d8171f1f46c18630e59727812e969204e`), per-shard sizes and output hashes,
and the packer's check results. `sha256-manifest.txt` carries the local file hashes taken before upload,
so any download can be verified byte-for-byte.

Built and measured by the GLM-5.3-Flash single-Spark release program, 2026-09-16/17 UTC.