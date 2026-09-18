# GLM-5.3-Flash on one DGX Spark — serving recipe

Public, receipted recipe for serving GLM-5.3-Flash EXL3 quants on a single
DGX Spark (GB10, arm64, 128 GB unified memory): the a0 2.05bpw baseline and the
M288 mixed-precision mosaics built from it.

Everything here was produced by the GLM-5.3-Flash single-Spark release program.
Every number has a receipt path; raw JSON receipts live beside the docs that
cite them.

## Quickstart (one GPU, one node)

```bash
git clone https://github.com/0xSero/glm-5.3-flash-spark-mosaic
cd glm-5.3-flash-spark-mosaic/docker

# 1) weights (~85 GB, public, pinned revision, sha manifest written on completion)
./download-model.sh ~/models/GLM-5.3-Flash-exl3-2.05bpw

# 2) census + serve (the image itself is public on GHCR, arm64)
./run-spark.sh ~/models/GLM-5.3-Flash-exl3-2.05bpw
# -> HEALTH_OK ... serving on http://127.0.0.1:8000/v1  (model: glm-5.3-flash)

curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": "Say OK."}]
}'
```

`run-spark.sh` censuses the downloaded tensors first and refuses to start the
server on a layout mismatch, then launches the exact panel-proven flag set
(262,144 context, fp8 KV cache, vision on, `SGLANG_EXL3_MAX_BATCH_TOKENS=256`
+ chunked-prefill 256 — the pairing that avoids the scheduler's
`EXL3 batch exceeds preplanned N tokens` crash; see
`benchmarks/lm-eval-bench/relaunch-a0-bench5.sh` for the original receipted
command).

Serving config preserved per program rules: **262,144 context and vision on**.

### The mosaic — best quality, published weights

The M288-12L mosaic is the strongest artifact in this programme, and its weights are published, so
there is nothing to build:

```bash
git clone https://github.com/0xSero/glm-5.3-flash-spark-mosaic
cd glm-5.3-flash-spark-mosaic/docker

# 1) weights (~96 GB, pinned revision, verified against the model repo's own sha256 manifest)
./download-mosaic.sh ~/models/mosaic-12l

# 2) census + serve
GLM53_MEM_FRACTION=0.95 GLM53_MAX_RUNNING=1 ./run-spark.sh ~/models/mosaic-12l
```

Weights: [`0xSero/GLM-5.3-Flash-EXL3-Spark`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Spark) @
`2642851741fc833764e77d03039117be559dc83e`. Those two variables are the mosaic's **measured** values,
not defaults: it is ~11 GB larger than a0, and at `mem-fraction-static 0.90` the 262,144-token KV gate
does not fit. Expect **9–10 tok/s decode** — this artifact has no MTP, for the reason in
[`mosaic-gatea/MTP-CODEBOOK-BLOCKER.md`](mosaic-gatea/MTP-CODEBOOK-BLOCKER.md). Model card:
[`mosaic-12l/MODEL-CARD.md`](mosaic-12l/MODEL-CARD.md); measurements and receipt paths:
[`mosaic-12l/MEASURED.md`](mosaic-12l/MEASURED.md).

## The two artifact lineages (measured boundary)

| lineage | codebook | runtime | MTP | measured decode (262k ctx class) |
|---|---|---|---|---|
| turboderp EXL3: a0 (2.05bpw), **M288-12L/10L mosaic** | `mul1` | SGLang `exl3-plain` (this recipe) | no — overlay gate: `this overlay only implements codebook=mcg` | 9–10 tok/s; **11.2–11.5** with decode graphs (10L) |
| staging K2 (mcg) | `mcg` | vLLM B12x, depth-2 native MTP, B12X attn, `fp8_ds_mla`, FULL_DECODE_ONLY graphs | yes, 97.7–100% acceptance | **18.7–19.2 tok/s**, flat 1k→260k input |

Details: [`mosaic-gatea/MTP-CODEBOOK-BLOCKER.md`](mosaic-gatea/MTP-CODEBOOK-BLOCKER.md),
[`mosaic-gatea/RESULTS-CONFIRMING-RUN.md`](mosaic-gatea/RESULTS-CONFIRMING-RUN.md).

## Results

**Quality — M288-12L mosaic vs a0** (full 65,504-position G4 panel, 32/32 rows,
`mosaic-gatea/RESULTS-CONFIRMING-RUN.md`):

| metric | a0 (2.05bpw) | M288-12L mosaic |
|---|---:|---:|
| top-1 agreement | 0.78902 | **0.80842** |
| KL lower-bound mean | 0.38378 | **0.32522** |
| candidate PPL | 4.28692 | **4.03563** |
| quick MMLU (57 subj × 20) | 0.8342 ± 0.0105 | **0.8360 ± 0.0104** |
| GPQA diamond MC | 0.4343 ± 0.0353 | **0.4394 ± 0.0354** |

**Speed — native MTP, full context-window sweep** (non-greedy, cold prefix
cache, `mosaic-gatea/receipts-557f/`): decode 18.70–19.23 tok/s from 1,023 to
260,095 input tokens (2.7% spread), prefill ~452 tok/s, TTFT 2.6 s @1k →
575.7 s @260k, MTP acceptance 0.977–1.000, 14/14 cells sustained.

## Mosaic artifacts and weights

The M288 mosaics are byte-copy combinations of the sealed 2.05bpw and 3.05bpw
quants (no new quantization error): `mosaic-gatea/mosaic_pack.py`,
plans `mosaic-gatea/mosaic-12l-plan.json` (+ pack receipts with per-tensor
SHAs).

**The 12L mosaic weights are published** — 96.1 GB, 12 shards, pinned revision,
each file verifiable against the manifest the model repo ships:

| | |
|---|---|
| weights | https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Spark |
| revision | `2642851741fc833764e77d03039117be559dc83e` |
| download | `docker/download-mosaic.sh` (pinned + verified, fails closed) |
| provenance | [`mosaic-12l/PROVENANCE.json`](mosaic-12l/PROVENANCE.json) — base pin, layer plan, per-shard hashes |
| measured | [`mosaic-12l/MEASURED.md`](mosaic-12l/MEASURED.md) |

Upgraded layers are 3, 32, 33 and 36–44 — all 288 experts of each, at 3.05bpw; everything else,
including MTP layer 45, is carried over from the 2.05bpw base byte-for-byte. The 10L and 20L siblings
exist on the build hosts but are not published; rebuild either from the two public quants with the pack
scripts and the corresponding plan file.

Known negative result kept on purpose: per-expert (not per-layer) bit
allocation fails at load — the overlay fuses all experts of a layer into one
stacked tensor (`mosaic-gatea/GATE-A-RESULTS.md`).

## Provenance

- `DATA_PROVENANCE.md` — data and artifact lineage.
- `REPO-EXCLUDED-FILES.txt` — every large binary receipt that exists on the
  build hosts but is deliberately not in git.
- Secrets are not in this repository. Internal infrastructure identifiers in
  scripts/receipts have been replaced with `.internal` placeholders; the
  placeholder substitutions do not affect any measurement.

## License

Code and docs: MIT. Model weights: MIT, © 2026 Z.AI Co., Ltd, inherited through
[`turboderp/GLM-5.3-Flash-exl3`](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3); the mosaic's own
`LICENSE` travels with the weights in
[`0xSero/GLM-5.3-Flash-EXL3-Spark`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Spark).
Upstream: `zai-org/GLM-5.3-Flash`.
