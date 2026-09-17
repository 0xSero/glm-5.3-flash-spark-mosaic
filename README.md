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

## Mosaic artifacts

The M288 mosaics are byte-copy combinations of the sealed 2.05bpw and 3.05bpw
quants (no new quantization error): `mosaic-gatea/mosaic_pack.py`,
plans `mosaic-gatea/mosaic-12l-plan.json` (+ pack receipts with per-tensor
SHAs). The mosaic weights themselves are large binaries and are **not** in
this repository; rebuild them from the two public HF quants with the pack
scripts, or apply the plan directly.

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

Code and docs: MIT. Model weights: see the upstream HF repos
(`zai-org/GLM-5.3-Flash`, MIT; `turboderp/GLM-5.3-Flash-exl3`).
