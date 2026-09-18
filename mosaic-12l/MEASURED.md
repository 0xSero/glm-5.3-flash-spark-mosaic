# M288-12L mosaic — measured results

Everything here was measured, not predicted. Each number carries the receipt path it came from; raw
JSON lives in `mosaic-gatea/` (copies) and on the build hosts (`557f:/home/valentine/mosaic-gatea/`,
`2384:/home/sero/w2port/out/`).

Artifact: **M288-12L-GateA2**, 96,105,137,024 B, plan_sha256
`7938939c3a5b1cbaaa120ff1fdb51e7d8171f1f46c18630e59727812e969204e`, published at
`0xSero/GLM-5.3-Flash-EXL3-M288-Mosaic-12L` @ `2642851741fc833764e77d03039117be559dc83e`.

## Quality — full G4 panel, 65,504 positions, 32 rows

Both sides measured the same day, same harness, same image, same teacher rows; teacher reproduction
exact on both (`top1_exact_all_rows: true`). The 2.05 bpw baseline was re-measured for this comparison
rather than quoted from a leaderboard.

| Metric | a0 (2.05 bpw) | M288-12L | Δ |
|---|---:|---:|---|
| top-1 agreement | 0.78902 | **0.80842** | +1.94 pp |
| KL lower-bound mean | 0.38378 | **0.32522** | −15.3 % |
| KL p50 / p95 | 0.1044 / 1.797 | **0.0811 / 1.540** | better |
| KL p99 / max | 3.849 / 13.73 | 3.424 / 14.23 | ≈ |
| candidate PPL | 4.28692 | **4.03563** | −5.9 % |

Per-row: the mosaic wins **32/32 rows** on top-1, **32/32** on KL and **32/32** on NLL. No row regresses.

Receipts: `mosaic12l-full-panel.json`, `mosaic12l-full-summary.json`, `a0-full-panel.json`,
`a0-full-summary.json` — panel fixture manifest `46255621…`, teacher manifest `a1604015…`, identical on
both sides.

## Task benchmarks

| Benchmark | a0 | M288-12L | Bar |
|---|---:|---:|---|
| MMLU quick (57 subjects × limit 20, seed 1234) | 0.8342 ± 0.0105 | **0.8360 ± 0.0104** | > 0.8342 ✓ |
| GPQA diamond zeroshot (MC, 198 docs) | 0.4343 ± 0.0353 | **0.4394 ± 0.0354** | ≥ 0.4343 ✓ |
| samples (20) | 15/20 | 14/20 | ≥ 16 ✗ |

The sample bar is the one miss and applies to both sides equally (15/20 vs 14/20).

Receipts: `benchmarks/lm-eval-bench/results/mosaic12l-quick/bench-mosaic12l-quick/results_2026-09-16T20-09-27.917455.json`,
`results/mosaic12l-gpqa-mc/bench-mosaic12l-gpqa-mc/results_2026-09-16T20-24-02.421014.json`, a0 counterparts
in `results/a0-quick/` and `results/a0-gpqa/`, driver log `results/status.txt`
(`2026-09-16T21:44:55Z start` → `2026-09-17T00:09:37Z exit=0`; GPQA `00:13:08Z` → `00:24:02Z exit=0`).

This is the only artifact in the program that beats a0 on the panel *and* regresses on neither task
benchmark. The e216 and f216 mosaics sit 4–6 pp below a0 on both (`mosaic-gatea/GATE-A-RESULTS.md`).

## Serving

Served on spark-557f as `glm53-mosaic-12lh`, image `glm53-flash-sglang-exl3-plain:serve4`
(`c65c840f1908`), `--mem-fraction-static 0.95`, KV 595,200 tokens, `max-running-requests 1`,
262,144 context, vision on, fp8 KV, DSA attention, no CUDA graphs.

**Decode 9–10 tok/s.** The 10-layer sibling reaches 11.2–11.5 tok/s with decode graphs enabled.

There is **no MTP** on this artifact — see `mosaic-gatea/MTP-CODEBOOK-BLOCKER.md`. Both MTP-capable
runtimes reject the `mul1` codebook at config validation:

```
ValueError: this overlay only implements codebook=mcg; got 'mul1'
```

That gate is the first blocker, not the only one. The MTP experiment of 2026-09-18 relaxed it and found
the next one (`mosaic-gatea/M1-FINDINGS.md`): with `mul1` accepted, the engine loads weights and dies at
the first MoE layer with

```
File ".../vllm/models/glm5next/nvidia/model.py", line 904, in load_weights
    param = params_dict[name]
KeyError: 'layers.0.mlp.down_proj.mul1'
```

because this artifact descends from the turboderp/SGLang line and its keys are
`model.language_model.layers.N.mlp.experts.E.*`, while the vLLM MTP loader addresses experts as
`layers.N.mlp.down_proj.*`. Layer 0 is a stock layer, so this is a naming-lineage mismatch, not a
consequence of the 12 upgraded layers. MTP here needs a checkpoint-key remap and then the `mul1` decode
path — code work with a known entry point, not a research problem, and more than the gate.

For reference, the `mcg`-codebook staging line the 1×-Spark EXL3/MTP work targets runs at 18.7–19.2
tok/s but scores below a0 on quality. Trade fidelity for speed, not both.

## Serving validation — 2026-09-18

Measured on the published artifact, served from a directory that was itself fetched through
`./download-mosaic.sh` and hash-checked, so these are stranger-facing numbers.

| Check | Result |
|---|---|
| Download via the published path | 27/27 files, 1 h 23 m |
| `sha256sum -c sha256-manifest.txt` | 26/26 OK, exit 0 |
| Served bytes vs published | `served_bytes_match_published: true` |
| Context served | `max_model_len 262144`; longest request exercised 236,510 prompt tokens |
| Output cap | 128,000-token budget accepted; no artifact-level `max_new_tokens` in `generation_config` |
| Reasoning | 8,192 reasoning tokens / 30,912 chars in one response, server did not truncate |
| Determinism | identical output on repeat at `temperature 0` |
| Vision | 4096×4096 accepted → 7,921 image tokens, 99 % of the declared 8,000/image ceiling |
| **Prefill** | **503.5 tok/s marginal** (r² 0.9999 over seven rungs, 990 → 236,510 tokens) |
| **Decode** | **10.79 tok/s mean**, 10.61–10.94 across a 238× range of prompt length |

The prefill figure comes from a streaming sweep with a unique nonce opening each prompt. An earlier
ladder used `FILLER × N` prompts, which made each rung a prefix of the next; radix caching then served
the shared prefix and the rungs measured only their deltas (its 260 k rung came out *faster* than its
131 k rung). Both sweeps are kept — the superseded one as the record of the method error, with the
server's own `#cached-token` accounting as the explanation. Full receipts and method notes:
`GLM-5.3-Flash-EXL3-2x-DGX-Sparks` → `docs/VALIDATION.md`.

## Reproduce

```bash
# 1) weights (~96 GB, pinned revision, verified against the repo's own manifest)
git clone https://github.com/0xSero/glm-5.3-flash-spark-mosaic
cd glm-5.3-flash-spark-mosaic/docker
./download-mosaic.sh ~/models/mosaic-12l

# 2) census + serve (the census refuses to start on a layout mismatch)
GLM53_MEM_FRACTION=0.95 GLM53_MAX_RUNNING=1 \
  ./run-spark.sh ~/models/mosaic-12l
```

The a0 baseline for comparison is the same recipe on the 2.05 bpw weights:
`./download-model.sh ~/models/GLM-5.3-Flash-exl3-2.05bpw && ./run-spark.sh ~/models/GLM-5.3-Flash-exl3-2.05bpw`.

## Verification of the artifact itself

- All upgraded expert tensors byte-equal to the 3.05 bpw source; all kept tensors byte-equal to the
  2.05 bpw base (`mosaic-12l-pack-receipt.json`, `shas-*-12l.txt`).
- `12/12` output shards re-hashed on the build host and matched the published manifest before upload.
- The packer's whole-file delta-equality check reports **FAIL** on this artifact — it compares total
  bytes to a prediction and trips on `+10,706 B` of header drift across the 5 rewritten shards. The
  per-tensor checks are the load-bearing ones; `n_failures: 0`.
- The 10-layer sibling shows the identical `+10,706 B` drift, confirming it is systematic and not a
  defect: `mosaic_pack2.py`'s exact-delta check produces false FAILs.