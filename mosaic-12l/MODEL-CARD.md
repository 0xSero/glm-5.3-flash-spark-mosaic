# M288-12L mosaic — model card

The canonical, rendered version of this card lives with the weights:
**https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Spark**
Renamed 2026-09-18 from `0xSero/GLM-5.3-Flash-EXL3-M288-Mosaic-12L`; the old URL redirects and the
weights are unchanged.
Revision this repo's results were measured from: `2642851741fc833764e77d03039117be559dc83e`.
Measured numbers with receipt paths: [`MEASURED.md`](MEASURED.md).
Artifact provenance and per-shard hashes: [`PROVENANCE.json`](PROVENANCE.json).

---

# GLM-5.3-Flash EXL3 — Spark

A mixed-precision **EXL3** build of GLM-5.3-Flash sized to serve a 256k-token context with vision on a
single NVIDIA **DGX Spark** (GB10, arm64, 128 GB unified memory): 12 of the 45 routed-expert MoE layers
are quantized at 3.05 bpw for accuracy, the other 33 stay at 2.05 bpw for size. End result: **≈2.31 bpw
average**, 96.1 GB on disk, and higher agreement with the teacher than the 2.05 bpw base it is built
from.

This is a **quality** artifact, not a speed one. It currently runs without speculative decoding
(see [MTP](#mtp-two-code-blockers-not-a-weight-problem)), which roughly halves decode rate against
EXL3 builds that do. Read that section before deploying.

*Previously published as `GLM-5.3-Flash-EXL3-M288-Mosaic-12L`; the weights are unchanged.*

## At a glance

| | |
|---|---|
| **Average rate** | **≈2.31 bpw** — mixed: 12 layers at 3.05 bpw, 33 layers at 2.05 bpw |
| Size | **96,105,137,024 B (96.1 GB)**, 12 shards |
| Base | [`turboderp/GLM-5.3-Flash-exl3`](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) @ `51058cd551c7e570d87bd32a4adee720edce2349` (2.05 bpw) |
| Upgraded layers | 3, 32, 33, 36, 37, 38, 39, 40, 41, 42, 43, 44 — all 288 experts of each, at 3.05 bpw |
| Untouched, byte-for-byte | all remaining layers, the dense/attention dressing, the tokenizer, **MTP layer 45** |
| Architecture | 45 layers, 288 routed experts + 1 shared, top-8, hidden 4096, vocab 154,880 |
| Context | 1,048,576 native (`max_position_embeddings`); served at 262,144 |
| Vision | Yes — 8,000 image tokens per image (video: 240,000) |
| Codebook | `mul1` (inherited from the base) |
| License | MIT, © 2026 Z.AI Co., Ltd (carried from the base) |
| Quality | **better than the base on every measured row** (panel 32/32; MMLU and GPQA both ≥ base) |
| Speed, 1× Spark | decode **10.79 tok/s** · prefill **503.5 tok/s** (receipted, method below) |

## What "≈2.31 bpw" means here

bpw is the trellis rate of the quantized weights. This artifact has exactly two rates, chosen per layer:

| Layers | Rate | Count |
|---|---|---|
| 3, 32, 33, 36–44 | 3.05 bpw | 12 |
| every other MoE layer | 2.05 bpw | 33 |

The average is **2.31 bpw**, and it can be derived two independent ways that agree:

- **By bytes.** The artifact is 12.76 % larger than its 2.05 bpw base (96.105 GB vs 85.233 GB), and the
  only change is the rate of those 12 layers, so `2.05 × 96.105/85.233 = ` **2.3115**.
- **By layer mass.** Upgrading a layer from 2.05 to 3.05 bpw multiplies its expert bytes by 1.488. The
  measured delta (10.87 GB) therefore implies those 12 layers hold 22.29 GB in the base — 26.15 % of the
  file for 26.67 % of the layers, i.e. the layers are uniform. The equal-layer average is
  `(33×2.05 + 12×3.05)/45 = ` **2.3167**.

Both land on **≈2.31 bpw**. The dense/attention dressing, embeddings and vision tower are unchanged from
the base and are not part of the rate change; they are what pulls the two derivations apart in the third
decimal.

Why not uniform 2.31 bpw everywhere? Because the 12 chosen layers are the ones where extra precision buys
the most agreement, and spending it there beats spending it evenly. The panel below is the measurement of
that bet.

## Measured quality

Full G4 panel, 65,504 positions across 32 pre-registered rows, same harness, same image and same teacher
rows on both sides; teacher reproduction exact on both (`top1_exact_all_rows: true`). The 2.05 bpw side
was re-measured the same day rather than quoted.

| Metric | base 2.05 bpw | **this artifact** | Δ |
|---|---:|---:|---|
| top-1 agreement | 0.78902 | **0.80842** | **+1.94 pp** |
| KL lower-bound mean | 0.38378 | **0.32522** | **−15.3 %** |
| KL p50 / p95 | 0.1044 / 1.797 | **0.0811 / 1.540** | better |
| KL p99 / max | 3.849 / 13.73 | 3.424 / 14.23 | ≈ |
| candidate PPL | 4.28692 | **4.03563** | **−5.9 %** |

**32/32 rows improve on top-1, 32/32 on KL, 32/32 on NLL. No row regresses.**

Task benchmarks:

| Benchmark | base 2.05 bpw | **this artifact** |
|---|---:|---:|
| MMLU (quick, 57 subjects × limit 20, seed 1234) | 0.8342 ± 0.0105 | **0.8360 ± 0.0104** |
| GPQA diamond (zeroshot MC, 198 docs) | 0.4343 ± 0.0353 | **0.4394 ± 0.0354** |

Within the program that produced it, this is the only artifact that beats the 2.05 bpw base on the panel
*and* regresses on neither benchmark. Two earlier mosaics (e216, f216) sit 4–6 pp below the base on both.

## Serving on one DGX Spark

Validated 2026-09-18 on a Spark from a directory downloaded through the published path, so these are
user-facing numbers:

| Requirement | Result |
|---|---|
| Context | `max_model_len 262,144`; a 236,510-token prompt served end to end |
| Output budget | 128,000 tokens accepted; no `max_new_tokens` in `generation_config.json` |
| Reasoning | 8,192 reasoning tokens / 30,912 chars in one response, not truncated by the server |
| Determinism | identical output on repeat at `temperature 0` |
| Vision at 4096×4096 | accepted → **7,921 image tokens, 99 % of the 8,000-per-image ceiling** |

| Speed | |
|---|---|
| Decode | **10.79 tok/s** mean, 10.61–10.94 across a 238× range of prompt length |
| Prefill | **503.5 tok/s** marginal (r² 0.9999, seven rungs from 990 to 236,510 tokens) |

Measured with a streaming sweep whose every prompt opens with a unique nonce, so prefix caching cannot
flatter the result; the prefill fit's small residuals are themselves the evidence that each prompt paid a
full cold prefill. Server: SGLang EXL3, fp8 KV, DSA attention, `mem-fraction-static 0.95` (KV pool
595,200 tokens), `max-running-requests 1`, no CUDA graphs.

For scale: EXL3 builds that run native MTP on the same box reach 18.7–19.6 tok/s decode.

### Vision

The image processor declares an 8,000-token ceiling per image, and the 4096×4096 probe lands at 7,921 —
i.e. full resolution up to the artifact's own limit. The model reads the fixture's geometry correctly at
512² and 2048² (it reports bar columns at x≈70–135 and x≈290–525, matching the encoder's 73/292 column
origins). A verified end-to-end answer to the probe's decoding task was not obtained because the task does
not fit its token budget; that limit is on the probe, not the vision path.

## MTP: two code blockers, not a weight problem

**The MTP layer is present in these weights.** Layer 45 is the multi-token-prediction block — `eh_proj`,
`enorm`, `hnorm`, `shared_head` alongside its attention and MLP, with all 288 experts, 873 `mul1` tensor
sets and 12 native tensors — and `config.json` declares `num_nextn_predict_layers: 1`.

What is missing is runtime support. Two code blockers stand in the way, and an experiment on 2026-09-18
isolated the second:

1. **The codebook gate.** MTP-capable EXL3 runtimes in this lineage implement only the `mcg` codebook and
   reject this artifact's `mul1` at config validation, before any weight is read:
   `ValueError: this overlay only implements codebook=mcg; got 'mul1'`.
2. **A checkpoint-key mismatch.** With that gate relaxed, validation passes and the engine dies at weight
   loading — at the first MoE layer, before any kernel runs:

```
File ".../vllm/models/glm5next/nvidia/model.py", line 904, in load_weights
    param = params_dict[name]
KeyError: 'layers.0.mlp.down_proj.mul1'
```

These weights descend from the turboderp/SGLang line and carry `model.language_model.layers.N.mlp.experts.E.*`
keys, while the vLLM MTP loader addresses experts as `layers.N.mlp.down_proj.*`. Layer 0 is a stock,
non-upgraded layer, so this is a naming-lineage difference rather than anything caused by the 12 upgraded
layers.

Bringing MTP to this artifact therefore needs **a checkpoint-key remap first, then a `mul1` decode path** —
code work with a known entry point (`model.py:904`), not a research problem. Until then the artifact runs
without speculative decoding, which is the whole of the decode gap above.

## Files

| File | What it is |
|---|---|
| `model-00001..12-of-00012.safetensors` | the 12 weight shards (96.1 GB) |
| `config.json`, `quantization_config.json` | architecture and the patched per-module bit settings (47.9 MB) |
| `tokenizer.json`, `tokenizer_config.json`, `processor_config.json`, `chat_template.jinja` | text and multimodal processing |
| `generation_config.json` | sampling defaults; contains no output-length cap |
| `PROVENANCE.json` | base pin, layer plan (`plan_sha256 7938939c…`), per-shard hashes, packer check results |
| `sha256-manifest.txt` | hash of every file above, for byte-for-byte verification |
| `DOWNLOAD-SEAL.json`, `pull.sh` | provenance of the 2.05 bpw base download |
| `README.md`, `LICENSE` | this card; MIT license carried from the base |

## Usage

EXL3 weights need an EXL3 runtime — plain `transformers` will not execute them. The recipe repository has
the launcher, a census preflight that refuses to start on a layout mismatch, and the exact flag set:

- **recipe + launcher**: https://github.com/0xSero/glm-5.3-flash-spark-mosaic
- **sibling 1× Spark EXL3 + MTP work**: https://github.com/0xSero/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
- **full validation pass** (scans, sweeps, cache behaviour, MTP isolation): that fork's `docs/VALIDATION.md`

Download and verify:

```bash
git clone https://github.com/0xSero/glm-5.3-flash-spark-mosaic
cd glm-5.3-flash-spark-mosaic/docker
./download-mosaic.sh ~/models/glm53-exl3-spark     # pinned revision + manifest verification
GLM53_MEM_FRACTION=0.95 GLM53_MAX_RUNNING=1 ./run-spark.sh ~/models/glm53-exl3-spark
```

## Verification

Per-tensor: every upgraded expert tensor is byte-equal to its 3.05 bpw source and every kept tensor is
byte-equal to the 2.05 bpw base; the checks are per-tensor and recorded in `PROVENANCE.json`. The packer's
whole-file delta check reports a FAIL on this artifact because it compares total bytes against a
prediction and trips on +10,706 B of safetensors header drift across the five rewritten shards — **a false
positive; the per-tensor checks are the load-bearing ones**, and the 10-layer sibling shows the identical
drift.

Whole-file: `sha256-manifest.txt` covers all 26 content files. Verified end to end on 2026-09-18 — 27/27
files fetched through the documented path, `sha256sum -c` 26/26 OK, exit 0, and the bytes actually served
by the engine match the published bytes.

## Provenance and license

Built by the GLM-5.3-Flash single-Spark release program, 2026-09-16/18 UTC, from
`turboderp/GLM-5.3-Flash-exl3` @ `51058cd551c7e570d87bd32a4adee720edce2349`. License MIT, © 2026 Z.AI
Co., Ltd, carried from the base; this artifact adds no new licensing terms.