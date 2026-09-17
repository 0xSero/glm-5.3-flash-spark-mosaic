# MOSAIC-288 — plain-language plan (outcomes + dependencies)

## STATUS after the confirming run + MTP serving (2026-09-17) — see `mosaic-gatea/RESULTS-CONFIRMING-RUN.md`

- **The mosaic wins the release comparison.** Full G4 panel, 65,504 positions, same-day a0 re-measured
  with the same harness and teacher rows (`teacher_reproduction ok`, top-1 exact): **top-1 0.80842 vs
  a0 0.78902 (+1.94 pp), KL lower bound 0.32522 vs 0.38378 (−15.3%), PPL 4.03563 vs 4.28692 (−5.9%);
  32/32 rows won on top-1, KL and NLL.** First artifact in the program to beat a0 on all three.
- **Task benchmarks agree**: quick MMLU **0.8360 ± 0.0104** vs a0 0.8342 ± 0.0105 (highest in the
  program); GPQA MC **0.4394 ± 0.0354** vs a0 0.4343 ± 0.0353 → both release bars met. The 20-sample
  set is the one miss: 14/20 vs the ≥16/20 bar (a0 re-run: 15/20).
- **The 10-layer variant confirms a dose–response**: top-1 78.90 (0 layers) → 80.28 (10 layers) →
  80.84 (12 layers); the 10L at 94,293,197,696 B still beats a0 on all three panel metrics.
- **MTP is now serving — but not for this artifact.** The program's own D2/B12x recipe
  (`b12x-runtime/mtp-baseline-replay/d2-c1-attempt2`, image `glm53-b12x-exl3:jovian-3aada677-r3`, B12X
  attention, `fp8_ds_mla`, MTP depth 2, `FULL_DECODE_ONLY` graphs) serves the K2 target at **19.23 /
  19.00 total decode tok/s** with 98.3–99.6% speculative acceptance and ctx 262,144 — the receipted D2
  rate, with sampling on. The mosaic cannot use that runtime: it is a **mul1** artifact and both
  MTP-capable images reject `codebook != "mcg"` (`exl3.py:567`). So the two-lineage split is now measured:
  **fast path (mcg) has no mosaic; the mosaic (mul1) keeps the exl3-plain path at 9–10 tok/s.**
- **What it would take to close it**: (1) implement mul1 in the vLLM exl3 overlay — loader/kernel work,
  multi-day plus GPU qualification; (2) re-quantize the mosaic to mcg — still blocked (exllamav3 CUDA
  extension fails to build on Spark); (3) accept the split. Owner decision; nothing is running toward it.
- **Fleet note**: the two sealed turboderp source quants the mosaics were byte-copied from were removed
  from 557f `/home/valentine/models/` (~300 GB) by someone other than this program between
  2026-09-16T22:00Z and 2026-09-17T12:00Z. Both mosaic artifacts are intact and byte-identical to their
  pack receipts; nothing in this program needs those sources any more (the mosaics are already built).

## STATUS after Gate A (2026-09-16, "let's test it") — see `mosaic-gatea/GATE-A-RESULTS.md`

- **Tested and it works — at a coarser granularity than planned.** The overlay fuses all experts of a layer
  into one stacked tensor, so bits must be **per layer**, not per expert (per-expert mosaic failed at load:
  `stack expects each tensor to be equal size … [256,128,32] vs [256,128,48]`).
- **The per-layer mosaic was built by byte-copy only and serves**: 12 layers (3, 32–33, 36–44) at 3 bits,
  all 288 experts everywhere, **96,105,137,024 B**, every tensor byte-equal to a0 or the 3.05bpw base.
- **It beats a0 on the frozen mini-panel (rows 0–3, 8,188 positions): top-1 0.8282 vs 0.8096 (+1.86 pp),
  KL lower bound 0.3957 vs 0.4749 (−16.7%), PPL 2.8463 vs 3.0162** — the first artifact in the program to
  beat a0 on the panel.
- **Context gate**: fails at mem-fraction 0.90 (KV 136,896/193,472), **passes at 0.95 (KV 595,200 ≥
  262,144)**; CUDA graphs fail (mamba state cache) at this size.
- **Speed**: mosaic ≈ a0 (~9–10 tok/s in the eval config; both run without MTP and without graphs). The
  20 tok/s target traces to a **real receipt from this program**: D2 (native FP8 MTP draft, depth 2,
  target/draft CUDA graphs admitted, C1, 262144 ctx, mf .93) measured **19.11 tok/s @1k / 19.03 @16k** vs
  D1 14.98/14.86 — see `EXPERIMENTS.md` 2026-09-12 09:32Z and `benchmarks/structured-sustained-20260912/TABLE.md`.
  The current exl3-plain path **drops the MTP draft at load** (`dropped:nextn_mtp_off 3508 tensors 1.936 GB`,
  all graph stages 0.00), so speculative decoding never runs. **The ~2× gap is a serving-stack capability gap
  (no nextn draft, no graphs), not a mosaic property** — and the mosaic keeps MTP layer 45 verbatim, so
  nothing in it blocks closing that gap.
- **Next**: full G4 panel + quick MMLU + GPQA MC vs a0 (confirming run); a smaller variant (8–10 layers) to
  fit mf 0.90 with OS headroom; per-expert granularity only if we patch the overlay (days).

Written 2026-09-16. This replaces the dense version of this document. The frozen technical spec and every
receipt are in the Appendix, the test outcome is in `mosaic-gatea/GATE-A-RESULTS.md` — you do not need
either to make the decision.

## 1. The goal

One model file that:
- runs on **one Spark** (TP1),
- keeps **262k context + vision**,
- scores **better than what we ship today** (a0: 83.42 MMLU / 43.43 GPQA).

## 2. The idea in three sentences

1. We ship a0: all 288 experts, 2 bits per weight, 85 GB — it works but it is 2-bit everywhere.
2. We already proved the Spark can serve a **96 GB** file with full context (F216 did it) — we're leaving
   ~11 GB of working space unused.
3. There is no "uniform 2.5 bits" option in this format, but there **is** a way to spend that 11 GB: keep all
   288 experts, and give the **most-used ~30% of experts 3 bits instead of 2** (copy those experts
   byte-for-byte out of the 3.05-bit file, which we already have). Bigger file, same experts, better
   precision where it matters.

## 3. Outcome ladder — each step, what you get, cost, and what blocks it

| # | Step | What you get (outcome) | Cost | Blocked by |
|---|---|---|---|---|
| 1 | **One-layer test**: build a copy of a0 where only layer 20's top 86 experts are 3-bit (≈85.5 GB) and serve it | **Go / no-go for the whole idea.** Proves the serving software can read a file that mixes 2-bit and 3-bit experts in the same layer — nobody has ever tried this | ~2 h (CPU pack + smoke serve) | Nothing. Sources and tools are already on 557f |
| 2 | **Full build**: upgrade the top 86 experts in all 42 layers (3,612 copies) | A 96.6 GB artifact, all 288 experts, ready to serve | ~1–2 h CPU | Step 1 says yes |
| 3 | **Serve + measure**: run the same eval suite we used for every other candidate (panel + quick MMLU + GPQA) | **The numbers.** Either a new release candidate or a clean negative result | ~1 GPU-day | Step 2 artifacts, eval box free (2384/557f idle) |
| 4 | **Decision** | Ship the new file **only if** MMLU beats 83.42 **and** GPQA doesn't drop below 43.43; otherwise a0 stays | — | Step 3 numbers |

So the shortest path to an answer is **step 1 (~2 hours)**, and the whole question is answered in
**~1.5 days of mostly-idle hardware**.

## 4. The one real unknown (the dependency that matters)

**Can our serving software decode a file where experts inside one layer have different bit widths?**
Everything above hangs on that single yes/no. It's untested, and it's the reason step 1 exists: for ~2 hours
of CPU time we find out before building anything big. Two ways it can fail, both cheap to detect:
the file won't load, or it loads but the 4-sample check shows the outputs are garbage.

If it fails, the options (and what each needs):
- **Whole layers instead of experts** (all of layer X at 3-bit, others at 2-bit) — same idea, coarser
  granularity, maybe already supported. Cost: still ~1–2 h to test.
- **Patch the serving software** to read bits per expert — a real but bounded engineering job. Cost: 1–2 days.
- **Build our own quantization pipeline** — this is the "proper" version but it needs a working
  exllamav3 build on the Spark, and that **currently fails to compile** (tried on Sep 12, logged on 557f).
  Multi-day before we could even start. That's why it's last, not first.

## 5. What we already have (so it costs nothing extra)

- **Both source files**: a0 (2.05-bit) and the 3.05-bit base, both on 557f, both with all 288 experts.
- **The packer tool**: pure CPU, already used four times to build artifacts (E216/S216/T216/F216).
- **The ranking data**: the full observation capture (37.3M tokens) tells us which experts are most used.
- **The eval harness**: same panel/MMLU/GPQA runners used for every previous candidate.
- **Disk**: 557f has 1.5 TB free (needs ~100 GB). No downloads, no training, no requantization.

## 6. Honest expectations

- The experts we'd upgrade carry about **half** the model's usage mass. So this improves precision on ~half
  of the expert compute by one bit — expect a **small gain, possibly zero**, not a step change. It is,
  however, the only route left that the published literature backs for beating a0 at this size.
- There is **no mechanism for a big loss**: every expert stays, the router is untouched, and the
  non-upgraded experts are byte-identical to what a0 ships today.
- The file is 13% bigger, so serving may be slightly slower. We measure that in step 1.

## 7. If the mosaic route ends below a0

| Option | What it needs first | Honest expectation |
|---|---|---|
| Richer attention/dense layers instead of experts | Same tooling, +3 GB | Unknown; literature says dense layers are precision-sensitive |
| Heal the old pruned model | Training runs (2–5 GPU-days) | Literature: recovers at most half the loss → still behind a0 |
| Build a full mixed-precision quantization pipeline | A working exllamav3 build on the Spark (currently broken) + days of quantization | Highest ceiling, highest cost, blocked today |
| Stop at a0 | Nothing | a0 already beats every pruned candidate we built |

---

## Appendix — frozen technical spec (for the record)

**Alignment finding (the reason byte-copy works).** a0 and the 3.05 base are quants of the same model from
the same quantizer lineage (turboderp/GLM-5.3-Flash-exl3, exllamav3 v1.4.4, codebook mul1, cal 250×2048),
both with all 288 experts and identical tensor keys
(`model.language_model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.{trellis,suh,svh,mul1}`).

**Byte accounting** (measured 557f, index + safetensors headers only, this turn):

| Quantity | a0 (2.05bpw) | base (3.05bpw) |
|---|---:|---:|
| per-expert payload (12 tensors) | 6,328,332 B | 9,474,060 B |
| 2→3 upgrade delta per expert | — | **+3,145,728 B** |
| index total | 85,128,745,176 B | 122,286,101,684 B |
| expert tensors | 78,370,063,488 B (12,384 incl. MTP) | 114,598,229,760 B (12,096) |
| non-expert trunk | 4,823,181,364 B | 7,687,871,924 B |
| du total | 85,233,484,348 B | 129,546,716,672 B |

**Variant M288 (frozen):** a0 dressing byte-exact (MTP layer 45 verbatim, head_bits 5, mtp_bits 2, kpool
in-index); for each of 42 trunk layers the top **86/288** experts by full-seal `weighted_ean_sum`
(23,088 records / 37,328,459 tokens, `f216/f216-plan.json`; ties → lower id) are the base's 3-bit tensors;
all 288 ids present, no count deviation. Predicted size **96,595,853,116 B** = a0 + 3,612 × 3,145,728 B —
within 2.5 MB of the F216-proven envelope (96,593,390,114 B served KV 278,528 ≥ 262,144 at mem-fraction
0.90). Coverage: top-86 = 49.6% of activation-weighted mass (per-layer 44.8–66.1%). Effective ≈2.30 bpw.

**Serving (Gate C) config** = a0-equivalent: ctx 262144, fp8_e4m3 KV, dsa/flashinfer_sparse_mla,
mem-fraction 0.90, multimodal on, chat template mm, reasoning glm45 / tool glm47, disable-cuda-graph.
KV gate ≥ 262,144. Bars: MMLU > 83.42 (±1.2), GPQA ≥ 43.43 (±3.5), samples ≥ 16/20, G4 panel vs a0
78.90 top-1 / 0.384 KL_lb.

**Receipts:** a0 = 2384 `models-2p05-stage/2p05bpw/` + 557f copy; serving flags
`benchmarks/lm-eval-bench/a0-release-serving-inspect.json`; base = 557f
`models/turboderp-glm53-flash-exl3-3p05bpw/` (rev 332ab457…, shas in
`f216/receipts/f216-base-shas-2822.txt`); envelope = `f216/BUILD-F216.md` +
`f216/receipts/g4/f216-g4-summary.json`; tooling = `u216/prune_u216.py`, `u216/census_plan_aware.py`,
`f216/run-serve-f216.sh`, `f216/run-f216-g4.sh`; failed Spark build of the quantizer =
557f `models/p3b-out/build.rc=1`, `fa-build.rc=1`.

**Follow-up arms (after v1):** V2 = a0 experts + base's richer dense/attn (+2.87 GB → ≈88.1 GB);
V3 = V1+V2 (≈64 upgrades/layer); V4 = 4-bit tier for top-8/layer (source: 2822
`GLM-5.3-Flash-EXL3-Q4-d0b9301a`, lineage to verify); V5 = own quantization pipeline (blocked on the ext
build). Fleet this turn (read-only): 2384 a0-bench up (1.3 TB free), 2822 idle (1.7 TB free), 557f nv shim
only (1.5 TB free), de5c DOWN expected. Free space is larger than the 10:52Z receipts (owner cleanup today)
— recompute from live `df` at build time.