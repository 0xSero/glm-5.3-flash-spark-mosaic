# MoE compression literature synthesis for the GLM-5.3-Flash prune program

Written 2026-09-16 (interactive turn, live arXiv/HF literature search; no GPU work). Purpose: answer
"is there anything at all we can do to improve a prune, and what about REAM?" against our exact
constraint set, and turn the answer into a ranked experiment ladder. All literature numbers below are
**as reported by the cited papers**, not our measurements; our own numbers are marked as such and every
one has a receipt path in the repo.

## 0. Our constraint set (what makes this unusual)

| Constraint | Consequence |
|---|---|
| Weights are EXL3 trellis-packed, third-party-produced; we have **no BF16 source on-fleet** and no active quantizer pipeline (though `exllamav3` v1.4.9 source IS on 557f: `/home/valentine/models/exllamav3-v1.4.9`) | Any method that merges, averages, edits or re-encodes weights is blocked or expensive |
| Router + dense + embeddings are "protected/native" | No router adaptation, no bias redistribution without a scope decision |
| SGLang/exl3_plain historically requires a **uniform expert count per layer** (a2d era) | Non-uniform per-layer budgets blocked until a loader probe passes |
| One node, 128 GB UMA, must hold 262144-token context | Servable artifact ≲ 96-107 GB; the 3.05bpw parent (129.5 GB) is unmeasurable |
| Available quants: 2.05bpw (85.2 GB), 3.05bpw (129.5 GB), 4.05bpw | The only servable "more precision" route was pruning plus 3.05bpw |

## 1. The literature predicts our negative result (three independent sources)

1. **REAP, Appendix E** (arXiv:2510.13999, Cerebras; the paper whose method we implemented): "weight
   quantization offers superior accuracy at a given compression ratio down to 4-bits… quantizing
   Qwen3-30B-A3B to 4 bits outperforms a 50% expert-pruned REAP model at 16 bits, despite having half
   the checkpoint size." Their A5 table: 16-bit/64 experts (50% size) 78.0 vs 4-bit/128 experts (25%
   size) 80.5; 2-bit collapses (28.6). Crossover: pruning only pays **below ~3 bpw**.
2. **MoEXBench** (arXiv:2608.21693; 10 MoEs 30B-235B, pruning × quantization grid): "~19-25% expert
   pruning is far more damaging than aggressive 4-bit quantization" — Qwen3-30B-A3B at 19.35% expert
   pruning: PPL +28.5%, vs 69.5% Q4_K_M quantization: PPL +0.7%. At matched settings REAP@25% costs
   5.31 quality points and REAM@25% costs 6.05, while adding Q4_K_M costs 1.23 and KV-Q8 only 0.31.
3. **Half the Experts, All the Code** (arXiv:2607.16721): pruning beats quantization **only where
   quantization would have to drop below 3 bits per weight**; five pre-registered attempts to overturn
   that crossover all failed. Also: damage lands outside the intended domain; PPL can rate a broken
   model above an intact one.

**Applied to us**: our pruned artifact is 3.05bpw × 216/288 ≈ **2.29 effective bpw**; the incumbent a0
is **2.05bpw with all 288 experts**. We are exactly in the regime the literature says quantization wins,
and it does: +6.14pp MMLU (E216 vs a0, results/), +10.4pp panel top-1 (65.7 vs 78.9). Our negative
result is not an implementation failure; it is the expected outcome of the axis we chose.

Independent support for two of our observed anomalies:
- arXiv:2609.04453 (over-dispersed routing): under strong load balancing the importance signal
  *collapses*, PPL stops predicting accuracy, and different scoring families can differ by 18 points on
  GPQA alone — consistent with our 2.8-pt ranking spread being irreducible noise on a load-balanced model.
- arXiv:2507.23279 (super experts): a handful of individual experts are load-bearing; pruning just
  **three** specific experts destroys a 30B MoE's output. Aggregate ranking formulas can miss them.
- arXiv:2606.15716 + AIMER (arXiv:2603.18492): ranking choice is a real but bounded lever (top-two
  averaged ranks, up to +8.8 points on some suites; calibration-free ranking within noise of
  calibration-based). Matches our six-formula, 2.8-pt spread.

## 2. REAM specifically (arXiv:2604.04356)

REAM = "Router-weighted Expert Activation **Merging**" — the merging counterpart of REAP: instead of
deleting the lowest-scoring experts, group them and **merge weights** (router-weighted activation
merging), sometimes beating REAP on multiple-choice QA in the authors' table, with the MC/generation
trade-off depending on calibration mix.

Why it does not help us today:
- **It is a weight-surgery method.** Merged experts must be re-encoded. REAP states explicitly that
  merging "necessitates re-quantization" for group-scaled formats — exactly EXL3 trellis packing.
  We would need: unpack 3.05bpw (or 4.05bpw) to BF16, merge, re-quantize at ~2.3-3.0bpw, re-verify.
- **It measures worse than pruning at our rate**: MoEXBench 25%-reduction numbers above (6.05 vs 5.31
  quality points lost). So even after paying the requantization cost, the expected gain over the
  pruning we already did is negative.
- Its own motivation (REAP's "functional subspace collapse" argument is *against* merging; REAM argues
  the trade-off is data-mix-dependent) does not change the class: (W)+(Q).

Verdict: **deprioritized** — highest cost class, no measured advantage at 25%, requires a toolchain we
have not stood up.

## 3. What remains applicable without weight surgery (class S)

Ranked by expected value under our constraints. All are bounded: the literature's best non-uniform
budget results are +1.4-2.5 points average over uniform budgets, and no paper reports a 25%-pruned,
unhealed MoE beating a smaller unpruned quant.

1. **Do not prune for size; keep 288 experts at 2.05bpw (a0) and spend the headroom on context/KV
   quality** — already our release. KV-Q8 costs +0.31 points (MoEXBench); a0 already runs fp8_e4m3 KV
   with a 1,057,088-token pool. This is the literature-optimal configuration and needs no work.
2. **Serving-level routing-mass redistribution for removed experts** — the only zero-quantizer idea that
   attacks our measured structural damage directly. When a token's top-8 includes a pruned expert, the
   overlay currently falls through to the router's next-best survivor; a better fallback is the most
   *similar* survivor (expert-output similarity computable from the sealed observations). This is
   "merging at inference time" without touching weights. Cost: overlay/serving work (engineering, no
   GPU training). Expected: unknown; literature is agnostic. Measure on the G4 panel + a generative
   holdout before believing it.
3. **Super-expert protection** (arXiv:2507.23279): identify the small set of load-bearing experts (from
   observation data: top ean × weight, or the repo's `perserve_super_experts` mechanism) and hard-protect
   them in any future plan; drop more from demonstrably harmless layers to keep the count uniform.
   Cheap (plan-side only), bounded upside, but it is the one intervention that addresses a *qualitative*
   failure mode rather than average-case ranking.
4. **Re-rank on the deployed precision** (MoEXBench's score-deployment mismatch): all six of our
   rankings were fit on observations from the **Q4 (4.05bpw-class)** model and applied to a 3.05bpw
   artifact. Re-capturing a small observation subset from the 3.05bpw base and re-ranking tests whether
   our 2.8-pt spread is a precision-transfer artifact. Cost: a partial capture (the pipeline exists) —
   days, not hours, but no weight edits.
5. **Non-uniform per-layer budgets** (GRAPE arXiv:2604.06542, EvoESAP arXiv:2603.06003, DiEP
   arXiv:2509.16105; LAMP arXiv:2010.07611 as the principled template): +1.4-2.5 points average in the
   literature, up to +7-19% on single generation benchmarks at 50% sparsity. Blocked by the
   uniform-expert-count loader; a ~1 h config-only probe decides feasibility.
6. **Cheap cross-check rankers**: MAN/MSAN (arXiv:2606.15716) or AIMER (arXiv:2603.18492, calibration-free)
   as a sanity check against our six — cheap, but expected to move the panel by ~1 point, not 10.

## 4. Blocked today, but the literature's recommended fix for our exact hardware

**Mixed-precision bit allocation over all 288 experts** — do not delete experts; spend bits where they
matter. This is the one class of method aimed precisely at "fit a ~3bpw-class model into 85-96 GB on a
128 GB node":
- QuantMoE-Bench (arXiv:2406.08155): shared experts need more bits than routed experts; fine-grained
  mixed precision 65.35% vs 64.30% GPTQ.
- MC-MoE (arXiv:2410.06270): importance-driven bit allocation, 2.54 avg bits, 76.6% compression, 3.8%
  average accuracy loss, training-free. MC# (arXiv:2510.10962): 6.2× reduction at 2.57 bits, 1.7% drop.
- BitsMoE (arXiv:2606.00079): +27.83 points over GPTQ at 2 bits. AlphaQ (arXiv:2606.04980):
  calibration-free allocation. Q-Strata (arXiv:2608.30564).
- Deployment precedent (arXiv:2509.25689): DeepSeek-V3 1.3 TB → **103 GB on a strict 128 GB platform**,
  beating uniform low-bit quantization at equal memory.
- MoQE (arXiv:2310.02410) / QMoE (arXiv:2310.16795): expert layers tolerate quantization far better
  than dense layers; no training needed in most cases.

Requirements we lack: a working quantization pipeline and a full-precision-enough source. Both are
engineering decisions, not walls: exllamav3 v1.4.9 source is already on 557f, and the BF16 model
(`zai-org/GLM-5.3-Flash`) is public. Cost would be: source weights (download/dequant), a quantizer run
for a 330B-class model on one Spark (multi-day), plus G2/census/panel validation. **This is the only
path the literature supports that could plausibly beat a0 at the same memory** — and it needs the user's
go-ahead plus a disk/spend plan.

## 5. Healing, priced by the literature

- "Beyond Retraining-Free MoE Compression" (arXiv:2609.06076): with 3,000 examples and one epoch, full
  fine-tuning recovers only **37.3%** of the original-to-compressed gap on average.
- "Half the Experts" (arXiv:2607.16721): a lightweight fine-tune recovers about half of what aggressive
  pruning loses. "Not All Experts are Equal" (arXiv:2402.14800): 2-of-8 pruning costs 2.9 points
  task-agnostic, 6.2 task-specific, 1.6 with task-specific fine-tuning.
- No paper found that heals expert-pruned MoE by tuning **only** the router; the closest (arXiv:2608.07890)
  tunes the router to *find* prunable experts, and arXiv:2608.11212 finds route-mediated damage is
  detectable but not selectively repairable.

Therefore healing's honest ceiling is ~50% of our ~6pp gap → still below a0, matching healing/PLAN.md's
low prior. Not worth its 2-5 GPU-days unless the goal changes from "beat a0" to "ship a 96 GB artifact".

## 6. Experiment ladder (cheap → expensive), all pre-registered against a0

| # | Experiment | Class | Cost | Expected | Decision value |
|---|---|---|---|---|---|
| 1 | Retained-routing-mass diagnostic across the 6 existing plans | S, offline | ~1 h CPU | n/a (diagnostic) | Screens any future plan for free |
| 2 | keep-216-of-a0 control (isolate pruning damage at fixed bpw) | S | ~1 GPU-day | −4 to −6 pp MMLU expected | Turns healing/PLAN.md's "pruning-dominated" inference into a measurement |
| 3 | keep-240 @ 3.05bpw (mild-prune slope) | S | ~1.5 GPU-days | partial recovery, still < a0 | Pins damage-vs-keep-count slope |
| 4 | Non-uniform-keep loader probe | S | ~1 h CPU | pass/fail | Unblocks the only structural budget lever (+1.4-2.5 pts literature) |
| 5 | Super-expert protection + gate-free ranker, combined plan, re-eval | S | ~2 GPU-days | +0-2 pts | Final word on selection-only pruning |
| 6 | Serving-level routing fallback to most-similar survivor | S (serving) | engineering + 1 GPU-day | unknown | Only idea that targets the measured mechanism |
| 7 | Re-rank on deployed precision (3.05bpw observations) | S | ~2-3 days (capture) | unknown | Tests whether our spread is a precision-transfer artifact |
| 8 | Mixed-precision bit allocation, 288 experts, re-quant pipeline | Q (+W) | multi-day build + toolchain | literature: the winning axis at equal memory | The only literature-supported path that could beat a0 |

Methodology fixes the literature recommends regardless of which experiment runs:
- Keep the G4 panel (teacher agreement) but **add a small generative/agentic holdout** before declaring
  any pruned artifact dead — PPL and teacher-agreement panels are documented to invert rankings of
  broken-vs-intact models (arXiv:2607.16721, arXiv:2609.04453). Our GPQA/MMLU legs partly cover this;
  a task-shaped holdout would cover it better.
- Report **effective bpw** (bpw × keep fraction) in every comparison, not nominal bpw: our family's
  numbers are only interpretable that way (96.6 GB pruned = 2.29 effective bpw vs a0 at 2.05).

## 7. Sources

Primary (read in full or abstract+key tables): arXiv:2510.13999 (REAP), arXiv:2604.04356 (REAM),
arXiv:2608.21693 (MoEXBench), arXiv:2607.16721 (Half the Experts), arXiv:2609.04453 (load balancing /
MESA), arXiv:2507.23279 (super experts), arXiv:2609.06076 (cost of compression recovery),
arXiv:2509.25689 (103 GB DeepSeek-V3 on 128 GB), arXiv:2410.06270 (MC-MoE), arXiv:2406.08155
(QuantMoE-Bench).
Secondary (abstract-level, cited for classification): arXiv:2402.14800, 2404.05089, 2407.00945, 2410.08589,
2410.12013, 2310.01334, 2506.23266, 2509.10377, 2510.14436, 2603.06003, 2604.06542, 2509.16105,
2606.15716, 2603.18492, 2606.00079, 2606.04980, 2608.30564, 2608.07890, 2608.11212, 2505.05799, 2410.10962
(as arXiv:2510.10962), 2310.02410, 2310.16795, 2502.00997, 2603.12645, 2605.13997, 2310.09832, 2608.24938,
2509.11177, 2408.11796, 2411.10272, 2010.07611, 2607.01444, 2608.08910.