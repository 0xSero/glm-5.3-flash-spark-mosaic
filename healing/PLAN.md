# Healing experiment plan — queue item 5 (brief) — WRITE-ONLY, nothing launches without user approval

Written 2026-09-15T01:1xZ (driver run 18). Status: **PLAN ONLY** — no GPU job, no build, no deletion is authorized by this file. This document exists so the open fork decision — **(a) a0-fallback as release** vs **(b) healing/layer-adaptive keep** — can be made with real costs, real gates, and an honest prior. The brief's queue item 5 says exactly this: write the plan, stop there for user review.

## 1. What healing has to overcome (measured, receipted)

| Fact | Number | Receipt |
|---|---|---|
| Pruning damage vs teacher (panel): best keep R216 | KL +0.393 nats over A0 (0.777 vs 0.384), top-1 −10.4pp (68.53 vs 78.90) | brief leaderboard; r216/receipts/r216-g4-summary.json |
| Ranking quality recovers | only ~0.1 nats (U216 0.905 → R216 0.777) | same |
| Benchmark: best keep vs a0 | MMLU-quick −6.14pp (~3.6σ, E216), GPQA MC −2.5pp (R216, ~1σ ns), GPQA CoT-flex −14.1pp (R216) | results/{a0,e216}-quick, results/{a0,r216}-gpqa, results/e216-gpqa-mc |
| What pruning buys | 3.05bpw fits 128 GB ONLY because 25% of experts are gone (unpruned 3.05bpw ≈ 125 GB + KV does not fit); a0 at 2.05bpw is ~66 GB with more KV headroom | BUILD-R216.md / brief |
| Damage attribution | a0 (2.05bpw, aggressive quant, ALL experts) beats 3.05bpw-25%-pruned ⇒ the loss is PRUNING-dominated, not quant-dominated — which is the damage healing targets | EXPERIMENTS.md 2026-09-15 ~00:15Z |

**Binding burden of proof for any healed candidate** (pre-registered here, before any healing run): on the same prereg configs —
1. MMLU quick (1,140 docs, seed 1234): ≥ **84.62%** (a0 83.42% + 1 combined-σ of ~1.7pp... σ_1.20 combined ⇒ +1.20pp), i.e. a statistically robust beat, not a tie;
2. GPQA MC (198 docs): ≥ **46.96%** (a0 43.43% + 1σ ~3.5pp);
3. G4 panel (same lower-bound estimator as E216's row): KL_lb ≤ **0.80** (clearly below E216's 0.898; approaching the R216-exact era is not checkable same-estimator — R216 0.777 was exact-vocab — so the honest same-estimator baseline is E216's 0.898);
4. Samples gate: ≥ 16/20 (E216's level; R216/S216/U216 were 12/20).

If any gate fails after the bounded budget below, healing STOPS and a0-fallback stands; the program archives as a negative result with all receipts.

## 2. Constraints that shape the design (all binding)

- Artifacts are EXL3-packed at 3.05bpw — packed tensors are **not fine-tunable in place**. Any weight update requires unpack → update → re-quantize/re-pack with G2 re-verification.
- **Protected tensors stay native**: dense layers, head, embeddings, router/shared — no updates there. Router-bias "healing" (redistributing load toward kept experts) is therefore **OUT OF SCOPE**. MTP layer 45 stays verbatim (288 experts, no saliency, never pruned). Expert-side tensors (gate/up/down of kept experts) and per-expert output projections are the only healable surfaces.
- Eval hygiene: healing data = the sealed reap calibration corpora ONLY (0xSero/reap-calibration-data-v1 @115e754a + sealed observations corpus). **Never** MMLU/GPQA/tbench/panel texts — the panel and benchmarks stay frozen eval surfaces.
- One GPU job per node: healing compute belongs on **2822** (idle, ~2 TB free post-purge). 2384 is owned by the eval/recorder servers; 557f is owned by serving. Sealed models/teacher rows/calibration data/receipts are never deleted; no reboots; no network changes.
- Every attempt preserved with receipts (attempt logs, plan shas, G2 receipts); 0444 plan JSONs; UTC everywhere.

## 3. H1 — calibration-folded correction (cheap probe, ~½ GPU-day total)

No training loop. Estimate per-layer/per-expert corrections from teacher-vs-student residuals on calibration data and fold them into expert-side tensors only:

- **H1a (global)**: per MoE layer 3..44, one scalar s_l = E[teacher routed contribution] / E[student routed contribution] on a calibration subset; multiply kept experts' down-projection outputs (folded into the per-expert o_proj/down tensor) by s_l.
- **H1b (per-expert)**: per kept expert e in layer l, scalar s_{l,e} from the same pass — finer, still zero backprop.
- Data: sealed observations.json already holds the TEACHER side (router weights × expert norms per routed token); the STUDENT side needs one forward pass of a kept-216 candidate over a ~10% calibration subset (~3 M tokens) at prefill speed — hours on 2822.
- Compute budget: ≤ 12 GPU-hours on 2822 (forward passes + pack + G2 + census).
- Eval after H1: G4 panel on 2384 (fail-fast gate 3) → only if passed, quick MMLU + GPQA MC on 557f serving slot.
- Honest prior: heuristic zero-shot corrections in the literature recover a minority of the pruning gap. H1 most likely FAILS gate 1/2 — its purpose is to measure, for ~½ GPU-day, how much of the damage is linearly correctable, which decides whether H2 is worth its multi-day cost.

## 4. H2 — kept-expert calibration fine-tune (the brief's named approach; multi-day)

Sequential per-layer healing so it fits 128 GB:

1. **Pre-launch verification (blocking)**: locate + size + sha the BF16 original (zai-org/GLM-5.3-Flash-BF16 @ a6c167b6 was confirmed real via the G4 head shard; full-model location on-fleet UNVERIFIED — check 2822 archive/NFS NAS read-only). If the BF16 original is absent: H2 teacher becomes the on-fleet a0 2.05bpw EXL3 (already proven a valid teacher for this panel), and BF16-unpack healing is infeasible — then only LoRA-into-EXL3 or H1 remain.
2. **Pilot, 2 layers only**: unpack kept experts of the 2 worst panel-KL layers to BF16 (~11 GB/layer unpacked — fits alongside the ~66 GB teacher), fine-tune on calibration tokens (bounded: ≤ 2 M tokens/layer, LR ≤ 1e-4, seq ≤ 2k), re-quantize to the SAME per-tensor bpw (3.05), G2-verify, fold back.
3. **Go/no-go extrapolation**: measured per-layer KL gain × 42 ≥ enough to plausibly pass gate 3, else stop after the pilot (total spend ≤ 1 GPU-day).
4. **Full run**: 42 layers × (load → train ≤ 2 M tokens → requantize → G2) on 2822; honest estimate **2–5 GPU-days** at Spark speeds; checkpoints per layer (never delete; +~190 GB working, disk OK on 2822).
5. **Known quantization risk (receipted in advance)**: re-quantizing a fine-tuned weight at fixed 3.05bpw re-incurs quant error and forfeits the original quant-calibration optimum; the pilot's step-3 gate exists precisely to catch "healing gains eaten by requantization". A LoRA-delta-at-native-bpw variant has the same risk, not less.

## 5. Related non-healing variant (fork (b) other half): layer-adaptive keep

Non-uniform keep budget per layer (e.g. keep-192 where routes concentrate, keep-240 where they don't) within the same total-expert budget. Caveat recorded from the a2d failure: variable expert counts per layer previously FAILED SGLang's uniform-expert-count requirement (a2d/ note in the brief — "do not resurrect without a loader path"). So layer-adaptive-keep requires FIRST proving the exl3_plain overlay accepts per-layer n_routed_experts (a config-only probe on the EXISTING s216 artifact dirs is ~zero-GPU: edit config copies, try to load on 2822). That probe is the only cheap part; the ranking/build cost afterwards is a full pipeline run. Do not start without the loader-path result + user approval.

## 6. Cost/decision summary for the fork

| Option | GPU cost | Storage | Chance of passing the binding gates (honest) | Recommendation logic |
|---|---|---|---|---|
| (a) a0-fallback | 0 (restore = panel-tune serving inspects, preserved) | a0 already on 2384 | n/a — already the benchmark-backed release | Recommended: smaller artifact (~66 GB vs ~94 GB), better KV headroom, wins every measured surface |
| H1 probe | ≤ 12 h (2822) | +94 GB candidate | Low (prior: partial linear recovery) | Only worth it as the cheap measurement that prices H2 |
| H2 full | 2–5 days (2822) + pilot 1 day | +94 GB + ~190 GB working | Unknown; must recover ~100% of a 0.4-nat gap to break even vs a0 | Only after H1 shows substantial linear headroom |
| Layer-adaptive keep | loader probe ~0; then full pipeline | +94 GB | Unknown; blocked by the a2d loader finding | Dead unless the config probe passes |

**Default if the user says nothing: a0-fallback stands; this plan waits unlaunched.**
