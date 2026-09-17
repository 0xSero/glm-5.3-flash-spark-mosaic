# Post-F216 analysis — what the capture was for, and where improvement is actually possible

Written 2026-09-16 (interactive turn, no GPU work, no artifact mutations). Every claim below is either a
measurement with a receipt path or is explicitly marked as an inference.

## 1. What the capture was for

The F216 goal was: *"do another prune on the exl3 3bpw in a new way — REAP methodology from the original
repo, using the FULL observation dataset"* (f216/PLAN.md).

- The **observation dataset** is the REAP saliency input: forward passes of the original model over the
  361-shard calibration corpus (`0xSero/reap-calibration-data-v1 @115e754a`, 16,384-token records)
  recording, per routed token, which experts fired, the renormalized router weight, and the expert
  output L2 norm (`ean`). REAP's score is `mean_routed(weight x ean) / frequency` (pruning_metrics.py:198).
- The original capture (2026-09-07 → seal 2026-09-11) covered **332 of 361 shards**: the partial seal was
  21,248/23,088 records and 32.6M/37.3M tokens. Missing: **science 195 records (0 captured) + cuda 1,645
  of 2,000** = 1,840 records / ~4.73M tokens.
- **The hypothesis under test**: E216 (the REAP-native prune) underperformed the route-mass-informed
  R216, and one candidate explanation was that REAP had been scored on incomplete calibration data —
  notably with the whole science domain and most of cuda absent on a model that is used for coding and
  systems work. So the capture existed to close that data gap and re-rank, rebuild, and re-evaluate.
- The capture (2026-09-15T14:42Z → 2026-09-16T00:38Z, ~13 h, 2822 rank0 + 557f rank1, resume protocol
  processing only the 29 uncommitted shards) produced the **FULL seal**: 23,088 records / 37,328,459
  tokens, min expert routes 49,895 (vs 49,419 partial).

**Answer: the hypothesis is refuted.** With the complete dataset the REAP keep-set changed a lot
(mean overlap with E216 only **149.52/216** — ~31% of kept experts per layer swapped) and the model got
*slightly worse* on every G4 panel metric, worse on GPQA MC, and better only on quick MMLU.
Receipts: f216/BUILD-F216.md, f216/receipts/build/*, f216/receipts/g4/*, results/f216-{quick,gpqa-mc}/.

## 2. Why the result looks worse than expected — decomposition

### 2.1 The ranking axis is small; the cut itself is the cost

| Family (all keep-216 @ 3.05bpw) | Ranking input | panel top-1 | KL (lb) |
|---|---|---:|---:|
| R216 | sensitivity x ln(routes+1), dense routing capture | 68.53% | 0.777 |
| S216 | sensitivity x ln(routes+1), thin capture | 68.02% | 0.801 |
| U216 | massmax_domain routing mass | 66.83% | 0.905 |
| T216 | sensitivity only (refuted) | 66.50% | 0.855 |
| E216 | REAP-native, partial seal | 66.41% | 0.898 |
| F216 | REAP-native, FULL seal | 65.69% | 0.939 |
| a0 (reference, unpruned 2.05bpw) | — | **78.90%** | 0.384 |

Spread across six ranking formulas: **2.84 pp top-1** (all on 65,504 panel positions).
Gap from the best pruned point to a0: **10.37 pp**. So ~73% of the deficit is shared by every point,
i.e. it comes from the design decision all six share: **uniform removal of 25% of experts (72 of 288)
per layer**. F216 vs E216 differences (-0.72 pp panel, +1.40 pp MMLU ~1σ, -1.01 pp GPQA ~0.3σ) are at or
near the noise floor of the harnesses (MMLU ±1.2 pp, GPQA ±3.5 pp; panel positions are
sequence-correlated). Robust statement: *all six pruned points sit far below a0; which experts get cut
barely matters.*

### 2.2 The mechanism (inference, consistent with the numbers)

Routing is top-8 of 288 with `norm_topk_prob=true` (config.json, verifiable in
a0's config: `num_experts_per_tok=8, n_routed_experts=288, n_shared_experts=1, 45 layers`).

- Each token consumes only ~2.8% of experts, so *which* 72 you delete matters little to any given
  token only if the router's preferences are respected — but the router is a **protected/native tensor**
  (constraint) and cannot adapt; deleted experts' weights are silently renormalized onto survivors.
- Under a random 25% cut ~90% of tokens lose at least one of their 8 preferred experts
  (1-(216/288)^8 ≈ 0.90). Mass-ranked cuts concentrate the hit on the tail, which is why route-aware
  scoring (R216/S216) wins the family — but the tokens that do lose a low-mass expert have no
  equivalent replacement, and no mechanism learns one.
- This also explains why F216's big ranking change (66 experts/layer swapped) moved the panel by less
  than a quarter of a point: the damage is structural, not selection-specific.

### 2.3 The control that was never measured

`turboderp/GLM-5.3-Flash-exl3` publishes **exactly three** plain quants — 2.05bpw (rev 51058cd5),
3.05bpw (rev 332ab457), 4.05bpw (rev 2a30229e) — confirmed from the repo README and refs API this turn.
Every pruned artifact (R216/S216/T216/U216/E216/F216) was built from **3.05bpw**, and a0 is the
**2.05bpw** quant of the *same repo, same quantizer, same calibration recipe* (2384:
`/home/sero/models-2p05-stage/2p05bpw/DOWNLOAD-SEAL.json` = repo turboderp/GLM-5.3-Flash-exl3, bits
2.05, head_bits 5, codebook mul1).

Consequences:

- The a0-vs-pruned comparison is **not** confounded by quantizer provenance (same lineage). That part of
  the story is clean.
- But **the unpruned 3.05bpw parent was never evaluated and cannot be evaluated on this fleet**: at
  129.5 GB it exceeds the 128 GB UMA (GPU pool ~115 GB at mem-fraction 0.90). The program therefore never
  knew what "+1.0 bpw of precision" is worth here; it only knew that a 25% expert cut costs ~6 pp MMLU.
- Corollary (this is the important one): to use 3.05bpw at all on a Spark with 262144 context you are
  **forced** to remove experts — keep-216 = 96.6 GB (KV 278,528 tok measured), keep-240 ≈ 107.3 GB
  (fits, but the 262144-context gate likely fails), keep-252 ≈ 112.7 GB (does not fit with usable KV).
  So "prune to buy precision" was never a free choice; it was the only way to serve 3.05bpw, and the
  price (≈ -6 pp MMLU) turned out to be larger than the precision rebate.
- Consequence for documentation hygiene: healing/PLAN.md §1 states *"the loss is PRUNING-dominated, not
  quant-dominated"*. That is an **inference**, not a measurement — it assumes 3.05bpw-per-expert >
  2.05bpw-per-expert. Two cheap experiments (§3.1, §3.2) can turn it into a measurement or falsify it.
- Also correct the size figure used in healing/PLAN.md: a0 is **85.2 GB** on disk (2384 `du -sb` this
  turn), not the "~66 GB" quoted there.

## 3. Improvement options, ranked by information gained per GPU-day

### 3.1 keep-216 of a0 itself — the missing control (~1 GPU-day) [recommended first]

- **What**: pack a0's own 2.05bpw checkpoint (85.2 GB, 2384:/home/sero/models-2p05-stage/2p05bpw) with
  the existing packer (`prune_u216.py`) to keep-216 using the best-known ranking; G2-verify, census,
  serve on 557f, run G4 + quick MMLU + GPQA MC.
- **Cost**: ~10 min pack (tooling proven), artifact ≈64-68 GB, KV headroom large (no context-gate risk),
  1 GPU-day total on idle hardware. 10-minute pre-check needed: a0's dir has no separate
  mtp/kpool_aux files, so the packer's input contract must be confirmed for this dir.
- **What it buys**: Δ_prune(bpw=2.05, keep-216) with quantizer and bpw held fixed. Combined with the
  already-measured Δ(a0 → keep-216@3.05) = -6.14 pp MMLU, it separates pruning damage from the 3.05
  precision rebate: gain(3.05 vs 2.05) ≈ Δ@3.05 − Δ@2.05, and therefore whether *any* servable pruning
  program on this quant family could ever have beaten a0.
- **Pre-registered decision rule**: if keep-216-of-a0 loses ≥4 pp MMLU vs a0 → pruning damage dominates,
  healing is the only remaining path, and its burden of proof (recover ~100% of a 0.4-nat panel gap)
  stands as written. If it loses ≤2 pp → the loss lives in the 3.05 quant itself (the precision rebate is
  ~zero on this eval surface), and the pruning program closes with a much stronger negative result.

### 3.2 keep-240 @ 3.05bpw — mild-prune frontier point (~1.5 GPU-days)

- 107.3 GB artifact; serves for quality measurement, but the 262144-context release gate likely fails
  (≈8-13 GB left for KV) — this is a **mechanism probe, not a release candidate**.
- Uses the same pipeline as F216 with `n_prune=48` instead of 72 and the best-known ranking.
- Prediction to test: if damage scales with experts removed, keep-240 recovers ~1/3 of the 6 pp and still
  loses to a0 → family closed. A surprise win would reopen the frontier (keep ∈ {240, 252} within budget).
- Bonus: it pins the slope of damage-vs-keep-count, which is the number the whole program should have
  had up front.

### 3.3 Retained-routing-mass diagnostic across all six existing plans (~1 hour, no GPU)

F216's plan records `route_mass_retained_diag = 0.7276` (f216/BUILD-F216.md). Compute the same
diagnostic for r216/s216/t216/u216/e216 plans from the existing routing counts
(r216/routing-counts-r216.json + plan files) and test whether panel top-1 orders by retained mass.
If it does, future plans get a one-number screen that costs nothing; if it does not, that is worth
knowing before spending GPU time on any more ranking work.

### 3.4 Non-uniform per-layer keep — the original "new way" (loader probe ~1 h, then full pipeline)

Blocked since the a2d era by SGLang's uniform-expert-count requirement. The brief already scopes a
config-only probe (edit config copies of an existing artifact, try to load) that answers whether the
exl3_plain overlay accepts per-layer `n_routed_experts`. Worth doing: F216's result shows *coverage*
(how many experts survive) is the binding variable, and per-layer budgets are the only structural knob
that changes coverage distribution rather than identity. But the size budget still binds: any layer
above 216 must be paid for by another layer below 216 at fixed bpw, so the expected upside is bounded
by §3.2's slope.

### 3.5 Healing (parked on the user; unchanged)

H1 calibration-fold probe ≤12 GPU-h; H2 full 2-5 GPU-days. Its target is precisely the mechanism in
§2.2 (routing fallback with a frozen router), so F216 strengthens its rationale — but nothing in the
F216 result changes its burden of proof, and the honest prior stays low. Note: **router-side adaptation**
(router bias redistribution) would be the cheapest fix for the actual mechanism and is currently out of
scope only because "protected tensors native" is a user constraint — it is a candidate for a scope
decision if §3.1 shows pruning damage dominates.

### 3.6 Do not repeat

- More capture completeness (F216: +15.4% data, ranking moved, quality did not improve).
- More ranking-formula engineering on this quant family (six formulas, 2.84 pp spread, all far below a0).

## 4. What this means for the release

Nothing here changes the release: **a0 stands** (benchmark-backed on every surface measured).
Options §3.1/§3.2 are cheap, use idle hardware, mutate nothing existing, and would each replace an
open inference in the record with a measurement. They need a user go-ahead because the program is
formally closed (PRUNE-PROGRAM-CLOSED.md). §3.3 is free and can be done on request; §3.4's loader probe
is ~1 h of CPU and decides whether the original non-uniform idea is even implementable.