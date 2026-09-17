# F216 — full-observation REAP prune (user goal, 2026-09-15) — plan

User objective (verbatim intent): "do another prune on the exl3 3bpw in a new way, maybe non-uniform
pruning (use the reap methodology in the original repo for pruning and use the full observation dataset)".

Status: **PLAN + prep analysis complete; capture blocked on ONE user action (de5c power-on).**

## 1. What "full observation dataset" means (measured, receipted)

- `sealed-observations/observations.json` = **SEALED_PARTIAL**: 21,248/23,088 records,
  32,602,850/37,328,459 tokens (`seal.json` sha 4fd8de1e…/b2e36095…).
- Token manifest `ours-original-records-v1` (0xSero/reap-calibration-data-v1 @115e754a, sha 421d8c14…,
  16,384-token records, 361 shards × 64): missing = **science 195 records (0 captured)** +
  **cuda 1,645 of 2,000 missing** = **1,840 records / ~4.73M tokens** (shards ~332–361).
- E216 was scored on the PARTIAL set; completing the set changes the REAP scores by construction.
- Capture model identity: **0xSero/GLM-5.3-Flash-EXL3-Q4 @ d0b9301a** (sha256 6a6357fd…).

## 2. Capture architecture (read from the pipeline code — glm53_record_observation_pipeline.py, 314 lines)

- **Pipelined TWO-NODE protocol**: rank0 loads its model stage, forwards, sends (ids, hidden) to
  rank1; rank1 completes the record and merges. Collectives: gloo init, `all_gather_object` peer
  identity validation ("neither peer trusts metadata alone"), `broadcast_object_list`, NCCL group.
  Each rank loads HALF the Q4 model (~65 GB) — **a single node cannot hold both stages (~129 GB),
  so single-node capture is impossible by design.**
- **Resume is first-class**: rank0 validates existing `shard-*/receipt.json` in `--output-root` and
  `phase_plan` processes ONLY uncommitted shards. The old run-root (332 committed shards) lives at
  `de5c:/home/valentine/glm53-full-observations-20260907/observations-q4-original-records-v1/records-8d8dd5ec0a6608f97d88b158/`.
- Availability verified 2026-09-15: Q4 model + observer image (`glm53-full-observer-fla:20260907`,
  sha-pinned in run_glm53_record_observations.sh) + old run-root exist ONLY on de5c (2822/2384/557f
  checked — absent). 2822 has the full src mirror (`work/glm53-single-spark-release-20260911/observer-src/reap/`)
  + token manifest + 2.1 TB free.
- **⇒ BLOCKING USER ACTION: power on de5c.** (Never attempted remotely per no-reboot constraint.)
- **de5c-FREE PATH REJECTED 2026-09-15T14:xxZ (receipted math):** without the old run-root,
  `committed=[]` → ALL 361 shards re-captured = 37.3M tokens ≈ 4–5 days at the measured rate —
  not viable. The resume property (skip committed shards) is what makes this goal affordable, and
  the committed receipts exist only on de5c. Q4 model therefore also downloading to 2822 from HF
  (187.6 GB, 509 files, public repo, revision-pinned) so rank1 staging does not wait on the de5c
  copy; 2384 staging NOT needed (de5c-free path rejected; release serving stays untouched).

## 3. Execution sequence (original protocol verbatim)

1. **[USER]** Power on de5c.
2. Stage 2822 as rank1: Q4 model dir de5c→2822 direct rsync (306 MB/s ≈ 8 min) with whole-dir sha
   verification; observer image `docker save|ssh|load` over the same link; src already present.
3. **2-node resume capture**: de5c rank0 (old run-root, `--qualify-then-collect`, chunk 512) +
   2822 rank1 → processes ONLY the ~29 missing shards (~4.7M tokens). **CORRECTED ESTIMATE
   2026-09-15T14:xxZ: ~12–14 h** — measured from the ORIGINAL run's timeline (project 2026-09-07
   → seal 2026-09-11T14:34Z ≈ 3.5–4 days for 32.6M tokens ≈ 94–107 tok/s aggregate through the
   serial pipelined pair; records are NOT processed in parallel). An earlier 30–90 min estimate
   in this plan was wrong; kept here as a receipted correction. New receipts land IN the
   existing run-root (pipeline's atomic receipt write, foreign-range guard active).
4. **Seal FULL**: `seal_glm_observations.py` over the COMPLETE run-root (network-none container,
   CPU-only, per queue_seal_observations.py pattern) → `sealed-observations-full/` (new seal; the
   partial seal stays untouched). full_suite_pass must be True.
5. **F216 plan**: `f216/build_f216_plan.py` (adapted from e216/build_e216_plan.py; same score =
   weighted_ean_sum / expert_frequency = the repo's `reap`; require the FULL seal state; uniform
   keep-216; MTP layer 45 verbatim; provenance pins on the new seal + repo files). Overlap
   diagnostics vs E216/R216/S216/U216; **pre-registered build rule: build if mean overlap with E216
   < ~213/216** (same rule that triggered the E216 rebuild).
6. **Build** on 2822 (idle, 2.1 TB free; sealed 3.05bpw base copied 557f→2822 or built from the
   sealed base there) via the 8-step candidate pipeline verbatim → serve → G4 (teacher = a0-bench
   on 2384) + quick MMLU + GPQA MC.

## 4. Non-uniform pruning — analysis (the "maybe" in the goal)

- The repo's `prune.py` is **uniform per-layer** (one `n_experts_to_prune`, topk per layer). Its
  only built-in non-uniform elements: `perserve_super_experts` (saliency=inf on super experts;
  excludes the last 25% of layers) and `perserve_outliers` (all layers) — both computable from the
  SAME full observations at zero extra capture cost. Pre-register both score variants; **build the
  repo-default plain reap first (F216)**; a preserve-super variant (F216-S) builds only if F216's
  evidence supports continuing.
- TRUE per-layer variable keep counts are **BLOCKED by the a2d precedent** (SGLang uniform
  expert-count loader requirement; brief: "do not resurrect without a loader path"). A path exists
  in principle (exl3_plain overlay patch exposing per-layer n_routed_experts) but is a separate
  engineering project — out of scope here unless F216's results + user request justify it.

## 5. Gates (pre-registered, binding)

G4 panel fail-fast; then quick MMLU (1,140 docs, seed 1234) vs a0 83.42±1.20 and E216 77.28±1.20;
GPQA MC vs a0 43.43±3.53 / E216 37.88±3.46. Honest prior: E216 (same score, partial data) was the
worst measured point; full data changes scores but the prior is unfavorable — the user reopened the
program knowing this. If F216 fails the same way, the negative result is final with the full-data
caveat removed (the strongest possible closure of the REAP-native line).

## 6. Costs

de5c power-on (user) + ~10 min staging + capture ~12–14 h (2 nodes, overnight) + seal minutes +
plan minutes + build ~1–2 h (2822) + serve + evals ~2 h. Total ≈ 1.5 days wall-clock after the
user powers de5c on. 557f stays idle
(e216 artifact preserved); 2384 release serving + a0 full-MMLU run UNDISTURBED throughout
(one-GPU-job-per-node: capture owns de5c+2822).
