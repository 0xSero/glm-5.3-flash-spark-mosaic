# E216 build (2026-09-14, spark-557f) — REAP-NATIVE score: mean over routed tokens of (renormalized router weight × expert output L2 norm)

Artifact: `valentine@spark-557f.internal:/home/valentine/models/glm53-3p05-pruned-e216/` — **96,580,353,371 B logical (du 94,329,620 KB), 31 files** (15 shards + config + index + receipts). Base: turboderp/GLM-5.3-Flash-exl3 3.05bpw (same sealed base as U216/S216/T216/R216). Serving: container `glm53-e216`, image glm53-flash-sglang-exl3-plain:serve4-pruned (b9cf5753), receipts /home/valentine/e216/ + /tmp/e216-receipts, port 8000.
Reason (queue item 4, driver run 5): the brief's REAP-native item — rank by the reference repo's actual criterion (`reap-strict-serialization` src/reap/pruning_metrics.py:198: reap = mean over ROUTED tokens of ean_norm × active_router_weights, top-k router weights renormalized to sum 1; src/reap/prune.py prunes the LOWEST saliency) computed from the sealed 0xSero observations, NOT from route counts.

## Ranking

| Item | Value |
|---|---|
| Plan | `e216/e216-plan.json` (0444), frozen 2026-09-14T09:31:41Z; **on-disk sha256 = 5cea3a31…, identical Mac↔557f** — the plan SELF-RECORDS sha `37fa10fd…` and the G2 receipt embeds that same self-recorded value; recorded as a self-referential-sha discrepancy, canonical = on-disk file (receipted in EXPERIMENTS.md 2026-09-14T23:15Z) |
| Score | per expert: `weighted_ean_sum / expert_frequency` (sealed observations.json; = the repo's `reap` key) — NO route counts, NO sensitivity in the score |
| Observation source | `sealed-observations/observations.json` schema glm53-original-records-seal-v1, state SEALED_PARTIAL_ORIGINAL_RECORDS: 21,248/23,088 records, 32,602,850/37,328,459 tokens, science domain missing, min expert routes 49,419 (no unseen experts) |
| Pre-registered decision rule (Loop state) | rebuild + eval only if the keep-set plausibly differs from R216's (≥ ~213/216 overlap ⇒ skip) |
| **Decision** | **mean overlap with R216 = 149.6/216 (min 144, max 162) ⇒ REBUILD triggered** — also vs S216 149.3, T216 149.0, U216 180.6 |
| Diagnostics | Spearman(reap_mean, ln routes_fresh) mean −0.141 (min −0.392, max +0.044, 42 layers) — NEGATIVE: REAP keeps high gate×activation experts over frequently-routed ones; fresh route-mass retention of the E216 keep-set 72.96% (R216 79.72%, S216 80.0%, U216 86.0%, T216 74.8%) |
| Totals | 3,024 removed / 9,072 kept (per-shard removed counts in pack.log sum to 36,288 tensor-level removals = 3,024 × 12); MTP layer 45 verbatim (288 experts) |
| Provenance | plan inputs_sha256 pins observations.json, routing file, all four comparison plans, and the two reference-repo files (prune.py, pruning_metrics.py); score components (weighted_ean_sum, expert_frequency) embedded so the ranking is recomputable from the plan alone |

## Build

| Step | Result | Receipt |
|---|---|---|
| Pack | 15 shards, 5 workers, 20:18:42Z→20:20:08Z (85.4 s), output 96,580,353,371 B; per-shard sha256 logged; `g2_verified:false` in pack.log is the PRE-verify state (G2 runs as its own step) | e216/receipts/pack.log (Mac + /home/valentine/e216/pack.log) |
| Index | augmented with mtp.safetensors + kpool_aux.safetensors (extra index tensor bytes 2,873,724,964; index total 96,485,484,504 B) | e216/receipts/augment.log + build.out (BUILD_DONE 2026-09-14T20:21:46Z, all exits 0) |
| G2 verify | **PASS — `pass: true`, 111,736 tensors / 93,611,759,540 B byte-verified, problems [], 98.1 s**; expert_count_by_layer 3..44 = 216 each | e216/receipts/g2-receipt.json (in-artifact copy) + e216/receipts/verify.log |
| Census | plan-aware **contract_ok=True, moe native=True, dense native=440/440**; stock contract_ok=False with the 42 EXPECTED-by-plan deviations, residual=[] missing=[]; expert layers 3..45 (43 incl. MTP), codebook=mul1; exit=0 | e216/receipts/census.out + census-pruned-e216-{stock,plan-aware}.json |
| Serve receipt | pre-staged BEFORE first launch (T216 attempt-1 trap): `make-served-census-receipt-e216.py` → exl3-plain-census-2p05.json (contract cleared) | e216/receipts/exl3-plain-census-2p05.json |

## Serve (container `glm53-e216`, port 8000)

- **Attempt 1**: READY-path but max_total_num_tokens=218,176 < 262,144 required (KV-profiling variance — known from S216/T216/R216; one relaunch recovered every time). Preserved: e216/receipts/serve-attempt1-short-kv.log.
- **Attempt 2 (UP)**: launched by the orchestrator with mem-fraction **0.96 (user directive)**; engine KV **344,256 tokens** ≥ 262,144 gate passed 2026-09-14T20:38:01Z; `max_running_requests` self-capped to **1** by hybrid-mamba slots (140.78 MB/req; harmless for sequential G4); greedy smoke try-1 empty / try-2 OK 20:40:08Z.
- Orchestrator KV-gate blind spot FIXED mid-flight: run-serve script didn't route engine logs into serve.out → `docker logs -f glm53-e216 >> serve.out` bridge; receipted in e216/orchestrator.log (Mac) — relaunch attempts while the engine was already UP were the symptom.
- Full engine log preserved on 557f only (4.0 MB, not copied): /home/valentine/e216/serve.out; serve-attempt2.out (attempt-1 verdict + relaunch) copied to Mac.

## G4 (from spark-2384, `g4/run-e216-g4.sh`, absolute-path teacher-row symlinks)

| Metric | Value |
|---|---|
| top-1 agreement | **66.4143%** (65,504 positions; teacher top-1 reproduced exactly, max_nll_sum_abs_delta 0.0051) |
| mean KL lower bound | **0.89796** (p50 0.31931 / p95 3.94230 / p99 6.99787 / max 16.42341) |
| mean KL top-k renorm | 0.87213 |
| candidate PPL | 7.23760 (teacher recomputed 3.19953 = reference) |
| samples | **16/20 PASS** (samples sha256 5684425f… matches expected 5684425f… — the only panel surface where E216 beats R216's 12/20) |
| Receipts | 2384:/home/sero/w2port/out/e216-g4-{summary,panel,samples}.json; Mac copies e216/receipts/e216-g4-*.json |

Estimator note (from the summary's method_labels): the KL column is the LOWER BOUND on the coarsened partition (teacher top-256 ∩ served top-2048 + rest) — not identical to the older reference rows' exact full-vocab KL; same panel, same teacher.

## Verdict (2026-09-14T23:15Z orchestrator + 2026-09-15 benchmarks)

- **G4 panel: WORSE than the deleted R216 on every panel metric except samples** — top1 0.6641 vs 0.6853, KL_lb 0.8980 vs 0.7771, KL_topk 0.8721 vs 0.7683, PPL 7.238 vs 6.447; samples 16/20 vs 12/20.
- **Quick MMLU like-for-like (1,140 docs, seed 1234)**: E216 77.28% ± 1.20 vs a0 83.42% ± 1.20 → **−6.14 pp, ~3.6σ** (receipts results/{e216,a0}-quick/bench-*/results_*.json).
- **GPQA MC (198 docs)**: E216 37.88% ± 3.46 vs a0 43.43% ± 3.53, r216 40.91% ± 3.50 → worst of the three (receipt results/e216-gpqa-mc/bench-e216-gpqa-mc/).
- CONCLUSION: REAP-native saliency ranks BELOW route-mass-informed selection on teacher agreement, and BOTH keep families lose to the unpruned q2 reference on benchmarks. a0 = benchmark-backed release config pending the a0-fallback vs healing fork (user decision; healing/PLAN.md written 2026-09-15, see that file).
