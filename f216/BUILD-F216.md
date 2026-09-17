# F216 build (stage 3, 2026-09-16, spark-2822) — REAP-native on the FULL seal

Artifact: `sero@spark-2822.internal:/home/sero/models/glm53-3p05-pruned-f216/` — **96,593,390,114 B (du -sb), 30 files** (15 shards + config + index + shards-only index + mtp + kpool_aux + plan.json 0444 + g2-receipt.json + manifest.json + aux). Base: turboderp/GLM-5.3-Flash-exl3 3.05bpw (same sealed base as U216/S216/T216/E216). Census/serving image lineage: ghcr.io/0xsero/glm53-flash-exl3-plain `2p05-sglang-mul1-r1` = c65c840f1908 (RepoDigest 85cb3fa8… — the same image the E216 census ran on).

Reason (stage 3 of the F216 chain, driver-f216-10): E216's plan was ranked on the PARTIAL seal (21,248/23,088 records, science domain missing, min routes 49,419). Stage 1 completed the original-record capture (29/29 shards, merge-sealed 2026-09-16T00:38Z: 23,088 records / 37,328,459 tokens, min expert routes 49,895) and stage 2 re-ranked REAP-native on the FULL seal → mean overlap with E216's keep-set only 149.52/216 ⇒ rebuild triggered (E216 itself rebuilt on a ≥~213 threshold).

## Ranking

| Item | Value |
|---|---|
| Plan | `f216/f216-plan.json` (0444 on 2822), frozen 2026-09-16T00:55:12Z, on-disk sha256 `342b840d1fd709f364d7f6ac4a2771b60afaaed6558ac0a4ca178d53dfb33590` — identical Mac↔2822 (shasum at ship time) |
| Score | repo-verbatim REAP (`weighted_ean_sum / expert_frequency`, pruning_metrics.py:198) on the FULL seal `f216/sealed-complete/observations.json` (sha `beea9f01…`) |
| Keep structure | 42 layers (3..44) × 216 kept; MTP layer 45 absent from keep_by_layer → verbatim 288; `mtp.safetensors` byte-identical copy |
| Overlap diagnostics | mean overlap with E216 149.52, S216 149.24, U216 181.21, T216 149.14; route_mass_retained_diag 0.7276 |
| Build gate | PASSED (stage-2 receipt; score_threshold_first_layer 1.7636) |

## Build

| Step | Result | Receipt |
|---|---|---|
| Base transfer | 557f→2822 node-to-node rsync, 129,546,716,672 B / 34 files, ~21 min; **20 files sha256-identical 557f↔2822** (15 shards + mtp + kpool_aux + index + config + tokenizer; both receipt files hash 80c2a520…) | f216/receipts/f216-base-shas-{557f,2822}.txt |
| Pack | 15 shards, 5 workers, 01:33:39Z→01:34:59Z (**80.3 s**), output 96,580,353,464 B (E216: 96,580,353,371 — Δ93 B header metadata); per-shard sha256 logged; removed sum **36,288** tensor-level = 3,024 experts × 12 (same count as E216 — identical keep structure, different experts); `g2_verified:false` is the PRE-verify state | f216/receipts/build/pack.log |
| Index | augmented with mtp.safetensors (3,508 tensors / 2,862,179,364 B) + kpool_aux.safetensors (22 / 11,545,600 B); index total **96,485,484,504 B — byte-identical total to E216's index** | f216/receipts/build/augment.log |
| G2 verify | **PASS — `pass: true`, 111,736 tensors / 93,611,759,540 B byte-verified (identical counts to E216), problems [], 105.0 s**; expert_count_by_layer 3..44 = 216 each | f216/receipts/build/g2-receipt.json + verify.log |
| Census | stock: 42 PROBLEM lines (layer 3..44: 216 experts, expected 288); plan-aware **contract_ok=True, moe native=True [], dense native=440/440**; 115,266 tensors, 17 shards, codebook=mul1, expert layers 3..45 (43 incl. MTP), bits 3/28,080 tensors; residual=[] missing=[] | f216/receipts/build/census-pruned-f216-{stock,plan-aware}.json |
| Manifest | dynamic-container-manifest-v1, point_id F216, removed_experts 3,024; config dynamic_container stamps plan_sha256 342b840d… + packer_version u216-packer-v1 + mtp_layer {layer 45, 288 experts, kept verbatim} | f216/receipts/build/manifest.json |

## Provenance

- Build tooling pulled verbatim from the E216 build dir on 557f → 2822:/home/sero/work/f216-build/ (sha256: prune_u216.py `8d9eeb2b…`, augment_index.py `7de8d892…`, verify_u216.py `4ee469e7…`, census_plan_aware.py `5d4901a2…`, run-build-f216.sh `8356e821…` — the F216-adapted copy of e216/run-build.sh).
- Census image transferred 557f→2822 by `docker save | ssh | docker load` pipe (31.8 GB); loaded id **c65c840f1908** identical to 557f's — receipted in /tmp/f216-census.log tail.
- Census image deviation from run-census.sh's digest pin: 2822 has the same image by id c65c840f (RepoDigest 85cb3fa8 was 557f's pull digest; E216 census ran on this same image id).

## Execution defects (preserved, not retried silently)

- Plan-aware census attempt 1: failed — `python3: can't open file '/census_plan_aware.py'` (quoting: `$W` empty inside the nested bash -c); relaunched with literal paths → success. Logs: f216-census.log (failure preserved) + f216-census-aware.log (success).
- Stale `docker load` (pid 1299766, started Sep 15, remote mac-client.internal) found sitting on 2822 during the census-image load — not ours to kill; our load completed normally alongside it.

## Serve (container `glm53-f216`, port 8000) — attempt 1 PASSED all gates

- Launched 02:12Z by run-serve-f216.sh (attempt-2 recipe from E216: mem-fraction 0.90, image serve4-pruned b9cf5753, /tmp/f216-receipts mounted with the PRE-STAGED served-census receipt; logs bridge from t0). No attempt-2 needed.
- **KV gate PASSED attempt-1: max_total_num_tokens = 278,528 ≥ 262,144** (self-caps max_running_requests=1, harmless for sequential evals); READY; greedy smoke OK on try 1 ("The capital of France is → Paris."); zero tracebacks/OOM in the engine log; drop_caches loop kept running on 557f through the whole eval (UMA trap).
- Runner receipts: /home/valentine/f216/serve-launcher.log + serve-f216.out (full engine log, 557f); f216/run-serve-f216.sh (Mac).

## G4 (from spark-2384, f216/run-f216-g4.sh mirror of run-e216-g4.sh) — ALL PHASES exit 0, F216_G4_DONE 03:34:27Z

| Metric | F216 | E216 (best REAP-native) | R216 (deleted leader) |
|---|---|---|---|
| top-1 agreement | **65.691%** | 66.414% | 68.53% |
| mean KL lower bound | **0.93888** (p50 0.33735 / p95 4.09987 / p99 7.17421) | 0.89796 | 0.77709 |
| mean KL top-k renorm | 0.91077 | 0.87213 | 0.7683 |
| candidate PPL | **7.56398** | 7.23760 | 6.44689 |
| samples | **11/20** | 16/20 | 12/20 |
| teacher reproduction | top1_exact_all_rows true, max NLL-sum delta 0.00511, teacher PPL 3.19953 = reference | same | same |
| Receipts | panel sha 3ba29328…, samples sha 3490bd9a… (both match embedded); samples sha 5684425f… = expected panel hash | e216-g4-* | r216-g4-* |

## Quick benchmarks (Mac lm-eval local-completions, glm53-f216 endpoint)

- **Quick MMLU like-for-like (1,140 docs = 57×20, seed 1234)**: F216 **78.68% ± 1.17** vs E216 77.28% ± 1.20 → **+1.40 pp (the only gate F216 wins)**; a0 reference 83.42%. Receipts results/f216-quick/bench-f216-quick/results_2026-09-16T01-27-12.json + run-f216-quick.log (validate leg queued ~80 min behind the G4 panel — engine serializes at max_running_requests=1; not a failure, logged).
- **GPQA MC (198 docs, zeroshot)**: F216 **36.87% ± 3.44** vs E216 37.88% ± 3.46, r216 40.91%, a0 43.43% → **gate FAILED (−1.01 pp)**. Receipts results/f216-gpqa-mc/bench-f216-gpqa-mc/results_2026-09-16T01-27-12.json (exit=0 05:27:12Z in results/status.txt).

## Verdict (stage 4 complete, 2026-09-16T05:35Z)

- **F216 is WORSE than E216 on every G4 panel metric** (top1 −0.72 pp, KL_lb +0.041, KL_topk +0.039, PPL +0.33, samples 11 vs 16) and loses GPQA MC (−1.01 pp); quick MMLU is the only win (+1.40 pp, on a panel where E216 was already −6.14 pp vs a0).
- **ANSWER to the F216 question: completing the observation dataset (science 195 + cuda 2000, min routes 49,895 vs 49,419) did NOT improve the REAP-native ranking.** Capture completeness is not why the REAP-native family trails route-mass-informed selection on the panel; the ranking changed (149.52/216 overlap with E216) but the direction is unchanged and slightly worse. Both keep families remain below the unpruned q2 reference on benchmarks. **Benchmark-backed release remains a0; no release change from F216.** The a0-fallback vs healing fork stays parked with the user (healing/PLAN.md).
- 2822 drop_caches loop STOPPED with receipt 05:32:13Z (pid 1420484, own helper job); 557f drop loop kept while glm53-f216 was up — **teardown COMPLETE 2026-09-16T09:43:21Z** (user "keep going"): glm53-f216 stopped/removed (exit 0 both), loop pid 672379 killed + verified, no glm53 containers on 557f, engine log preserved (serve-f216.out 4.2MB), artifact untouched (relaunchable via run-serve-f216.sh). Receipt: f216/receipts/f216-serve-teardown-20260916.txt (sha c97d0341…).

