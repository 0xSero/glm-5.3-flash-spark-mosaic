# S216 build (2026-09-13, spark-557f) — sensitivity-ranked uniform keep-216 re-prune of the 3.05bpw base

Artifact: `valentine@spark-557f.internal:/home/valentine/models/glm53-3p05-pruned-s216/` (96,593,256,540 B on disk; index total 96,485,484,504 B incl. mtp + kpool_aux).
Base: turboderp/GLM-5.3-Flash-exl3 3.05bpw rev 332ab457 (sealed). Serving: container `glm53-s216`, image glm53-flash-sglang-exl3-plain:serve4-pruned (b9cf5753), receipts /tmp/s216-receipts, port 8000.
Reason: U216 (massmax_domain routing-mass ranking from the 0xSero Q4 REAP observations) lost 12 top-1 points vs A0; hypothesis was that mass-only rankings inflated the loss.

## Ranking

| Input | Source | Notes |
|---|---|---|
| sensitivity[l][e] | `k3-projection-errors.json` (12,096 experts, layers 3..44, sha 89a26929…) | mean of gate/up/down paired K3 proxy errors; K2 proxies rank identically (within-layer r = 0.99993) |
| routes[l][e] | fresh top-8 routing counts, SGLang expert-distribution recorder (stat mode) | captured from the **unpruned 2.05bpw** on spark-2384 (see deviation below); 222,285 prefill tokens (160 records, 16 per domain, 0xSero/reap-calibration-data-v1 @115e754a, seed 20260913) + ~8.6k generated reasoning tokens; every expert in every layer routed ≥700 times |
| score | sensitivity × ln(routes+1); top 216 per layer kept, ties lower id | corr(sensitivity, ln routes) ≈ 0.09 per layer → the score is dominated by sensitivity |

Plan sha256 `4aa4018b9e9892bec7dc10164cd74c89298f3b8661eb08a5e291c1066a2dd897` (`s216-plan.json`, 0444): 3,024 removed / 9,072 kept, route-mass retention 80.0% (min 74.4%, max 85.7% per layer; U216 massmax retained 86.0%), sensitivity retention 79.0%, mean overlap with the U216 keep sets 158.5/216 (146–170). MTP layer 45 kept verbatim (288 experts).

**Deviation from the brief:** the brief said the 3.05 was serving on 557f. It was not — 557f served U216 (pruned), and the unpruned 3.05 (~127 GB of weights) cannot fit a 121 GB Spark at all. Routing was therefore captured from the unpruned 2.05bpw (all 288 experts present; the router/gate weights are unquantised and identical across bpw branches). Generation prompts were cut to one after the first prompt ran past 8k reasoning tokens at 10 tok/s (uncapped generation of 16 prompts would have taken hours); the routing of the generated tokens was harvested.

## Build

| Step | Result | Receipt |
|---|---|---|
| Pack | `prune_u216.py` unchanged (plan schema identical), 5 workers; config n_routed_experts=216, dynamic_container.point_id S216 | receipts/pack.log, receipts/manifest.json |
| Index | augmented with mtp.safetensors + kpool_aux.safetensors | receipts/augment.log |
| G2 verify | PASS: 111,736 tensors byte-compared to source, 0 problems, 144.9 s | receipts/verify.log, receipts/g2-receipt.json |
| Census | stock exit 2 = 42 expected 216-vs-288 deviations; plan-aware contract_ok=true, residual [], moe native, dense 440/440 | receipts/census.log; served copy receipts/exl3-plain-census-2p05.json (problems cleared, verdict.contract_ok=true, as for U216) |
| Serve | READY in ~6 min, max_total_num_tokens 270,016 (≥ 262,144); greedy smoke coherent | — |

## G4 (from spark-2384, `g4/run-s216.sh`, teacher rows cached in panel/, candidate rows panel-s216/)

| point | top-1 | KL lower bound mean / p50 / p95 | PPL | samples |
|---|---|---|---|---|
| A0 2.05bpw unpruned | 78.90% | 0.384 / 0.104 / 1.80 | 4.287 | 15/20 |
| U216 3.05bpw massmax keep-216 | 66.83% | 0.905 / 0.294 / 4.06 | 7.371 | 12/20 |
| **S216 3.05bpw sensitivity×log-routes keep-216** | **68.02%** | **0.801 / 0.305 / 3.43** | **6.597** | 12/20 (structured_decode 1/5, generation 1/5, reasoning 5/5, retrieval 5/5) |

Samples: same 12/20 total as U216 but a different profile — reasoning 5/5 (U216 3/5) and retrieval 5/5 (U216 3/5; both 262k rows now admitted, max_total_num_tokens 270,016), but structured_decode 1/5 (U216 5/5): every sd-json300 row and every tips240 row hit the 2,048-token cap (`finish_reason=length`, the json300 rows keep counting past 300), i.e. S216 lost the stop behaviour on long enumerations.

Per row: S216 beats U216 on 18/32 (top-1), 21/32 (KL), 22/32 (NLL); loses to A0 on 32/32. Receipts on 2384: `/home/sero/w2port/out/s216-g4-{panel,samples,summary}.json`; local copies in receipts/.

Conclusion: the better ranking recovers ~1.2 top-1 points and ~0.1 nats of the U216 loss; the remaining ~11 points / ~0.4 nats vs A0 are the cost of removing 25% of experts without healing, not the REAP ranking.

Notes
- The G4 out dir is created by docker as root; teacher-row symlinks must be made inside a container (`ln -sf ../panel/teacher-row-*.pt`) or the compare phase fails with FileNotFoundError.
- spark-2384 now runs `glm53-exl3-plain-serve-rec` (same 2.05 serve4 launch + recorder flags, dumps to /home/sero/w2port/routing); the old `glm53-exl3-plain-serve` container was removed. Recorder is idle unless /start_expert_distribution_record is called.
- 557f: `glm53-a2d-u216` container removed; the U216 artifact dir remains on disk (96.6 GB) — delete it if space is needed (310 GB free before S216, ~213 GB after).
