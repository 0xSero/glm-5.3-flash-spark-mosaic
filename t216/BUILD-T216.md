# T216 build (2026-09-13, spark-557f) — sensitivity-only uniform keep-216 re-prune of the 3.05bpw base

Artifact: `valentine@spark-557f.internal:/home/valentine/models/glm53-3p05-pruned-t216/` (96,593,268,599 B on disk; index total 96,485,484,504 B incl. mtp + kpool_aux).
Base: turboderp/GLM-5.3-Flash-exl3 3.05bpw rev 332ab457 (sealed). Serving: container `glm53-t216` (attempt 3, 8c933c7d8a11…), image glm53-flash-sglang-exl3-plain:serve4-pruned (b9cf5753), receipts /tmp/t216-receipts, port 8000.
Reason (queue item 2): S216's score is sensitivity-dominated (corr(sensitivity, ln routes) ≈ 0.09 per layer) and the fresh route capture was thin on generated tokens (generation cut to one prompt). Hypothesis: the ln(routes+1) factor is mostly noise at this capture depth; drop it and rank by K3 sensitivity alone.

## Ranking and provenance

| Item | Value |
|---|---|
| Plan | `t216-plan.json` (0444), sha256 `4e4d8973661b414cb0f2bc8496e756d41b25f1dd6a8e2299d5bcb97c574f492a` |
| Score | sensitivity[l][e] alone; top-216 kept per layer (ties lower id); routes EXCLUDED from the score by design |
| Sensitivity source | inherited verbatim from the sealed S216 plan's `sensitivity_by_layer` (S216 plan sha `4aa4018b…`, which pins k3-projection-errors.json `89a26929…`); independent reduced copy `s216/sensitivity.json` cross-checked at max \|diff\| 0.0 over all 12,096 experts (layers 3..44) |
| Routes (stats only) | same sealed `routing-counts.json` (sha `1f0e98c8…` matches the S216 pin); used only for retention/overlap statistics |
| Totals | 3,024 removed / 9,072 kept; route-mass retention 74.80% (S216 80.0%, U216 86.0%); sensitivity retention 79.26%; no zero-route experts pruned anywhere |
| Keep-set overlaps | mean overlap with S216 keep sets 202.7/216 (sensitivity-dominated, as expected); mean overlap with U216 152.4/216 |
| MTP | layer 45 kept verbatim, 288 experts (mtp.safetensors byte-identical) |

Builder: `t216/build_t216_plan.py` (guards: S216 plan embedded-sha reproduction, sensitivity equality, routing sha pin). Lower route-mass retention than both mass-aware rankings is expected — sensitivity-only tolerates dropping high-route experts; that is exactly what the G4 test measures.

## Build

| Step | Result | Receipt |
|---|---|---|
| Pack | `prune_u216.py` unchanged, 5 workers, 15 shards; config n_routed_experts=216, dynamic_container.point_id T216 | t216/pack.log, artifact manifest.json |
| Index | augmented with mtp.safetensors (3,508 tensors) + kpool_aux.safetensors (22); total 96,485,484,504 B — identical to U216/S216 as required by the shared removal count | t216/augment.log |
| G2 verify | PASS: 111,736 tensors byte-compared to source, 0 problems, 140.4 s | t216/verify.log |
| Census, stock | exit 2 with exactly 42 expected "layer L: 216 experts, expected 288" deviations; bits {3: 28080} = (9,072 + 288 MTP) × 3; dense 440/440 | t216/census-pruned-t216-stock.json |
| Census, plan-aware | contract_ok_plan_aware = true, residual [], moe native | t216/census-pruned-t216-plan-aware.json |

## Serve (three attempts; all logs and receipts preserved on 557f in t216/)

| Attempt | Result |
|---|---|
| 1 (25775e7d) | CRASHED ~15 s in: overlay `config.py::_load_census` requires `$EXL3_PLAIN_CENSUS` (default /receipts/exl3-plain-census-2p05.json) — the served census receipt was not part of my build scripts (U216/S216 had it pre-staged in /tmp receipts). Logs: `serve-attempt1-crash.log`. |
| 2 (f555d14f) | Receipt produced by `make-served-census-receipt.py`: full plan-aware census with the 42 expected deviations cleared (problems → [], verdict.contract_ok → true, top-level contract_ok, provenance note; plan sha `4e4d8973…`, index sha `73f25f83…`; raw plan-aware census preserved untouched). Receipt landed ~5 s before the container's census check (same name as the failed guard asserted `problems_expected_by_plan` was a count; it is the 42-string list — fixed). READY but **max_total_num_tokens 258,240 < 262,144 required**. Container cmd/env/shm/devicerequests byte-equal to glm53-s216 (which measured 270,016 at equal avail mem) ⇒ run-to-run memory-profiling variance, not config. Logs: `serve-attempt2-short-kv.log`. |
| 3 (8c933c7d) | **READY: max_total_num_tokens 302,656 ≥ 262,144**; load 330.70 s, quant exl3 bits 3. Greedy smoke PASS: exact reply "smoke ok", finish_reason=stop, matched_stop 154827. |

557f serving slot: `glm53-s216` was stopped with `docker stop` (container preserved, NOT removed; inspect + ports captured in `s216-container-inspect-pre-stop.json` / `s216-ports-pre-stop.txt` before the stop). `glm53-t216` now owns the slot.

## G4 (from spark-2384, `g4/run-t216.sh` + `run-t216-g4-fixup.sh`)

Panel candidate 2026-09-13T23:59:07Z (exit 0), compare attempt 1 FAILED 00:04:43 (FileNotFoundError:
my symlink step ran with container cwd=/ so `../panel/...` created one dangling literal link —
preserved in `g4-t216.out`); samples ran 00:04:54 → complete (20 rows); fixup 01:26:44: corrected
absolute-path symlinks (32 linked), compare exit 0, summary exit 0, T216_G4_FIXUP_DONE 01:27:03Z.

| point | top-1 | KL lower bound mean / p50 / p95 / p99 / max | PPL | samples |
|---|---|---|---|---|
| A0 2.05bpw unpruned | 78.90% | 0.384 / 0.104 / 1.80 | 4.287 | 15/20 |
| U216 3.05bpw massmax keep-216 | 66.83% | 0.905 / 0.294 / 4.06 | 7.371 | 12/20 |
| S216 3.05bpw sensitivity×log-routes keep-216 | **68.02%** | **0.801** / 0.305 / 3.43 | **6.597** | **12/20** |
| **T216 3.05bpw sensitivity-only keep-216** | 66.50% | 0.855 / 0.347 / 3.58 / 6.28 / 13.88 | 6.997 | 6/20 (sd 0/5, gen 1/5, reasoning 1/5, retrieval 4/5) |

Per row (32): T216 vs S216 top-1 5/32, KL 4/32, NLL 5/32 — S216 dominates. T216 vs U216: top-1
12/32, KL 15/32, NLL 15/32. Receipts: `/home/sero/w2port/out/t216-g4-{panel,samples,summary}.json`;
local copies in t216/ (including `g4-t216.out` with the failed compare attempt).

**Conclusion — hypothesis REFUTED.** Dropping the route factor made every panel metric worse than
S216 (KL 0.801 → 0.855, top-1 68.02 → 66.50, PPL 6.597 → 6.997) and collapsed samples to 6/20
(reasoning 5/5 → 1/5, retrieval 5/5 → 4/5, structured_decode 1/5 → 0/5). The thin route capture
(one generation prompt, ~8.6k generated tokens) still carried real ranking signal. Notable sample
failures: sd-json300-200000 hit the non-stop length cap again (same stop-behaviour loss family as
S216's structured_decode failures). Ranking quality order so far: S216 > T216 > U216 on panel;
S216/U216 tie 12/20 on samples, T216 6/20. Next lever (queue item 3): a denser route capture —
the route signal helps, so more of it should help more.
