# R216 build (2026-09-14, spark-557f) — sensitivity × ln(routes+1) with the DENSER R216 route capture

Artifact: `valentine@spark-557f.internal:/home/valentine/models/glm53-3p05-pruned-r216/` — 96,593,258,410 B on disk (du, receipt manifest.json); manifest total_output_bytes 96,580,353,416 B, index_total_size (kept tensors incl. mtp + kpool_aux) 93,611,759,540 B; finished 2026-09-14T02:15:16Z, g2_verified=true.
Base: turboderp/GLM-5.3-Flash-exl3 3.05bpw rev 332ab457 (sealed). Serving: container `glm53-r216`, image glm53-flash-sglang-exl3-plain:serve4-pruned (b9cf5753), receipts /tmp/r216-receipts, port 8000.
Reason (queue item 3): T216 refuted "routes are noise" (KL 0.855 vs S216 0.801, samples 6/20 vs 12/20), so keep S216's score form but densify the thin part — the generated-token routing (sealed capture: 1 partial prompt, ~8.6k tokens; S216 generation had been cut because the first uncapped prompt ran past 8k reasoning tokens at ~10 tok/s).

## Route capture densification

| Item | Value |
|---|---|
| New capture | `r216/capture_gen_r216.py` on spark-2384, recorder server `glm53-exl3-plain-serve-rec` (unpruned 2.05bpw): GEN_PROMPTS[1..15] verbatim from capture_routing.py (prompt 0 already in the sealed partial capture), temperature 0.7 / top_p 0.95, **hard max_tokens=2048 cap per prompt**; 8 batches of 2; receipt `r216/r216-gen-capture.json`, log `r216/r216-gen-capture.out` |
| New tokens | 26,560 completion tokens (+ each batch's prompt prefill also routes); several prompts hit the 2048 cap (finish=length) — routing harvest, not text quality |
| Combination | `r216/combine_reduce.py` (in docker image with torch; system python has none): full recomputation from ALL dumps on disk — 8 sealed batches (prefill-000..006 + generate-000-partial) + 8 new (gen2-000..007) → `routing-counts-r216.json` sha `ee8858f039811d22b67775b752c10755aeae8b462fc6874cf0ef6a989e1e94d4` |
| Sanity | totals 77,322,672 → 86,453,472 routes (+11.8%); zero-route experts 0; min per-expert count 113; layers 3..44 (recorder does not record MTP 45 — consistent with the plan schema) |
| Provenance | embedded in the routing file: sealed S216 counts sha `1f0e98c8…`, both capture receipts' shas, batch lists |

## Ranking

| Item | Value |
|---|---|
| Plan | `r216/r216-plan.json` (0444), sha256 `62300f562bd7d239bdf28e5169ccc1114f4be6bd94f1b35fc80c9a0c84a3baf3` |
| Score | sensitivity × ln(routes+1), routes from the denser R216 capture; top 216 per layer kept, ties lower id |
| Sensitivity source | identical chain to T216: sealed S216 plan's sensitivity_by_layer (pins k3 `89a26929…`), reduced copy cross-checked max \|diff\| 0.0 |
| Totals | 3,024 removed / 9,072 kept; route-mass retention 79.72% (S216 80.0%, U216 86.0%, T216 74.8%); sensitivity retention 78.98% |
| Keep-set overlaps | mean overlap with S216 **213.2/216** — the denser capture shifts only ~2–3 experts per layer vs S216; mean overlap with U216 158.1/216 |
| MTP | layer 45 kept verbatim, 288 experts |

Expectation recorded BEFORE G4: R216 should land very close to S216 (the ranking is stable w.r.t. capture depth at this density); the test's value is measuring that stability, not expecting a large jump.

## Build

| Step | Result | Receipt |
|---|---|---|
| Pack | `prune_u216.py` unchanged, 5 workers, 15 shards; config n_routed_experts=216, dynamic_container.point_id R216 | r216/pack.log, artifact manifest.json |
| Index | augmented with mtp.safetensors (3,508) + kpool_aux.safetensors (22); total 96,485,484,504 B | r216/augment.log |
| G2 verify | **PASS — 111,736 tensors byte-compared to source origin, 93,611,759,540 B, problems [] , 136.6 s, finished 2026-09-14T02:17:33Z** | r216/receipts/verify.log |
| Census | stock: 42 problems, **42/42 match the expected form "layer L: 216 experts, expected 288"** (regex-checked, residual 0); plan-aware: same 42 expected deviations; served receipt `exl3-plain-census-2p05.json` index_only=false, verdict.contract_ok=true, problems=0 — validated by the overlay at launch | r216/receipts/census-pruned-r216-{stock,plan-aware}.json, r216/receipts/exl3-plain-census-2p05.json |
| Serve receipt | pre-staged BEFORE first launch (T216 attempt-1 trap): `make-served-census-receipt-r216.py` → /tmp/r216-receipts/exl3-plain-census-2p05.json | r216/exl3-plain-census-2p05.json |

## Serve

Two attempts (KV-profiling variance pattern, same as S216/T216):

| Attempt | Result | Receipt |
|---|---|---|
| 1 (02:28:44Z) | READY but `max_total_num_tokens=253,248` < 262,144 required ⇒ relaunch | r216/receipts/serve-attempt1-short-kv.log |
| 2 (02:35:52Z) | READY, `max_total_num_tokens=321,664` ≥ 262,144; greedy smoke PASS; container `glm53-r216` owns 557f's slot (glm53-t216 stopped preserved) | docker logs glm53-r216 (KV line quoted), 557f |

## G4 (from spark-2384, `g4/run-r216.sh`, absolute-path teacher-row symlinks from cwd=out dir)

**R216 is the new leader.** Panel complete 2026-09-14T02:44Z, samples 03:58Z, summary written 2026-09-14T03:58:49Z, marker `R216_G4_DONE 2026-09-14T03:58:50Z`. All receipts copied to Mac `r216/receipts/`.

| Metric | R216 | (S216, previous leader) |
|---|---:|---:|
| top-1 agreement | **68.5286%** | 68.0187% |
| KL lower bound mean | **0.77709** | 0.80122 |
| KL p50 / p95 | 0.29003 / 3.36459 | 0.30486 / 3.42855 |
| KL p99 / max | 5.99003 / 14.21859 | 6.09066 / 12.83936 |
| KL top-k renorm | 0.76834 | — |
| Candidate PPL | **6.44689** | 6.5965 |
| Teacher PPL recomputed | 3.19953 (= reference ✓) | 3.1995 |
| Coverage: teacher mass / top1-in-topk | 0.99236 / 0.99959 | — |
| Teacher reproduction | top1_exact_all_rows=true, max NLL Δ 0.00511 | — |
| Samples | **12/20** (sd 1/5, gen 1/5, reasoning 5/5, retrieval 5/5) | 12/20, same profile |

Samples sha 5684425f… matches the pre-registered expected hash. All 8 failures are the **identical S216 failure set** (5× sd-json300, 3× gn-tips240, all `finish_reason_stop=false` token-cap stops) — the ~2–3 experts/layer ranking perturbation did not change failure behaviour at all, matching the pre-registered expectation of ranking stability.

Per-row wins over 32 panel rows: **R216 vs S216 top-1 23/32, KL 26/32, NLL 27/32**; R216 vs U216 18/32, 23/32, 22/32. Cross-check: recomputed S216 vs U216 (18/32, 21/32, 22/32) matches the brief's recorded numbers exactly.

Receipts: `r216-g4-{panel,samples,summary}.json` (2384 /home/sero/w2port/out/, copied to Mac r216/receipts/; summary sha-references panel b83cf5e1… and samples 8c03ed9a…/5684425f…).
