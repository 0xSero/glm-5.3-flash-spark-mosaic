# U216 build (2026-09-13, spark-557f) — uniform keep-216 rebuild of A2d

Artifact: `valentine@spark-557f.internal:/home/valentine/models/glm53-3p05-pruned-u216/` (96,592,562,975 B on disk; index total 96,485,484,504 B incl. mtp + kpool_aux).
Base: turboderp/GLM-5.3-Flash-exl3 3.05bpw rev 332ab457 (sealed). Image for census: ghcr.io/0xsero/glm53-flash-exl3-plain@sha256:85cb3fa8… (id c65c840f1908).
Reason: the variable-count A2d (plan 6654d8d3) failed the SGLang uniform-expert-count requirement; artifact removed, clean rebuild.

| Step | Result | Receipt |
|---|---|---|
| Plan | uniform keep 216 x 42 MoE layers (3..44) → 3,024 removed / 9,072 kept; within-layer massmax_domain top-216; matches sealed keep_maps["216"] per layer; pooled-mass retention 86.01%; sha256 3e3596a8378df519fd95c5f6ec5dbaa8805b10f559a2a060e2adc8ad2b7b09a5 | u216-plan.json (0444) |
| Pack | 88.5 s, 5 workers; routers sliced to 216 rows; contiguous renumber; config: n_routed_experts=216, routed_experts_per_layer (all 216), retained_expert_ids_by_layer, dynamic_container{point_id U216, uniform_keep 216, mtp_layer} | receipts/pack.log, receipts/manifest.json |
| Index | augmented with mtp.safetensors (3,508 tensors) + kpool_aux.safetensors (22); shards-only copy kept as model.safetensors.index.shards-only.json | receipts/augment.log |
| G2 verify | PASS: 111,736 tensors byte-compared to source origin, 0 problems; index total == source + 2,873,724,964 − 3,024 × (9,474,060 + 8,196) | receipts/g2-receipt.json |
| Census, stock | exit 2 with exactly 42 problems = "layer L: 216 experts, expected 288" (overlay EXPERTS=288 hard-code); expert layers 3..45 (43); bits {3: 28080} = (9,072 + 288 MTP) × 3; moe native; dense 440/440 | receipts/census-pruned-u216-stock.json |
| Census, plan-aware | contract_ok_plan_aware = true; residual = []; missing = [] | receipts/census-pruned-u216-plan-aware.json |

Notes
- MTP layer 45 is byte-verbatim with **288 experts** (no sealed saliency exists for layer 45). With n_routed_experts=216, a loader that builds the nextn/MTP MoE from config will expect 216 and find 288 expert tensors in mtp.safetensors. Fine if MTP/speculative decoding is not loaded; otherwise MTP needs either pruning to 216 (needs a ranking) or a per-layer override.
- Stock exl3_plain overlay still hard-codes EXPERTS=288 (contract.py); census passes only plan-aware. The n_routed_experts=216 config is what SGLang's uniform loader reads.
- Old A2d tools/receipts remain at ~/a2d on 557f and ../a2d locally; A2d artifact dir is gone.
