# A2d build (2026-09-13, spark-557f)

Artifact: `valentine@spark-557f.internal:/home/valentine/models/glm53-3p05-pruned-a2d/` (96,669,143,443 B on disk; index total 96,561,342,552 B incl. mtp + kpool_aux).
Base: turboderp/GLM-5.3-Flash-exl3 3.05bpw rev 332ab457 (sealed, DOWNLOAD-SEAL.json 2026-09-13T07:55:58Z).
Image: ghcr.io/0xsero/glm53-flash-exl3-plain@sha256:85cb3fa86d31a781b94dcf10ee168adf096cfeaac14d2f1e6c560504e58e4eed (id c65c840f1908).

| Step | Result | Receipt |
|---|---|---|
| Census, sealed base (stock index) | expert bits {3: 36288}, codebook mul1, 148,024 tensors, 15 shards, dense 431/431 native; only problem: layer 45 absent (mtp.safetensors not in index) | receipts/census-base-3p05-stock.json |
| Census, sealed base (index + mtp + kpool_aux) | contract_ok, layers 3..45 (43), bits {3: 37152}, moe native, dense 440/440 | receipts/census-base-3p05-aug.json |
| Plan (G1) | 3,016 removed / 9,080 kept, floor 184, block 8, keep 184–280, mean 216.19; normalized-proxy retention 93.06%; plan-v1 ledger regression reproduced bit-for-bit; sha256 6654d8d3b5f51eb672a92c13da54f0207ab68923b19013e7c939049d6e7c5519 | plan.json (0444) |
| Pack | 85 s, 5 workers, pure-python mmap byte copy; routers sliced to retained rows; contiguous renumber; config patched (routed_experts_per_layer, retained_expert_ids_by_layer, dynamic_container) | receipts/manifest.json, receipts/pack.log |
| G2 verify | PASS: 111,832 tensors byte-compared to source origin, 0 problems; index total == source + 2,873,724,964 (mtp+kpool) − 3,016 × (9,474,060 + 8,196) | receipts/g2-receipt.json |
| Census, pruned (stock) | exit 2 with exactly 42 problems = per-layer expert counts vs hard-coded 288; bits {3: 28104} = (9,080 + 288 MTP) × 3; dense 440/440; moe native | receipts/census-stock.json |
| Census, pruned (plan-aware) | contract_ok_plan_aware = true; residual problems = [] | receipts/census-plan-aware.json |

Notes
- Removal count 3,016 is the campaign §3 pre-registered number (B_arc = 96 GB planning prior); header-measured bytes are 9,474,060/expert + 8,196/router row (prior 9,713,306). Byte-derived alternative would be 3,088.
- Pooled-mass retention of the greedy plan is 84.5% vs 86.0% for flat keep-216 (the allocator minimizes normalized massmax proxy, not pooled mass). Recorded in plan.json summary.
- Serving blocker: overlay contract.py/moe.py/loader.py hard-code EXPERTS=288; G3 needs the per-layer expert-count loader path (DESIGN.md §5).
- `model.safetensors.index.shards-only.json` in the artifact is the un-augmented index.
