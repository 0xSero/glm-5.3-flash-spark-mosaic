# Prospective projection6 experiment

Use unpruned K2 experts, upgrading only down projections in layers 5, 26, 31, 32, 35, 36 to paired MIT K3. All 288 experts per target layer remain; protected target components and the original 288-expert MTP source stay native. The serving MTP policy is a separate runtime decision.

Selection was frozen from independent paired projection calibration errors before this candidate was measured. Its proxy improvement is 292.0973099162471 versus 198.53704880471759 for the earlier whole-layer 5/32 candidate, a 47.12483724060797% larger proxy reduction. This does not predict the agreement improvement. Held-out WikiText labels were not used for allocation.

| Property | Value |
|---|---:|
| Added tensor bytes versus K2 | 1,811,939,328 (1.6875 GiB) |
| Total tensor bytes | 113,086,732,152 |
| Weight file bytes | 113,163,969,240 |
| Native protected tensor bytes | 33,835,039,608 |
| Packed target experts | 42 × 288 |
| Selected down projection fields | 27,648 |
| Weight files | 133 |

CPU build exited zero with no OOM and hashes every source contributor and output shard. Manifest: `590ec08e84a66767e03964f7482e84c952e6b98c26cddc7d45983592c4e491f5`. Selection: `38efe0f4da5d382499d4a3fb220ac66c8597b59190e11cd7679038f21afc6567`.

The separate real 8-expert GPU proof exercised one dynamic fused EXL3 launch at K=(2,2,3), matching the FP32 reconstructed reference within 7.15e-5 maximum absolute error and replaying graphs exactly. This proves the mechanism, not full-model quality or capacity. Full quality uses the same frozen 32×2048 tokens, BF16 teacher/head arithmetic, and 65,504 scored positions as K2 and whole-layer controls; each reconstructed projection is checked against the prospective precision map. No serving speed or 262k admission claim is made for this artifact yet.
