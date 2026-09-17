# Whole-layer K2/K3 candidate

This private experimental artifact replaces routed layers 5 and 32 of the pinned K2 model with their paired MIT-lineage K3 encodings. All 288 logical experts remain in all 42 routed layers. Attention, routers, shared experts, embeddings/head, vision and the original native MTP payload remain byte-identical. This is neither REAP nor a new quantization.

| Property | Verified assembled value |
|---|---:|
| Default routed precision | K2 |
| K3 routed layers | 5, 32 |
| Average routed tier | 2.047619 bpw |
| Weight files | 133 |
| Tensor bytes | 113,086,732,152 |
| Weight file bytes | 113,163,970,040 |
| Protected native tensor bytes | 33,835,039,608 |
| Increment over complete K2 | 1,811,939,328 tensor bytes |

Layer selection uses the separate 600-row calibration proxy ranking recorded in `../mixed-precision-feasibility/`. It does not use the held-out WikiText quality panel. The proxy is not measured end-to-end sensitivity, and no quality improvement is claimed before evaluation.

## Assembly

Completed on de5c in a CPU-only container: exit 0, no OOM, no GPU device requests. All 133 weight files passed fresh source and destination SHA256 checks and were hardlinked. All 583,090 tensor names and the 2,482-tensor native protected closure passed structural verification. Candidate manifest SHA256: `73291ede5c0f2ad13dcd9fbdfe3887f133ef713d56c589c59c6392bdb887b45c`. Exact terminal evidence is under `assembly-receipt/`; `verification.json` binds the downloaded receipts.

Private candidate path: `/home/valentine/glm53-single-spark-release-20260911/quality/mixed-layer-candidate/model`.

`assemble.py` requires a private SHA256-to-path source map and a nonexistent output directory. It verifies every selected source hash, hardlinks immutable files where possible, checks destination hashes and packed tensor coverage, rebuilds the index, and verifies the complete native protected closure. Cross-filesystem copies are bounded and require the full copy budget plus 20 GiB free. No GPU or PyTorch is required.

`stage_and_build.py` launches only the CPU builder, capped at two CPUs and 8 GiB. Its terminal container inspect, full logs, final manifest and build status are retained. `transfer_fast.py` copies only the four selected K3 files through the wired 2822 fabric bridge; transfer completion alone is not a hash-validation result. The previous slow-route partial files and logs are preserved separately.

## Runtime and quality gates

Both `config.json.quantization_config` and `quantization_config.json` contain `bits: 2`, `layer_bits: {"5": 3, "32": 3}` and `rank_stacked_tp: 4`. The staged `../../b12x-runtime/mixed-layer-precision/` overlay resolves method-local precision and has passed CPU/config selection tests. Current serving remains unchanged. This candidate has not passed GPU loading, graph replay, context/vision/MTP admission, throughput or quality acceptance.

The manifest schema is `glm53-mixed-whole-layer-exl3-v1`. A separate mixed-source quality validation profile is required; the existing physical-REAP validator must not be bypassed or fed fabricated REAP metadata. The manifest and final status explicitly leave quality/runtime acceptance false.

`source-config.json` preserves the exact pinned original config for the separate native-MTP view. The source map and calibration fixtures are private and stay outside the candidate model. Publication requires a separate review of copied provenance and permissions; do not publish recoverable corpus data.
