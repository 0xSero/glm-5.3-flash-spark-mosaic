# SM121 router audit

The installed R3 GateLinear excludes Spark SM121 from two BF16-input, FP32-output paths even though its low-latency module imports successfully. This is a source-confirmed eligibility gap, not an attributed performance result. The initial source audit made no runtime changes. The subsequent authorized GPU microtest passed; see GPU_RESULTS.md and GPU_VALIDATION.json. The dispatcher patch remains staged, not deployed.

## Exact evidence

Audited de5c image `sha256:c0d5f76237a5fe1262291eb23b8086ef0b2637e6d8d45ccdbb9c5706df321399`; its filesystem/config equivalence to R3 is sealed in the preceding projection proof. `import-probe.log` records the actual imported GateLinear class and the installed low-latency module returning `is_available() == true`. Torch 2.13.0+cu130, CUTLASS DSL4.6.2, quack0.6.4 and cuda-bindings13.3.1 are present. The probe had no GPU exposure; CUDA remained uninitialized before and after import. The no-driver probe emitted vLLM `_C` availability warnings, so this is not a replacement for the earlier real native-extension GPU gate.

Installed source snapshots and hashes are retained in `installed-sources.json` and `AUDIT.json`. The router source SHA256 is `8ad2f44f02fbb709783c17dcceb628b16ae5adb381155f573b0cd0663243f596`.

## Dispatch consequence

| Path | Current SM121 eligibility at GLM4096×288 | Reason |
|---|---|---|
| Low-latency BF16→FP32 | Excluded | `is_blackwell_rtx` checks exact capability(12,0), not(12,1); this feeds `_can_use_ll_bf16` |
| cuBLAS BF16→FP32 | Excluded | `_can_use_cublas_bf16_fp32` derives from the same predicate |
| DSV3 specialized router | Excluded independently | Datacenter-only architecture predicate;4096×288 is not an instantiated shape |
| FP32 specialized router | Excluded independently | Datacenter-only and different fixed shapes |
| BF16x3 | Excluded | SM100-only, opt-in and FP32 weights |
| ReplicatedLinear fallback | Eligible | BF16 matmul output then cast when FP32 logits requested |

See `gate_linear.installed.py:60-68,125-151,186-252`. GLM creates this GateLinear without `force_fp32_compute` at `model.installed.py:228-234`; the source-exact native router weights remain BF16. Native FP32 correction bias is separate from the gate linear bias and is applied during routing.

The low-latency module has no4096×288 exclusion: K must be divisible by8, input/weight BF16 contiguous CUDA tensors and output FP32. Its default dispatch chooses dot-product for M<=4 and split-K for M5–16 at K4096. No SM121 or4096×288 tuned configs are present. The dot-product path converts operands and accumulates in FP32; split-K uses BF16 MMA with FP32 accumulation/output and clustered DSMEM reduction. Import availability does not establish that clustered execution compiles or runs correctly on GB10.

## Minimal isolated GPU qualification proposal

Keep native BF16 router weights and FP32 logits; do not enable DSV3 shape kernels or BF16x3. Test exact4096×288 geometry using source-backed weights and real hidden activations when available. Compare three explicit routes: current BF16-output-plus-cast, `torch.mm(..., out_dtype=torch.float32)`, and low-latency BF16/F32. Use a separate full-FP32 reference with TF32 disabled, record absolute/relative errors, normalized routing-weight differences, top8 selected expert sets/order and near-tie margins. FP32-output routes change rounding, so exact equality to the current BF16-rounded router is not an appropriate correctness requirement.

Cover M1,2,4 for dot-product, M5,8,16 for clustered split-K, and M17,32,2048 for cuBLAS/prefill. Bound allocated GPU memory below1GiB, synchronize, preserve failure receipts and compare graph/eager replay for each path. Test varied graph capture sizes sharing a pool: the existing SM120 cuBLAS graph-pool lifetime guard must be evaluated for SM121 rather than omitted accidentally when widening eligibility. Start with separate paths; enable only passing paths, retaining cuBLAS fallback if split-K fails. An eventual small patch should explicitly include tested SM121 for low-latency/cuBLAS eligibility and its graph guard, leaving datacenter-specialized predicates unchanged.

This microtest measures a router opportunity. It cannot establish the earlier trace's57.5% generic GEMM share as router time, nor predict total decode improvement. Require CPU-stack-attributed profiling and matched full-model quality/speed before promotion.

## Lane availability

After de5c's projection microproof, the quality agent reserved its GPU for the full six-down-projection quality capture. Root authorized a later <=1GiB router test on idle2822 only with a qualified image. Exact R3 was absent in2822's fresh image inventory, so no substitute-image GPU test or2384 export occurred. Eight retired compile-container cleanup reclaimed19,262,955,520B, leaving44,811,780,096B free; source/image/wheel retention and logs are sealed separately under `../retired-container-cleanup-20260912/`.
