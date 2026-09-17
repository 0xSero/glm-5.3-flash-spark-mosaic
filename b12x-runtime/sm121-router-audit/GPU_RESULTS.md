# Isolated SM121 native-router microtest

Actual native layer5 BF164096×288 router and FP32 correction bias; deterministic synthetic BF16 inputs. This is not model-quality or end-to-end throughput acceptance. Five timing samples, each a CUDA graph containing20 identical calls; median per-call GPU time, excluding Python launch gaps. TF32 off; BF16 reduced-precision reduction retains the installed default true.

| Tokens M | Baseline µs | cuBLAS FP32 µs | Low-latency FP32 µs |
|---:|---:|---:|---:|
|1|4.934|3.646|2.496|
|2|7.006|5.546|2.509|
|4|7.118|5.677|3.318|
|5|7.419|5.982|14.981|
|8|7.682|6.259|15.098|
|16|8.037|6.678|15.274|
|17|10.456|8.032|—|
|32|10.200|8.032|—|
|2048|69.406|60.118|—|

All24 cells passed finite outputs and exact graph/eager comparison; final ordered shared-pool replay passed. New FP32-output paths passed comparison to full-FP32 reference at atol/rtol0.0005. Maximum absolute error: cuBLAS7.39098e-6, low-latency2.98023e-7; baseline max0.00416136. At M2048, cuBLAS top8 expert sets matched the FP32 reference on all2048 rows; order matched2047/2048. Baseline sets matched97.119%, order84.766%. Near-tie behavior and synthetic inputs prevent treating these as language-model quality scores.

Peak allocated141,452,800B (134.9MiB);512MiB Torch allocator cap. Terminal exit0/noOOM. Image4ec526… is a source/package-qualified router-only base; it does not claim complete R3 image equivalence. Native model weights were never modified.

Candidate dispatcher is staged only: exactSM1214096×288, ll dot-product at testedM1/2/4; cuBLAS otherwise, retaining SM120 lifetime guard and existing behavior elsewhere. M3 intentionally falls through to cuBLAS pending qualification. Four CPU source/dispatch regression tests pass. The full installed candidate-class GPU/serving integration has not run.

Failed attempts retained:1 missing TP group;2 missing current VllmConfig;3 harness512MiB allocator cap reached by per-route warmup-stream cuBLAS workspaces. Attempt4 reuses one stream, passing without raising the cap.
