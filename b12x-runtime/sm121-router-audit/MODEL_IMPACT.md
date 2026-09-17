# Router impact and next native GEMM investigation

The measured router improvement is approximately0.1–0.2ms per target-plus-one-draft pass, not a57% model speedup. At M2, the native router saves4.497µs per call:42 target routers plus one M1 draft router yield about0.191ms. M1 gives about0.105ms over43 calls. Against the parent's roughly15tok/s baseline, this is on the order of0.1–0.3%, depending on accepted tokens per speculative iteration, overlap and actual memory-cache conditions. Repeated isolated weights can stay hotter than full-model execution, so these estimates are not throughput promises.

The large remaining native BF16 linear projections deserve priority. Existing trace has96,900 events and zero shape-recorded events. CUDA graph replay collapses CPU correlation to cudaGraphLaunch; its automatic DSV3-router label is unsupported. Shape/count patterns provide stronger hypotheses, but still require an independent mapping trace.

| Existing decode group | Calls / summed GPU time | Source-backed shape/count hypothesis | Confidence boundary |
|---|---:|---|---|
| WMMA128x2, grid8×194×1 |238 /215.34ms| KDA fused input weight24768×4096 BF16;34KDA layers×7 target steps=238;ceil(24768/128)=194 | Strong shape/count match, not resolved CPU callsite |
| WMMA128x1, grid8×32×1 |1022 /217.33ms|4096-output native matrices: KDA/MLA output, shared experts, dense FFN output, indexer projections | Multiple shapes share the same output dimension; cannot split by grid alone |
| WMMA128x2, grid8×192×1 |21 /18.09ms| Dense gate/up24576×4096 BF16;3dense layers×7=21 | Strong shape/count match |
| WMMA128x1, grid8×64×1 |476 /11.61ms| KDA f_b/g_b outputs8192:2×34layers×7=476 | Strong shape/count match |
| WMMA128x1, grid8×1210×1 |7 /38.41ms| Full vocabulary head154880×4096;154880/128=1210 | Strong geometry; these head launches are outside the graph |
| WMMA128x2, grid8×128×1 |84 /18.85ms| MLA q_b output16384;12target+draft MLA calls×7=84 | Plausible; distinguish target/draft in mapping |
| WMMA128x2, grid8×3×8 |294 /3.98ms|288-output target router;42×7=294 | Plausible router family, about0.4% of summed decode GPU time, not67% |

Do not add these percentages as wall-clock speedup: kernels may overlap and this tiny trace covers only seven decode target steps. GEMV launches with grid19360 correspond geometrically to154880/8 output rows and may also be vocabulary-head work; exact target/draft attribution is unresolved.

Installed KDA source confirms q/k/v plus beta/f_a merge: three8192 outputs plus64 and128 give24768. The weight is202,899,456B (193.5MiB) per layer;34layers read about6.425GiB per target pass if each weight is read once. The matched215.34ms cluster over seven steps implies roughly224GB/s effective weight traffic under this hypothesis. That is a bandwidth-scale problem in large protected BF16 matrices. A288×4096 router is only2.25MiB. Native output-head weight is1.1816GiB and remains protected BF16; this investigation does not propose quantizing it.

## Bounded next profile, after the active baseline sweep

Keep graphs enabled and all weights/native arithmetic unchanged. Capture a short mapping trace with `torch_profiler_record_shapes=true` and `torch_profiler_with_stack=true`, keeping context, speculative depth and prompt state matched. Record target-versus-draft module prefixes, input/output shapes, dtype, stride and weight shape during graph preparation through read-only metadata hooks; do not collect tensor values or call GPU `.item()`. Verify mappings rather than assuming shape recording alone reconstructs operators inside an already-captured CUDA graph.

If replay still exposes only graph-launch CPU sites, collect a separate isolated, source-exact native-linear mapping microtrace for one matrix at a time, with named module scopes and its CUDA graph. Start with KDA24768×4096, KDA-output4096×8192, then shared4096×4096/4096×2048 and MLA4096×16384. Leave the full1.18GiB vocabulary head for a separately admitted memory window. Match kernel names, grid, dtype, sizes and graph node evidence against the formal server trace; state any unresolved ambiguity.

Measure M1/2/4/8/16 plus the actual configured prefill chunk size, retaining FP32 references and native BF16 output dtype. Candidate kernel/tiling tests must compare output arithmetic and graph behavior before a full quality rerun. The existing KDA adapter already overlaps its small g_a/g_b branch with the large fused input projection, so proposing that overlap again would duplicate existing work. No new GPU requests, running-server edits, or protected-weight conversions were performed for this analysis.
