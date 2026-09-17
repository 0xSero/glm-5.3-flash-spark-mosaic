# Existing trace: optimization leads and limits

The Local Inference Lab profiler tool produced `triage-20260912.md` from the existing single-rank R3 trace. This is the original19-prompt-token request captured for eight engine iterations, not a new profiler run and not a representative long-prefill benchmark. Capture used CUDAgraphs without Python stacks or input shapes. Kernel-duration sums can overlap and are not wall time or achievable speedup.

| Stage in this trace | Kernel family | Share of summed stage GPU durations |
|---|---|---:|
|Short prefill|EXL3 fused MoE K2|66.1%|
|Short prefill|Largest BF16 GEMM family|20.9%|
|Decode|Two largest BF16 GEMM families|57.5%|
|Decode|EXL3 fused MoE K2|25.8%|

| Overlap evidence | Conclusion |
|---|---|
|Single trace, graph replay, unresolved Python locations|No verified overlap opportunity above the tool's1% reporting threshold. A source-mapping trace would be needed for a stronger claim.|

| Fusion suggestion | Qualification |
|---|---|
|Tool labels generic GEMMs as the DSV3 router family|This is NOT proof those GEMMs are router calls. CPU scopes/shapes are missing; the raw automatic table is preserved unchanged.|
|Pinned GateLinear source|Specialized DSV3 shapes exclude this model's4096hidden/288experts. The SM120 low-latency and FP32-output eligibility uses exactcapability12.0, apparently excluding12.1. Installed-image audit is assigned before proposing any change.|

The source-backed catalog already lists specialized router GEMM and projection-overlap families; they are existing upstream leads, not newly invented optimizations. This model's eligibility and numerical behavior must be checked before enabling them. The trace does not support attributing67% of decode to routing or predicting a67% speedup.

Next: identify installed SM121 eligibility and module support without touching the running server; validate any proposed kernel against original routing arithmetic before a speed test. Prioritize a future stage-separated, shape-recorded profile for larger-prefill optimization. Keep current full-context acceptance isolated.
