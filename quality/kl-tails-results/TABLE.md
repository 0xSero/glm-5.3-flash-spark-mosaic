# Full-vocabulary KL tail supplement

65,504 positions; lower KL is better. Two-pass head-only calculation, FP64 accumulation, original report crosschecks passed. No model layers re-run.

| Candidate | Mean | p99 | p99.9 | Max |
|---|---:|---:|---:|---:|
|original-q3-v1|0.152204|1.843328|4.039562|8.510485|
|k256-v1|0.687907|5.950869|9.335599|15.121773|

Tiny negative values from finite-precision rounding are retained and counted; no clamping. Quantiles use linear interpolation. Cross-method external GGUF comparisons remain unmatched.
