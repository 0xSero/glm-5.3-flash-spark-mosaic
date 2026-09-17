# Measured inference matrix

No inferred or estimated value is placed in the matched decode columns.
Request-native prefill uses summed overlapping request durations; it is not aggregate GPU throughput.

| Prompt target | Actual prompt/request | C | Repeat | Status | TOTAL matched decode tok/s | Mean/request matched decode tok/s | Window s | Server request-prefill tok/s | Prompt/TTFT effective tok/s | TTFT p50/p90 s | Client decode estimate tok/s | End-to-end total tok/s |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1024 | 1024 | 1 | 0 | MATCHED_SUSTAINED | 13.75 | 13.75 | 148.84 | 109.18 | 108.48 | 9.44/9.44 | 13.75 | 12.94 |
| 1024 | 1024 | 1 | 1 | MATCHED_SUSTAINED | 13.47 | 13.47 | 151.92 | 429.68 | 422.02 | 2.43/2.43 | 13.47 | 13.27 |
