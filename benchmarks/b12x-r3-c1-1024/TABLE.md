# Measured inference matrix

No inferred or estimated value is placed in the matched decode columns.
Request-native prefill uses summed overlapping request durations; it is not aggregate GPU throughput.

| Prompt target | Actual prompt/request | C | Repeat | Status | TOTAL matched decode tok/s | Mean/request matched decode tok/s | Window s | Server request-prefill tok/s | Prompt/TTFT effective tok/s | TTFT p50/p90 s | Client decode estimate tok/s | End-to-end total tok/s |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1024 | 1024 | 1 | 0 | MATCHED_SUSTAINED | 13.66 | 13.66 | 149.84 | 236.38 | 235.87 | 4.34/4.34 | 13.66 | 13.28 |
| 1024 | 1023 | 1 | 1 | MATCHED_SUSTAINED | 13.55 | 13.55 | 151.04 | 410.27 | 409.04 | 2.50/2.50 | 13.55 | 13.34 |
