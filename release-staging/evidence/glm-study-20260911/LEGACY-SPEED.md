# Measured inference matrix

No inferred or estimated value is placed in the matched decode columns.
Request-native prefill uses summed overlapping request durations; it is not aggregate GPU throughput.

| Prompt target | Actual prompt/request | C | Repeat | Status | TOTAL matched decode tok/s | Mean/request matched decode tok/s | Window s | Server request-prefill tok/s | Prompt/TTFT effective tok/s | TTFT p50/p90 s | Client decode estimate tok/s | End-to-end total tok/s |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1024 | 1023 | 1 | 0 | MATCHED_SUSTAINED | 12.58 | 12.58 | 162.77 | 117.68 | 117.23 | 8.73/8.73 | 12.58 | 11.94 |
| 1024 | 1024 | 1 | 1 | MATCHED_SUSTAINED | 13.30 | 13.30 | 153.86 | 437.36 | 426.87 | 2.40/2.40 | 13.30 | 13.11 |
