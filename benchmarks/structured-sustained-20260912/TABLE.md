# Verified structured generation

C1; temperature0; low reasoning; cold prefix cache; native MTP enabled. Correct normal-stop JSON answers required. Every timing value was recomputed from exact emitted token IDs. Warmups and sub30-second screens remain labeled. This counting task does not establish broad model quality or MTP speedup.

| Input tokens | Output tokens | Repeat | Class | Window s | TOTAL decode tok/s | Per-request decode tok/s | Native prefill tok/s | TTFT s |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 1023 | 683 | measured1 | MATCHED_SUSTAINED | 45.53 | 14.98 | 14.98 | 422.32 | 2.48 |
| 1023 | 683 | measured2 | MATCHED_SUSTAINED | 45.54 | 14.98 | 14.98 | 426.32 | 2.45 |
| 1023 | 682 | warmup | MATCHED_SUSTAINED | 45.48 | 14.97 | 14.97 | 427.09 | 2.42 |
| 4095 | 683 | measured1 | MATCHED_SUSTAINED | 45.57 | 14.97 | 14.97 | 443.14 | 9.30 |
| 4095 | 683 | measured2 | MATCHED_SUSTAINED | 45.83 | 14.88 | 14.88 | 445.47 | 9.25 |
| 4095 | 683 | warmup | MATCHED_SUSTAINED | 45.92 | 14.85 | 14.85 | 442.09 | 9.32 |
| 16383 | 684 | measured1 | MATCHED_SUSTAINED | 45.87 | 14.89 | 14.89 | 452.06 | 36.32 |
| 16383 | 691 | measured2 | MATCHED_SUSTAINED | 46.53 | 14.83 | 14.83 | 452.31 | 36.30 |
| 16383 | 684 | warmup | MATCHED_SUSTAINED | 45.84 | 14.90 | 14.90 | 451.73 | 36.35 |
| 65535 | 687 | measured1 | MATCHED_SUSTAINED | 46.38 | 14.79 | 14.79 | 456.95 | 143.59 |
| 65535 | 691 | measured2 | MATCHED_SUSTAINED | 46.57 | 14.82 | 14.82 | 455.28 | 144.12 |
| 65535 | 683 | warmup | MATCHED_SUSTAINED | 45.90 | 14.86 | 14.86 | 455.58 | 144.03 |
| 131071 | 682 | measured1 | MATCHED_SUSTAINED | 45.97 | 14.81 | 14.81 | 455.38 | 288.12 |
| 131071 | 682 | measured2 | MATCHED_SUSTAINED | 46.01 | 14.80 | 14.80 | 454.99 | 288.40 |
| 131071 | 682 | warmup | MATCHED_SUSTAINED | 45.84 | 14.85 | 14.85 | 455.17 | 288.28 |
| 199999 | 682 | measured1 | MATCHED_SUSTAINED | 45.89 | 14.84 | 14.84 | 453.89 | 441.04 |
| 199999 | 683 | measured2 | MATCHED_SUSTAINED | 46.03 | 14.82 | 14.82 | 453.92 | 441.03 |
| 199999 | 682 | warmup | MATCHED_SUSTAINED | 45.91 | 14.83 | 14.83 | 453.48 | 441.46 |
| 260095 | 682 | measured1 | MATCHED_SUSTAINED | 46.01 | 14.80 | 14.80 | 453.16 | 574.59 |
| 260095 | 682 | measured2 | MATCHED_SUSTAINED | 45.91 | 14.83 | 14.83 | 452.94 | 574.84 |
| 260095 | 682 | warmup | MATCHED_SUSTAINED | 45.90 | 14.84 | 14.84 | 452.92 | 574.89 |
