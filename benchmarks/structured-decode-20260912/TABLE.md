# Verified structured generation

C1; temperature0; low reasoning; cold prefix cache; native MTP enabled. Correct normal-stop JSON answers required. Every timing value was recomputed from exact emitted token IDs. Warmups and sub30-second screens remain labeled. This counting task does not establish broad model quality or MTP speedup.

| Input tokens | Output tokens | Repeat | Class | Window s | TOTAL decode tok/s | Per-request decode tok/s | Native prefill tok/s | TTFT s |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 1023 | 426 | measured1 | MATCHED_SCREEN | 28.21 | 15.07 | 15.07 | 426.26 | 2.44 |
| 1023 | 418 | measured2 | MATCHED_SCREEN | 27.96 | 14.91 | 14.91 | 425.41 | 2.46 |
| 1023 | 417 | warmup | MATCHED_SCREEN | 27.65 | 15.05 | 15.05 | 417.05 | 2.48 |
| 4095 | 420 | measured1 | MATCHED_SCREEN | 28.05 | 14.94 | 14.94 | 441.82 | 9.33 |
| 4095 | 419 | measured2 | MATCHED_SCREEN | 27.88 | 14.99 | 14.99 | 443.22 | 9.29 |
| 4095 | 420 | warmup | MATCHED_SCREEN | 28.20 | 14.86 | 14.86 | 434.64 | 9.48 |
