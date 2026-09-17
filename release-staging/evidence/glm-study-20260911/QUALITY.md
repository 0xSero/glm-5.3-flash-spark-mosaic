# Matched GLM quality measurements

Frozen WikiText2 panel:32×2048 tokens,65,504 next-token positions; full-vocabulary BF16 teacher comparison. This panel does not establish broad task quality or serving acceptance.

| Candidate | Experts/layer | KL (lower better) | Top-1 agreement | Perplexity | PPL change vs BF16 |
|---|---:|---:|---:|---:|---:|
| BF16 teacher |288|0|100%|3.19953|0%|
|Original Q3, unpruned|288|0.152204|87.384%|3.49691|+9.295%|
|Original K2, unpruned|288|0.438986|77.385%|4.54687|+42.110%|
|K2 REAP keep256|256|0.687907|71.350%|5.82429|+82.036%|
|Q3 REAP keep176|176|1.343988|58.807%|11.27267|+252.323%|
