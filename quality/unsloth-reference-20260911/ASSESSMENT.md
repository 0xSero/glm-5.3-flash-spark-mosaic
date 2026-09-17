# Higher-agreement GLM-5.3-Flash study

Checked 2026-09-11. Release selection is on hold following the user's request for substantially higher agreement. The existing K2 runtime remains an experimental baseline.

## External reference, not a matched ranking

Unsloth's [exact-model measurements](https://unsloth.ai/docs/models/glm-5.3-flash#quantization-analysis):

| GGUF | Weight GB | Top-1 agreement | Mean KL | KL p99.9 |
|---|---:|---:|---:|---:|
| UD-Q2_K_XL |108.72|78.34%|0.380134|6.8412|
| UD-IQ3_XXS |120.37|81.63%|0.283772|5.9611|
| UD-Q3_K_XL |147.54|86.25%|0.159697|4.0281|
| UD-Q4_K_XL |199.71|92.22%|0.049294|1.4894|

Anonymous HF metadata was captured in `gguf-model-info.json`, revision `621d456e93e926e4b52f85cff5f634358c1828f9`. IQ3_XXS files total 120367571715 bytes; the separate BF16 vision projector adds 1164010080 bytes. Weight size does not establish 262144-token capacity, activation/workspace headroom, MTP behavior, or throughput on a Spark.

Our frozen 65504-position panel measures original K2 at 77.385% / KL0.438986 and original Q3 at 87.384% / KL0.152204. Their complete artifacts occupy111.352GB and149.403GB, including native protected tensors. These are different evaluation setups and precision allocations: do not claim an EXL3/GGUF winner from juxtaposing published figures. Actual K2 serving measures77.3556% agreement; its top20 API output cannot establish full-vocabulary KL.

## Evaluation changes

[Unsloth Dynamic3](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs) combines selective tensor precision and diverse calibration with held-out KL, top1, and 32-token trajectory checks. Its trajectory evaluation uses300 held-out prompts. It does not establish one universal acceptance percentage for every model/bitrate. Perplexity alone can conceal differing token predictions.

Apply those principles here:

1. Preserve the current WikiText panel as a regression reference. Do not choose bit allocation using its held-out answers.
2. Rank precision upgrades using separate calibration/sensitivity evidence; keep protected target tensors native. Measure actual byte cost and runtime support for each allocation.
3. Add per-token KL p99/p99.9 through a separate head comparison of retained normalized activations. Existing row-summed KL cannot recover tails. Keep frozen evaluator files and completed receipts unchanged.
4. Freeze separate chat/code/tool/multilingual evaluation prompts before candidate selection. Measure full-vocabulary KL/top1 where teacher outputs are available; add greedy32-token trajectory, first divergence, tool validity, and task correctness as separate metrics. Do not label a custom suite as Unsloth's own benchmark.
5. Verify engine numerical agreement at identical weights independently of quantization agreement to BF16. Repeat final quality through the actual serving engine, then262k retrieval, images/video/MTP and sustained speed tests.

## Avoidable numerical errors

The [GLM llama.cpp PR](https://github.com/ggml-org/llama.cpp/pull/27754) reports a TF32-related fixture mismatch and warns about an MLA latent cast. That is engine parity evidence, not a quantization-quality score. Our offline workers already explicitly disable CUDA matmul TF32 in `quality/deps/evaluate_low_bpw_quality.py` and `release_route_capture_v2.py`; no corresponding fault has been established here. Audit the new B12x runtime's actual arithmetic without blindly importing llama.cpp flags.

## Next experiment

Prioritize a measured mixed K2/K3 expert allocation rather than further aggressive pruning. Keep256 and keep176 worsened agreement to71.35% and58.81%. The candidate agent is checking source compatibility, calibration evidence, and exact memory increments; the runtime agent is establishing the new B12x capacity. No higher-agreement candidate or single-Spark GGUF fit is verified yet. A90% aspiration would require a substantial improvement beyond the published120GB reference and cannot be promised within128GB.
