# Predeclared matched-memory nonuniform pruning experiment

**Status: CPU plans validated; no checkpoint built and no new GPU quality job launched.** The already-running unpruned mixed K2/K3 quality capture remains the priority. Four focused tests pass. Plans must remain frozen while evaluating them; do not tune allocation on held-out WikiText results.

| Arm | Retained experts across42 layers | Removed experts | Target tensor bytes, including original native MTP |
|---|---:|---:|---:|
| Uniform272 |11424|672|106,967,076,600|
| Nonuniform272budget |11424|672|106,967,076,600|

Both are 99.620853GiB of tensor payload, with exactly equal router bytes as well. These are derived payload budgets, not measured runtime memory or KV capacity. Serialized safetensors/JSON headers are excluded and must be measured after a physical build. The pinned source manifest's `physical_tensor_bytes` field actually equals weight **file** bytes; this study derives payload from audited per-expert tensor bytes plus native retained tensor bytes instead.

## Fixed selection method

Both arms keep the highest-ranked experts according to the exact sealed `massmax_domain` criterion, with lower original expert ID breaking equal-score ties. For each expert this score is the maximum over available domains of its routed weighted activation mass divided by the layer's total mass for that domain.

Those raw maximum-over-domain scores do not sum to one: layer sums range 1.996975–3.616899. Direct cross-layer comparison would partly reflect differing domain specialization. The nonuniform arm therefore divides each layer's score vector by that layer's sum, preserving its expert ranking while measuring relative saliency share. It greedily removes the lowest-cost next block of8 bottom-ranked experts until exactly672 have been removed. Ties use lower layer ID. Counts stay between240 and288, multiples of8. The240 floor caps removal at three times the mean16-expert removal and was fixed before quality evaluation.

This normalized score is a **heuristic proxy for concentration, not measured counterfactual layer sensitivity**. It removes absolute activation-scale differences, but it cannot measure residual-path importance, expert interactions, rerouting effects or final-logit error. Summed removed proxy is 0.442996464 for uniform and 0.406254211 for nonuniform, only the objective the allocator minimizes. Do not call that difference a measured quality gain.

## Observation and source provenance

The sealed partial observation set contains21,248 records and32,602,850 tokens. It covers nine domains: agentic, coding, CUDA, cybersecurity, deep reasoning, function calling, long context, math and terminal. Science is missing. Only homogeneous shards provide per-domain statistics; mixed shards contribute only pooled mass. Every expert was naturally routed at least49,419 times. No forced routing, synthetic unused-expert coverage or additional observations are introduced.

Observation model: `0xSero/GLM-5.3-Flash-EXL3-Q4` revision `d0b9301a10da765df1d76571107577041009f28d`. This is a Q4 observation proxy transferred to K2, not an observation directly measured on the K2 runtime. K2 source: `0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw`, revision `35b4b580debe2a1510a2eb042d9d6d4cd7fb3a6b`. Both plans embed source pins and input SHA256 bindings. `allocation-ledger.json` records every removal block. The allocator never loads the held-out quality fixture or measured quality reports; this does not assert corpus-wide semantic deduplication.

## Artifact/runtime contract

For **both** arms, retain global `text_config.n_routed_experts=288` and add complete42-layer maps `routed_experts_per_layer` and `retained_expert_ids_by_layer`. Exact maps are under `text_config_patch` in the two plan JSON files. Lists are sorted original IDs; route weights and correction biases must be sliced in that same order. Physically rename each retained expert's complete48 packed fields to contiguous IDs0..N-1 using the included original-to-contiguous mapping. All four rank fragments remain. Do not merely mask router logits while leaving all weights resident.

Each removed K2 logical expert removes6,402,096 packed tensor bytes plus8,196 router-row bytes (BF16 weight4096×2 plus FP32 correction bias). Every other native tensor remains byte-identical, including all889 MTP tensors and all288 native MTP experts. This contract is separate from the mixed-precision `layer_bits` contract. Neither arm modifies vision, target attention, shared experts, embedding/head or the separate FP8-MTP serving policy.

Runtime support is independently owned under `b12x-runtime/layerwise-expert-counts`. Physical artifact validation, actual loading and graph tests are still required. Same tensor bytes do not guarantee identical workspace, activation peak, KV admission or speed.

## Fixed comparison before any selection

After explicit scheduling: build each immutable artifact; verify complete expert/router remap and native closure; run both through the same45-layer capture and frozen32×2048 quality panel, using the unchanged native BF16 head. Report65,504-position KL, top1 agreement, perplexity and retained per-token tails against original BF16. Also compare to the retained unpruned K2 normalized baseline to separate pruning loss from quantization loss. Preserve failures and negative results. A quality advantage requires measured outcomes; proxy mass alone cannot select the release. No held-out result feeds back into these allocations.

## Exact layer counts

| Layer | Uniform | Nonuniform |
|---:|---:|---:|
|3|272|280|
|4|272|288|
|5|272|288|
|6|272|288|
|7|272|272|
|8|272|280|
|9|272|280|
|10|272|280|
|11|272|280|
|12|272|280|
|13|272|280|
|14|272|272|
|15|272|280|
|16|272|280|
|17|272|280|
|18|272|272|
|19|272|272|
|20|272|272|
|21|272|280|
|22|272|280|
|23|272|272|
|24|272|272|
|25|272|272|
|26|272|272|
|27|272|264|
|28|272|264|
|29|272|264|
|30|272|272|
|31|272|256|
|32|272|264|
|33|272|264|
|34|272|264|
|35|272|272|
|36|272|264|
|37|272|272|
|38|272|272|
|39|272|264|
|40|272|264|
|41|272|272|
|42|272|264|
|43|272|256|
|44|272|240|
