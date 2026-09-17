# Why the Unsloth unpruned IQ3 artifact fits a different memory budget

Inspected published GGUF metadata at revision `621d456e93e926e4b52f85cff5f634358c1828f9` of https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF. Four file sizes total120,367,571,715 bytes. This is a header audit, not inference, tensor-value parity, a full download, or full-weight hash verification. The9,429,859-byte metadata-only first shard was downloaded completely and its published SHA verified. For other shards only leading10MiB HTTP ranges were read; header SHAs and published expected full-shard hashes are recorded separately.

| Component | Payload bytes | Observed types |
|---|---:|---|
|Target routed experts,42layers|109,867,696,128|41layers:gate/upIQ2_S,downIQ3_S; layer44:allthreeIQ4_XS|
|Target nonexpert tensors|7,690,382,584|Q6_K,Q8_0,F32|
|MTP layer45|2,799,972,480|Q2_K,Q3_K,Q6_K,Q8_0,F32|

Target routed tensors contain304,405,807,104 parameters:2.8874008 effective bits per weight including format overhead. The file label is not a claim that every tensor uses3bits. Output head, token embeddings and many attention weights are Q6_K. Thus the size does not demonstrate our native-protected EXL3 configuration can fit at the same expert precision. Vision projector is a separate file and is excluded from the120.37GB weight total.

All126 target expert tensors retain288 experts. This supports the user's unpruned lead, but does not prove262144context+vision+MTP capacity or quality in our engine. The published81.63% agreement uses a different evaluation setup.

Action: examine per-projection EXL3 precision allocation, particularly down projections, under the existing native-protected memory budget. Do not quantize protected tensors implicitly. The current EXL3 native ABI has per-projection bit-width parameters; Python-loader support and GPU behavior still require validation. This header allocation is a research lead, not evidence of a causal quality gain.

The initial parser attempt rejected the metadata-only shard because its aligned data offset exceeds the unpadded EOF by29bytes. There are zero tensors in that shard, so the revised check enforces data bounds only when tensor payloads exist; all actual tensor ranges remain checked against published shard sizes. No full weight payload was synthesized.
