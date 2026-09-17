# Unpruned next-candidate decision

**Do not launch another full-model run yet.** The completed mixed experiment improved agreement by only0.714460 percentage points. Expert-level allocation is technically plausible but its calibration objective improves only10.97% over the current two-layer allocation at equal bytes, while requiring24 layers to dispatch two precision banks. That is insufficient evidence to justify a new serving implementation and another full-model quality capture immediately.

## Independently verified result

| Artifact | Matched BF16 top1 agreement | KL(BF16→candidate) | Perplexity |
|---|---:|---:|---:|
| Unpruned K2 |77.384587%|0.438985908|4.546866|
| K2 with layers5/32 K3 |78.099047%|0.417792191|4.455079|

The terminal container exited0 without OOM; all45 raw layer receipts and all32 normalized output rows are sealed. Baseline/candidate fixture, teacher, native imports, dependency hashes, head arithmetic and65,504 positions match. `mixed-result-validation.json` records independent verification of report/seal hashes and exact deltas. This is offline quantization fidelity, not serving acceptance.

No honest numerical prediction of the next top1 agreement follows from one experiment. Do not linearly extrapolate the0.71446-point gain to3/4layers or translate proxy reduction into agreement. The current working evidence is77–78% agreement; a large jump is unproven.

## Frozen calibration-only designs

All12,096 experts and all native protected tensors remain. These designs never read held-out labels, quality reports or normalized output rows. Selection uses the84 hash-verified paired MIT calibration sidecars, all42layers×288experts, same pinned K2 source and paired K3 source as the completed experiment.

| Design | K3 experts | Additional tensor GiB over K2 | Mixed-bank layers | Summed calibration proxy reduction |
|---|---:|---:|---:|---:|
| Current whole5+32 |576|1.68750|0|198.537049|
| Expert576 |576|1.68750|24|220.323156|
| Whole5+32+6 |864|2.53125|0|296.490351|

`expert576-design.json` contains complete original-ID selections and exact K2/K3 source-part hashes. It upgrades8-expert blocks by largest remaining K2-minus-K3 proxy benefit, ties lower layer/originalID. Block8 is a conservative design convenience, not a fused-kernel requirement. `whole3-design.json` follows the already-fixed independent layer ranking and uses the existing `layer_bits` mechanism. If a further full-model candidate is explicitly requested, whole5+32+6 has the smallest implementation risk; it is not forecast to deliver a large quality gain.

The quantizer proxy is the sum over12 projection/rank slices of `trace(E H E^T)/trace(W H W^T)`. Hessians use conditional natural-route samples, capped at8192; the observed minimum is1680. This dimensionless reconstruction objective is useful for paired comparisons but is not route-frequency-weighted end-to-end sensitivity. Its use across experts/layers remains heuristic.

## Memory frontier

The sealed B12x baseline reports751,007 aggregate KV tokens in5.84GiB,97.89GiB model loading,4.78GiB profiled activations and0.13GiB graphs. Proportional KV arithmetic reserves2.03849GiB for262,144tokens, leaving3.80151GiB for extra weights. Hybrid cache allocation and rounded logs prevent treating this as an exact admission test.

Expert576 leaves an estimated2.11401GiB beyond the context floor before new workspaces. Whole3 leaves1.27026GiB. Four whole layers would leave only0.42651GiB, which is too little unverified margin to choose before actual vision/full-context/MTP admission. These estimates require the existing runtime's expert-only FP8 MTP policy; the artifact itself retains the original native MTP payload byte-identically. They do not apply to a requirement for native BF16 MTP execution.

Both designs preserve33,835,039,608 native protected tensor bytes and889 native MTP tensors/all288 MTP experts. Expert576 payload equals the existing two-layer candidate exactly:113,086,732,152bytes. Whole3 is113,992,701,816bytes. Serialized headers and runtime workspaces are separate.

## Expert-level runtime feasibility

Existing EXL3 fused calls accept one scalar K per projection for the entire launch, and current packed banks allocate one `bits*16` trellis width. Mixed expert pointers cannot share that launch safely. K=0 does not provide per-expert bit selection.

A valid implementation needs two compact homogeneous K2/K3 banks per mixed layer, with unchanged native288-expert router/top8/weights. Map each original expert ID to its bucket/local ID, expand all four archive ranks inside that bucket, and discard routes belonging to the other bucket through the existing sentinel mapping. Sum both FP32 routed outputs **before** the single final dtype cast and shared-expert addition. Calling the existing cast-at-return helper twice and then adding BF16 outputs is a different arithmetic path.

Loader integration is real work: dispatch each mapped checkpoint expert into the proper compact parameter bank, enforce full loaded-key/shape coverage and preserve rank mapping. Padding all288 expert slots toK3 defeats the expected memory saving. Empty buckets should skip, and homogeneous layers retain the current single-launch path. Current expert-count GPU microtests prove mapping/fused execution, not mixed K2/K3 bank correctness.

Two banks add sort/count work and one additional fused call per mixed layer; rank4 expansion is virtual expertise within each call, not four separate fused calls. Shared temporary buffers may be reused sequentially on one stream. One extra FP32 output costs16KiB per scheduled token (128MiB at8192tokens), plus routing scratch. Exact graph/prefill/decode overhead requires a bounded microtest before any full implementation or capture. No speed claim is made.

## Fixed-K2 calibration/codebook route

A fixed-byte improvement would be more valuable than spending the remaining KV margin, but none is established. The pinned MITv3.1 path already uses calibrated Hessians, output scaling, golden-section global-scale search, LDLQ error feedback and exact encode/decode checks. There is no evidence of a simple missing-quality flag. A changed codebook is not a metadata tweak: it must remain compatible with the decoder's MCG/trellis ABI or requires new kernels and parity work.

The decisive bounded gate would be a few representative experts re-encoded at unchanged K2/MCG format with predeclared damping/seed variants, scored on separate fit/check activation partitions from the independent calibration corpus. Preserve a reproduction control, exact bytes and raw losses; require a consistent check-set improvement beyond repetition noise before considering a model-wide re-encode. Do not select using the WikiText regression panel. Stop if only training proxy improves, coverage/rank is inadequate, or calibration states cannot be obtained cheaply.

This pilot is **not ready to launch cheaply**: the current Omarchy inventory found only an old8-row smoke transient and no retained full-layer activation/Hessian cache. Full calibration capture must not be assumed available. First resolve source/check-activation availability without GPUs; if that requires another expensive45-layer pass, stop and reassess. No full re-encode, new quality capture or two-bank GPU job was started for this decision.
