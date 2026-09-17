# Unpruned expertwise K2/K3 feasibility

**Feasible with separate homogeneous packed banks; not supported by changing pointers in the existing single-bank wrapper.** No serving code/image/server changed. Five CPU oracle tests pass; no mixed-bit GPU or quality result is claimed.

## Source evidence

The serving image retains verified ExLlama0.0.43 GPU implementation matching the compared c5d9c657 reference sources. The Local Inference Lab Jovian vLLM pin is3aada677; EXL3 port details remain in the parent directory.

| Layer | Current contract | Consequence |
|---|---|---|
| `exl3.py:create_weights` | One packed bank with shared expert dimension and a single `bits`-derived trellis shape | K2/K3 payloads cannot occupy the same parameter tensor without a new representation |
| `build_exl3_fused_state`, line389 | `_exl3_k` equals the single layer bit width | Pointer arrays carry no per-expert precision |
| `apply_exl3_fused_moe`, lines433–449 | Passes the same scalarK for gate/up/down | Every expert in that launch is decoded using that precision |
| Native `exl3_moe.cu`, lines108–110,178–179,206 | ScalarK_gate/K_up/K_down select a compiled kernel | K=0 handles different **projection** precisions; it does not select precision by expert |
| `LinearEXL3` and Python expert loop | Each individual Linear stores its ownK | A mixed individual-expert loop is possible after loader changes, but is not the fused fast path |

Passing a K3 pointer to a K2-specialized launch risks incorrect decoding/address calculations. No such experiment was attempted.

## Minimal safe route without rebuilding C++

Retain a single outer288-expert FusedMoE and the exact native router, correction bias, top-k, routed scaling and shared-expert path. Its quantization method owns two packed banks, oneK2 and oneK3, rather than creating two independent routers.

For each layer:

1. At load time, form a complete partition of original expert IDs0..287. Keep sorted original IDs and persistent original→bank/local maps. Each expert's gate/up/down and four archive ranks must use the same selected precision in this initial implementation.
2. Allocate compact per-bank packed tensors with that bank'sK and expert count. Do not allocate288K3 slots, padK2 payloads toK3, or duplicate the native dense/router/shared/vision/MTP tensors.
3. The checked loader dispatches every original-ID checkpoint key to the correct bank and local row, preserving rank/projection/field mapping. Assert complete coverage of every expected native expert×rank×projection×field, exact source hashes and selected precision. Existing mapped names cannot simply be routed into the old uniform destination parameters.
4. Run the unchanged global router once. Map its IDs into each bank with an out-of-bank sentinel and zero contribution. Expand retained routes across the four archive ranks; repeat their original routing weights **without dividing by four**.
5. Invoke the existing fused kernel once per nonempty precision bank. Sum the two **FP32** outputs, then convert once to the intended dtype and add the shared path through the existing outer runner. Calling the current public `apply_exl3_experts` twice and adding its already-BF16 outputs would introduce an avoidable extra rounding step.
6. Empty banks are omitted using static load-time decisions. Uniform layers retain the original single-launch path. Runtime routing uses fixed-shape GPU lookups/masks, no host `.item()` decision or variable-size compaction, to permit CUDA graphs.

No expert-count multiple-of8 requirement was found in the fused ABI. The native kernel checks hidden/intermediate alignment and accepts pointer/count arrays with arbitrary expert count. With rankstack4, the virtual expert count is four times the physical bank count. A1/287 split is a valid CPU mapping contract; its actual GPU execution remains untested. Eight-expert allocation blocks may still be a study policy.

## Memory and overhead

One full4096×2048×three-projection expert costs an additional3,145,728 bytes (3 MiB) when upgradingK2→K3. Scales and geometry remain unchanged. Therefore576 upgrades equal1.6875 GiB, the same payload increment as two complete288-expert layers. Compact banks should preserve this byte advantage, apart from small maps/pointer tables; it must be measured after loading.

A mixed layer adds a second sort/count path, fused launch and FP32 output reduction. Routing itself and shared experts should execute once. The current eight-iteration profile attributes34.38% of summed kernel duration to EXL3 MoE, but does not predict the two-bank slowdown. Concentrating upgrades into fewer layers reduces additional dispatches; it does not establish which allocation best preserves quality.

The calibration-only576-upgrade allocation currently spans24mixed layers. Its saliency advantage versus whole layers5+32 is a proxy, not an observed quality gain. Twenty-four extra bank dispatches may be a material latency cost and must be measured at C1 and concurrent/prefill workloads.

## Required gates

- CPU: full288-ID partition, original→local→original bijection, router-weight conservation across four ranks, empty/allK3 cases, arbitrary bank counts, immutable protected metadata, unknown/missing/duplicate packed key rejection.
- GPU microtest: same layer with real K2/K3 experts, source-exact router/bias; compare fused buckets against per-expert LinearEXL3 and reconstructed-weight FP32; include routes using only one bank and both banks, count boundaries and representative prefill sizes.
- CUDA graphs: alternating bucket/layer execution, changing routing inputs, repeated replay, no scratch alias corruption. Shared expert output must be included in the complete runner comparison.
- Full checkpoint: exact expert-wise manifest and loaded-key closure, measured model/KV/activation memory, unchanged288router/native protected tensors, no hidden dense reconstruction.
- Quality: the same frozen held-out corpus/BF16 head arithmetic and full-model capture as the matched baseline. No selection based on held-out errors.
- Performance: matched prefill/decode/concurrency sweeps with native counters and profiler attribution. No speed claims based on CPU dispatch tests.

`bucket_contract.py` is deliberately a CPU oracle, not a deployment patch. `test_bucket_contract.py` verifies mapping/weight conservation and edge cases. `SOURCE_EVIDENCE.json` binds this report, exact inspected wrapper/native sources and test result.

## Lower-complexity projection alternative: K2 gate/up, K3 down

The same native ABI already supports a single fused launch with `K_gate=2, K_up=2, K_down=3`. Its dispatcher selects the compiled `exl3_moe_kernel<0,256>` variant, whose gate/up and down GEMM sections switch on their respective scalar precision. This is materially simpler than expertwise banks because the precision is uniform across all experts within each projection.

The present Python wrapper is the limiting layer:

- `Exl3MoEMethod.bits` currently determines both `w13_trellis` and `w2_trellis` last dimensions. Split those allocation widths; gate/up remain combined and must share their precision in the minimal patch.
- Preserve mapped checkpoint names, all288 original expert IDs, four archive ranks, router/bias/shared/native tensors. LoadK2 gate/up fields andK3 down fields from the paired artifacts. Existing strict shape checks then validate each destination.
- Set a method-local `(gate,up,down)` tuple for the selected layer. Pass the three values separately to the existing native call rather than repeating the defaultK.
- `LinearEXL3` derives its ownK from `trellis.shape[-1]//16`, so individual projections and reconstructed-weight references already support the shape difference. Verify every loaded projection's inferredK against metadata.

Minimal metadata proposal, inside EXL3 quantization configuration:

```json
{
  "bits": 2,
  "layer_projection_bits": {
    "5": {"down_proj": 3},
    "32": {"down_proj": 3}
  }
}
```

Unlisted projections/layers use the default bit width. Reject dense/MTP layer IDs, unsupported fields and unequal gate/up precision in this initial combined-w13 implementation. The isolated prototype does not silently merge this map with a conflicting `layer_bits` override; a candidate must describe one unambiguous policy.

For each of four archive ranks at native288-expert geometry:

| Tensor | K2 baseline shape | K2gate/up +K3down shape |
|---|---|---|
| `w13_trellis` | [288,2,256,32,32] | unchanged |
| `w2_trellis` | [288,32,256,32] | [288,32,256,48] |

The extra storage is1 MiB per upgraded expert/down projection, or288 MiB per complete layer. Six down-only layers cost1.6875 GiB; twelve cost3.375 GiB. There is one native fused launch per layer, but the dynamicK0 variant's speed must be measured; no zero-overhead claim follows from source compatibility.

Four additional CPU tests validate per-layer defaults, immutable metadata, exact packed shapes/byte deltas, dynamic-dispatch selection and rejection of protected/unsupported cases. `projection-gpu/` contains an isolated proposed wrapper copy and bounded real8-expert proof script; no live image or server is patched. The planned GPU gate compares mixed real packed projections against individual EXL3 and reconstructed-FP32 references and checks CUDA graphs plus an actual K0 kernel trace. Its result is recorded separately when run.

## Projection GPU proof completed

The real8-expert layer5 K(2,2,3) test passed on de5c:384 packed fields, distinct w13/w2 packed widths32/48, actual `exl3_moe_kernel<0,256>`, finite results, and six graph replays. Max absolute fused-vs-reconstructed-FP32 error was7.14008e-5; peak allocated293,047,808B. `projection-gpu/PROJECTION_GPU_VALIDATION.json` binds raw logs, terminal exit0/noOOM, source files and kernel trace. This validates the projection-specific fused path only; expertwise two-bank execution and full-model quality remain separate gates.
