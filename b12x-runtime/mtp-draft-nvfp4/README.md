# Experimental native MTP expert compression

This experiment quantizes only the native layer-45 MTP routed experts to NVFP4 and uses B12x W4A16 execution on SM121. Main-model weights, native MTP attention, router, shared expert, embedding and output head remain protected. Nothing here is enabled in the running GLM server or published recipe.

The current checkpoint contains 13.5 GiB of native BF16 MTP routed weights. The measured FP8 representation occupies about 6.75 GiB; nominal NVFP4 weights, group-16 scales and expert scales occupy about 3.797 GiB. Matched full-draft loads now measured **3,170,885,120 bytes (2.9531 GiB) lower resident CUDA allocation**. This is not yet a measured full-server saving with target/KV/graphs. Full loading, prepared storage, workspace, graph memory, proposal acceptance and useful output speed are separate gates.

## Evidence and failures

| Attempt | Change | Observed outcome |
|---|---|---|
| GPU 1 | Eight real native experts; online packing and B12x selection | Packing, native router/shared equality and W4A16 selection passed. First forward failed because the component harness had not initialized vLLM's workspace manager. |
| GPU 2 | Initialize workspace; add independent routed-only reference | Ordinary-input reference passed. Changed-input graph/eager comparison failed its original tolerance. |
| GPU 3 | Record module-level graph diagnostics | Same variation observed; the diagnostic's manual INT64 route IDs needed conversion before capture. This was a harness error, not the serving router. |
| GPU 4 | Preallocate INT32 routes; unchanged-input repetitions | Variation persisted across both eager/eager and graph/graph repetitions. Router and shared expert were bitwise stable; fixed-route routed experts varied. |
| GPU 5 | Observe actual physical launch flags | Direct routing selected `tc_decode_fused_sum=True`, consistent with the routed reduction variation. |
| GPU 6 | Packed routing only; same weights, precision, fast-math and tolerances | Actual `tc_decode_fused_sum=False`. M1 graph replay was bitwise equal for three changed inputs. The subsequent saturating-input FP32 reference failed; its omitted BF16 rounding was investigated in the next attempt. |

| GPU 7 | Independent source-pinned BF16 packed reference; unchanged tolerances | All five reference cells, including saturation, passed. All 15 changed-input graph replays across tested M1/2/4/8 were bitwise equal. Peak CUDA allocation 576.065 MiB. |

The independent reference initially decoded exact FP4/group-scale values but kept FP32 GEMM and activation intermediates. Source inspection found explicit BF16 rounding boundaries in the packed kernel. Those differences require a separately recorded arithmetic reference; failed checks must not be erased or passed by silently widening tolerances.

The eight-expert test exercises eight-of-eight routing. It does **not** establish sparse selection across 288 experts, MTP output quality, full-server memory admission or speed. Full draft loading was separately tested below. In the first ordinary-input test, routed-only output error versus native BF16 was approximately 15.5% relative L2 on synthetic inputs. Including the unchanged shared expert reduces that number and must not replace the routed-only report. Neither number is a whole-model quality score.

## Reproduction boundaries

`policy.py` reuses the upstream online weight quantizer with a separately gated SM121/B12x constructor. It does not remove the upstream SM100 guard. `cpu_scope.py` checks actual installed imports and protected precision dispatch; GPU selection is explicitly mocked in that CPU test. Both head-quantization flags must be explicitly zero at the full-draft entry point.

`gpu_component.py` performs manual eight-expert loading, independent weight reconstruction and graph checks, with a 2 GiB PyTorch allocation ceiling. `packed_routes.py` is a test-only, source-pinned physical-launch observer and pre-forward plan replacement. It must never mutate plans in a serving process or after capture. The source signature and six CPU helper tests pass.

`launch_component.py` checks the exact completed quality container and an idle GPU, preserves unique attempt directories, and freezes source copies before execution. Attempts 5 and 6 use the same frozen script and differ in the explicit packed-route setting. Every failed container and result remains retained.

`full_load.py` passed the separate memory gate: all 288 experts through the production native MTP loader, strict parameter completeness, native dtype audit, and full-value embedding/head comparisons. FP8 and NVFP4 use the same image and source view in separate containers. It loads no target weights, allocates no KV cache and starts no API; a passing load cannot be reported as serving acceptance.

Do not promote this experiment until the arithmetic and graph checks, full loader, actual memory saving, native protected-weight audit, and fresh full-server output/MTP acceptance all pass. Speed requires matched useful-output measurements.

## Matched full-draft load results

| Mode | Resident CUDA bytes | Peak CUDA bytes | Terminal |
|---|---:|---:|---|
| FP8 | 10,255,971,840 | 24,734,712,320 | Exit 0, no OOM |
| NVFP4 | 7,085,086,720 | 22,905,996,800 | Exit 0, no OOM |

Both use the same native view and protected parameter bytes (2,940,779,648). Full embedding/head values were compared; other protected parameters passed strict loading and dtype audits. The prepared INT32 tensors contain eight packed FP4 codes per word; INT32 storage is not 32-bit weight precision. The first NVFP4 load audit wrongly expected UINT8 and failed after successful loading; the corrected byte-count/dtype audit passed in a new retained attempt.

Seals: `FULL_MEMORY_COMPARISON.json` and `GPU_COMPONENT_ACCEPTANCE.json`. Full target/KV/postgraph/vision admission, native-MTP acceptance and matched useful-output speed remain pending. A separate runtime integration lives in `../mtp-nvfp4-runtime`; these test plan-mutation helpers must not be used in a live server.
