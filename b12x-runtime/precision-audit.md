# Arithmetic audit before any quality promotion

Source pins: vLLM 3aada67721bfdb8b98355207ca3780bb32e6f434; B12x 3b862805d2b7fd52e6fe507fd28038bd48797cf5.

The [Unsloth llama.cpp PR27754](https://github.com/ggml-org/llama.cpp/pull/27754) reports 89.6% to99.95% fixture agreement after disabling its unconditional cuBLAS TF32 mode, and warns about F32 MLA latent conversion toF16. This is engine fidelity evidence, not quantized-versus-BF16 quality evidence; it is not a quality score for our model.

| Path | Source finding | Implication |
|---|---|---|
| PyTorch F32 matmul | vllm/envs.py634 defaults highest; GPUWorker186 applies torch.set_float32_matmul_precision | No evidence of the same unconditional10-mantissa-bit cuBLAS default; explicitly record live worker configuration |
| mHC custom MMA | B12x _policy.py selects tf32_tma for supported hidden sizes and larger batches; _impl.py954 explicitly passes split_fp32_fn=True for FP32 projection weights; _kernels.py3002 accumulates high and residual-low products | NVIDIA_TF32_OVERRIDE does not control these handwritten MMA instructions. This is not equivalent to naive TF32 truncation, but must be compared numerically |
| MLA projections | mla_attention.py1318 stores absorbed W_UV/W_UK_T in activation dtype (BF16 here), then16-bit BMM; query fastpath requires BF16 output | Different arithmetic from a fullF32 latent reference exists by design; source inspection alone does not establish error magnitude |
| Sparse MLA | B12x backend supports BF16 query/output and packedFP8 orNVFP4 KV; fp8_ds_mla selected for baseline | Engine parity must independently account for FP8 KV, BF16 intermediate output, attention accumulation and selection |
| Quantization | Target K2 experts are unchanged; protected weights remain native; optional native288 MTP expertFP8 affects draft only | Target quantization quality and runtime numerical drift must be separate tables; currentK2 is a baseline, not accepted release |

Required gate: identical checkpoint/token IDs, same forced positions and teacher head arithmetic, compare offline reconstructed-weight reference to actual serving logprobs; report full KL where available, top1 agreement, top-k mass/tails, finite states and first divergence. Compare prefill and decode separately. If drift is material, isolate mHC, MLA, KDA, KV dtype and EXL3 kernel one at a time, without changing target quantization simultaneously. Do not attribute the current77% quantization agreement to a runtime issue without evidence.

No active557f server settings were changed. No claimed numerical parity or throughput result is derived from these source findings.

Upstream tests are kernel-level tolerances, not model agreement targets: sparse MLA tests use cosine>0.99 and rtol/atol0.05; mHC projection/mix tests use separate BF16-output and F32-scalar tolerances. Passing those tests would not establish99.95% model token agreement. The GLM mHC FP32 high+low dispatch is explicitly enabled in `_impl.py954-956`; some generic paths depend on whether `pre_mix` exists, so record actual dispatch rather than inferring from a function name.

## Confirmed default precision change caught in real loading

Actual probe4 on2384 logged `Quantizing LM head shards to NVFP4 with BF16 activations`. Jovian `envs.py1669,1679` defaults `VLLM_MXFP8_LM_HEAD=1` and `VLLM_MTP_NVFP4_LM_HEAD=1`. These are additional runtime quantizations of protected heads and are not acceptable under this study's native-sensitive-weight policy. Only the experimental probe was stopped, preserving its logs. Both flags are explicitly0 in our image and launch scripts fromR3 onward; final native-head source-value and dtype checks remain mandatory. This establishes a new-runtime default precision risk, not the cause of already measured offlineK2 quality loss.

The legacy-runtime dtype baseline is not identical to stored source dtypes. Probe5's first dtype mismatch was `index_kpool_compress_ape`: legacy allocatedF32, new pooled indexer allocatesBF16. Directly reading the pinned MTP safetensors header confirmed the source itself isBF16, shape[4,128]. The new pooling kernel explicitly promotes each load toF32 before score calculations. The follow-up gate compares all512 values exactly after F32 conversion and separately labels this legacy-upcast difference; it does not infer lost precision merely from differing parameter allocation dtypes. Jovian also split the legacy fused indexer `[wk; weights_proj]` tensor into two native matrices, requiring an explicit audit mapping rather than a model rewrite.
