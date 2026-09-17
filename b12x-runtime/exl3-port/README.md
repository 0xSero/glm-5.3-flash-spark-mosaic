# EXL3 port into pinned Jovian vLLM

Status: source compatibility checked; six CPU tensor/source tests pass. **The new
image has not passed real imports, GPU kernels, model loading, CUDA graphs or API
acceptance in this task.** Native MTP is a separate patch and acceptance gate.

This ports the proven campaign wrapper from
`runtime/spark/exl3.py` / the preserved native-MTP source snapshot, original SHA256
`41fc9bb42f3a555dbcef05b8b6c4fb1fa63dfe80e844f66078d160d4f3235db9`,
into Local Inference Lab vLLM commit
`3aada67721bfdb8b98355207ca3780bb32e6f434` (`dev/jovian-judgement`).
The wrapper retains its Apache-2.0 notice and original MiaAI-Lab attribution.
ExLlamaV3 remains source-pinned to `c5d9c657966ffeeaa9353f0cc899f18629da4a13`.
Build both vLLM and ExLlama native extensions against the chosen image's actual
Torch/CUDA. No old `vllm._C` or compiled extension is copied across source/ABI changes.

## Narrow changes

| Surface | Evidence and resulting change |
|---|---|
| Quantization registry | Add EXL3 to the Literal, lazy import and class map. Same three registration points as [Gilded prior art](https://github.com/local-inference-lab/vllm/blob/fa033bd4e1b16d9d729ad94be2d87da5a13210ce/vllm/model_executor/layers/quantization/__init__.py). Its frozen source/hash is recorded here. |
| FusedMoEMethodBase | Actual create_weights, apply and quant-config argument lists still match the proven wrapper; no adapter shim. |
| GLM5Next loader | Current hand-written loader still calls the parameter's own weight_loader. The actual expert mapper preserves `rankR.suffix` through `w13_rankR.suffix` or `w2_rankR.suffix`; no architecture patch. |
| Checkpoint transport | Use Jovian's copy_weight instead of direct copy_; preserve file-backed descriptors for packed rank-stacked tensors. Only transformed inputs, including scalar four-byte MCG, materialize. Flush queued transfers before marker validation/pointer construction. |
| Payload contract | K2/K3/K4–K6 support retained; liveTP1/archiveTP4 and512-wide virtual slices unchanged. Logical expert count is dynamic, including REAP256/176. No expert expansion to persistent BF16. |
| Native components | Dense/shared/attention linears keep UnquantizedLinearMethod. Native MTP must have a separate unquantized draft config; never pass target EXL3 config into its native288-expert layer. Runtime agent owns that fix. |

Apply to a clean source checkout during image build, **before** the separate
native-MTP patch (which may modify audited source files):

```bash
python3 /patches/exl3-port/apply_port.py --root /opt/src/vllm --check-only
python3 /patches/exl3-port/apply_port.py --root /opt/src/vllm
```

The script verifies exact upstream API-file hashes and refuses existing EXL3 files
or changed source. `exl3-jovian.patch` is the equivalent reviewable unified diff;
the apply script supplies stronger identity checks. No upstream checkout was
modified by the implementation task. The original namespace-only ExLlama import
helper is unchanged; no replacement math or serving ABI shim was added.

The initial dependency plan from the image owner is Torch2.13.0+cu130,
FlashInfer0.6.17, CUTLASS DSL4.6.2, TVM-FFI0.1.11 and Transformers5.16.1, with full
Jovian/ExLlama source builds. Those installed dependencies must be verified in the
resulting image; the plan alone is not an import test.

## B12x and EXL3 own different operations

Use the new GLM B12x KDA, sparse MLA and mHC paths where supported. Keep routed
experts on the EXL3 custom method. Do not force `--moe-backend b12x` for these
weights: [upstream B12x docs](https://github.com/local-inference-lab/vllm/blob/3aada67721bfdb8b98355207ca3780bb32e6f434/docs/features/quantization/b12x.md)
explicitly exclude EXL3. Target liveTP1; no expert parallelism/EPLB. The existing
wrapper can fall back to its Python expert loop if fused initialization fails;
acceptance must reject that fallback and prove `fused_moe=exl3_moe` plus completed
CUDA graph capture. A successful import cannot establish this.

## Verification

```bash
python3 -m unittest discover -s b12x-runtime/exl3-port/tests -v
```

Six local CPU tests use real Torch2.10 CPU tensors and exact selected source method
bodies. They cover source pin/reapplication rejection, new base API arguments,
K2/K3 config, native-linear dispatch, all four archive-rank mappings and exact
payload loading, wrong-rank rejection, file-descriptor transport and scalar MCG
materialization. Serving-only imports/base initialization are intentionally omitted
in these isolated tests; they are not a substitute for actual runtime imports.
No CUDA context is initialized.

After the image rebuild, run without exposing GPUs:

```bash
python3 /patches/exl3-port/cpu_import_smoke.py
```

This script performs **real** vLLM, registry, wrapper, ExLlama class and compiled
extension imports, checks required fused symbols and records dependency/file
hashes. It has been prepared but not executed on the new image. It installs no
stub modules for missing serving APIs. After that, the runtime owner must perform
native-MTP constructor/key audit, actual all-shard loading, graph capture,
text/image/video requests, matched offline/serving parity and the speed matrix.

### EngineArgs registration correction

The first actual Jovian EngineArgs gate found that EXL3 was registered in the quantizer registry but missing from ModelConfig's ordered quantization override list. ModelConfig therefore rejected it before any weights loaded. This was a missed integration gate, not a compiled-kernel failure.

`apply_config_override.py --model-file <installed vllm/config/model.py>` adds only `exl3` to that ordered list. It accepts exactly the pinned original file or its exact already-patched output, rejects unrelated source changes, and writes atomically. It also accepts `--root <vllm-checkout>` and `--check-only`. Original SHA256 `1d1429b0536a9ea30d369d1e8f8384732abb18bfad9fd2cde40eeec0eca1ca20`; corrected SHA256 `da9557e1e46bcf51334757cab5e176cd2d5e2095ef3a601853fafb470f5f4265`.

Fresh `apply_port.py` now includes the same ModelConfig correction. The existing six port tests and three new registration tests pass. This changes only Python files and requires no compiled-binary replacement. Actual EngineArgs validation is performed by the runtime owner's existing `probe_native_mtp_real.py` with `GLM53_CONFIG_ONLY=1`; source tests alone do not establish that gate or any model loading/inference success.

Actual configuration validation subsequently passed in the corrected image: target `Exl3Config`, target/draft288 experts,262144 context, and`fp8_ds_mla`. Container`b4d7a188004d860008bddda8ec88ffd7466f2036e3fb0f5b424a5ecbd0159488` exited0 before any model weights loaded. Configuration initialization did create a CUDA context. `ENGINE_CONFIG_VALIDATION.json` binds the original runtime-owner log and terminal inspect hashes. A later draft-constructor check independently found Jovian's C4 pooled-indexer block-size requirement (divisible by256); the runtime owner handles that configuration change. Neither gate is full model-serving acceptance.
