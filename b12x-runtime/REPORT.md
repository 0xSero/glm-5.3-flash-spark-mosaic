# GLM Flash EXL3 + B12x source-build audit

Status: source compatibility checked; bounded source compilation is running. No completed image or B12x GPU execution has been accepted yet. Existing quality and serving lanes were not changed. Heavy source compilation is separated from the final Python overlay, so native-MTP corrections do not invalidate the compiled wheels.

## Pinned source and dependency contract

| Component | Exact source/version |
|---|---|
| Local Inference Lab vLLM, `dev/jovian-judgement` | `3aada67721bfdb8b98355207ca3780bb32e6f434` |
| Local Inference Lab / Luke Alonso B12x | `3b862805d2b7fd52e6fe507fd28038bd48797cf5` |
| ExLlamaV3 | Exact public-bootstrap 0.0.43 binary/package; extension `0eeb983b09dfe33451b8bc7625174320529b986a73f60b58bf4ebfa9f43ad9c0`; not freshly compiled |
| eugr Spark Docker reference | `346dc04fa11a4e1cb343153557e8b699b8488d30` |
| Torch / toolkit / Rust | 2.13.0+cu130 / CUDA13.0.88 / Rust1.95.0 |
| FlashInfer / Transformers | 0.6.17 / 5.16.1 |
| CUTLASS DSL / TVM-FFI | 4.6.2 / 0.1.11 |

The local observer image on Spark2822 already has the exact Torch, FlashInfer, Transformers, DSL and TVM versions declared by Jovian. CPU inspection confirmed nvcc, ninja, gcc and g++; git, cmake, cargo and protoc need installation. The Dockerfile reuses that existing image, builds vLLM native extensions against its Torch/CUDA and installs B12x from pinned source. It retains the exact verified public-bootstrap ExLlama package and independent GPU extension. No older vLLM binary is copied into the new fork. B12x's wheel is Python code: its CUDA kernels still require real GPU JIT and numerical tests.

The eugr recipe advances DSL to4.7.0 by explicitly rewriting upstream package requirements. This proposal keeps the matching native4.6.2 contract initially. B12x's optional new NVFP4/MXFP8 dense A16 path requires CUDA13.3 according to Jovian docs; this release preserves sensitive dense weights in BF16 and does not select that path. If actual required attention JIT reports a compiler incompatibility, qualify a newer compiler in a separate build instead of relabeling the current environment.

## What executes where

| Component | Proposed implementation | Evidence / remaining gate |
|---|---|---|
| Routed K2/K3 experts, including rankstack TP4 export on one GPU | Real ExLlamaV3 fused EXL3 kernels | Existing wrapper matches Jovian APIs; source port handles new file-backed weight transfer; real rebuilt-image import and GPU parity pending |
| GLM sparse MLA | B12x, `--attention-backend B12X` | Explicit GLM5Next sparse backend in Jovian; accepts FP8 cache; real prefill/decode and MTP pending |
| KDA decode | B12x auto-selection for compatible BF16 / head128 / bounded speculative columns | GLM sets `enable_b12x_kda_decode=True`; must inspect actual bound plans and profiler |
| KDA prefill | `--additional-config '{"kda_prefill_backend":"b12x"}'` | Auto does **not** select B12x prefill; explicit option fails when unsupported; test against default before performance claim |
| Hyperconnections | B12x mHC on supported SM120 family | Explicit GLM auto-selection; actual kernel attribution pending |
| Sensitive tensors and vision tower | Native BF16/F32 | No extra quantization, transformer image/video route present; end-to-end media acceptance pending |
| Native MTP | Independent BF16288-expert companion, one token first | Preserve target/draft quantization separation and strict shared embed/head loading; V2 correction must also be included |

Do **not** use `--moe-backend b12x` for this EXL3 model. Jovian's B12x documentation explicitly excludes EXL3/NF3 from that MoE backend. Installation is not proof of execution; require B12x attention/mHC/KDA attribution while EXL3 handles routed experts. No single-Spark PCIe all-reduce tuning is relevant at TP1.

## Port and build contents

`exl3-port/` contains the source-pinned EXL3 registration/loader port and six CPU contract tests. Apply it before the MTP patch. `apply_native_mtp_port.py` adapts native MTP: the draft keeps its own configuration and no target EXL3 quantization; real shared embedding/head tensors load before strict completeness checks. New Jovian additionally filters checkpoint prefixes, so those native shared weights must be explicitly included. Do not weaken completeness checks or quantize the protected MTP.

`build-patches/` contains two verbatim eugr patches with source hashes in `source-pins.json`. The SM121 allow-list patch applies. The CUDA-graph profiling-pool patch identifies the equivalent fix already present in Jovian and leaves that source unchanged.

`Dockerfile` uses four build jobs and retains ordinary CUDA graphs. `build.sh` uses the already-local observer image for a fast bootstrap and runs the heavy `compile.sh` in a named container limited to4CPUs,64GiB RAM and no swap (requires at least80GiB host memory available), with no GPU device. Its successful result is committed as a compiled base; final Python changes and CPU gates are a separate stage. `smoke_build.py` and the EXL3 port's CPU smoke require real native extension imports and record extension hashes. Their success only establishes image import/ABI compatibility.

For a standalone rebuild, the Dockerfile defaults to `ghcr.io/0xsero/glm53-flash-exl3-k2-dflash@sha256:5ea6d04d65d1a2f30f632815776fb765359529ab117818ba2b5e5c5480e34963`. The runtime agent verified this base's Torch2.13.0+cu130, toolkit13.0.88 and DSL4.6.2 match the observer. Independent anonymous registry requests fetched that exact manifest and config: linux/arm64,59layers (`public-base-manifest-proof.json`). The local candidate build overrides that reference to avoid another large transfer; release must record the actual final image digest.

## Acceptance still required

1. Build finishes; record immutable image ID, source pins, patches, installed dependency inventory and native-extension hashes. CPU smoke passes with CUDA uninitialized.
2. Exact physical REAP index loads at TP1; native MTP remains288 experts; no missing/unexpected tensor bypass. Both V1 and V2 draft configuration paths preserve native precision.
3. Actual finite output, MTP acceptance, CUDA graph replay, image and video prompts pass. Capture proves B12x kernels executed.
4. Allocate and exercise262,144 context with the real full model, activations, recurrent states and native MTP; configuration alone does not establish fit.
5. Matched quality baseline and workload sweeps establish correctness, useful speed, and stable long-running service before publishing or replacing the active recipe.

## Measured build failures and correction

- Attempt1 failed CMake configuration because the runtime-derived base had versioned NVRTC13.0.88 libraries but lacked the unversioned development linker entry. The build now creates a symlink to the existing same-version library. No CUDA library is replaced.
- Attempt2 found the repaired library, but an old `CUDA_NVRTC_LIB=CUDA_nvrtc_LIBRARY-NOTFOUND` cached value survived. The build invalidates only a cache containing that failed entry.
- Attempt3 compiled under2CPUs/12GiB. After low initial RSS, a controlled cached resume qualified4jobs/20GiB.
- Attempt4 hit the20GiB cgroup limit in a heavy SM120 CUTLASS translation unit (exit137, one OOM kill). The model quality GPU lane was unaffected. Logs, cgroup counters and objects were preserved.
- Attempt5 resumed4jobs with64GiB and no swap after a fresh116.49GiB MemAvailable check. Its observed peak already reached31,806,595,072bytes (~29.62GiB), proving the20GiB cap was insufficient. That attempt later completed the CUDA objects but failed the DeepGEMM host wrapper; no final image acceptance yet.

The final MTP overlay additionally implements the external multimodal-embedding protocol: target-produced image/video features replace the exact masked positions; OOV media placeholder IDs are masked before text lookup. CPU tensor tests pass for unchanged text, image/video positions, no input-ID mutation, and rejected malformed masks/counts. `mtp-embedding-cpu-evidence.json` is explicitly a pre-build method test. The Dockerfile reruns against the actual imported Jovian MTP class; actual GPU vision/MTP remains a separate release gate.

## Experimental and publication images

The `runtime-fast` stage inherits the compiled filesystem for immediate private testing. It must not be published. The default `runtime` stage starts again from the verified bootstrap image and exposes `/opt/wheels` through a read-only BuildKit stage mount while installing them. The compiled source trees, object files, Rust targets and build caches therefore do not become publication layers. Both stages apply the same Python patches and run the same real CPU import and installed MTP embedding tests.

The final image contains `/opt/build-identity.json` and `/opt/mtp-embedding-cpu-evidence.json`. These distinguish successful imports and CPU feature placement from the GPU runtime, capacity, media and throughput checks still required. Audit the complete final image, including its inherited bootstrap layers, before public release.

## Optional MTP expert FP8

`GLM53_MTP_EXPERT_FP8=1` opts into online FP8 for only the native draft's routed experts at constructor prefix `model.layers.45.mlp.experts`. The flag is unset by default, preserving all-BF16 MTP. The policy accepts only the original 288-expert `Glm5NextMTPModel` after both V1/V2 paths have separated its configuration from the EXL3 target. It rejects existing draft quantization and target architectures; dense, embedding and other protected weights retain native methods. Target weights and source checkpoint files are unchanged.

The legacy runtime's separate real-load experiment measured a 6.75 GiB saving and strict native nonexpert dtypes; that is not yet a Jovian runtime result. The final image runs CPU precision-scope tests and records `/opt/mtp-fp8-scope-cpu-evidence.json`. These tests explicitly mock only the final GPU backend constructor to check dispatch, without claiming quantization or inference. The selected image still needs actual FP8 draft load, forward, MTP acceptance, graph and media tests before using this option in a release recipe.

## Subsequent build repair evidence

| Attempt | Observed failure/result | Narrow correction |
|---|---|---|
| 5 | DeepGEMM host C++ compile could not find `cusparse.h`; expensive CUDA objects completed | Retain all objects and use the already-installed matching vendor math headers |
| 6 | Global `CPATH` let vendor-wheel CRT headers override nvcc's own headers, causing a `__cudaLaunch` macro mismatch | Stop the isolated retry; resume from the clean attempt-5 snapshot; no global include override |
| 7 | Final link and DeepGEMM succeeded with only two Ninja tasks; all CUDA objects reused; Rust stage started | Add `-idirafter` only to DeepGEMM's host compiler command and preserve unchanged patched Torch-header mtimes |

`build-patches/repair_build_paths.py` patches the Torch header in a staging directory and copies the result with `ONLY_IF_DIFFERENT`. It preserves identical generated content instead of invalidating 89 CUDA objects on every CMake configure. It does not skip dependencies when content changes. Its hash is recorded in both private and public source pins.

The runtime bootstrap contains a legacy EXL3 Python overlay outside the package wheel's RECORD. Finalization retires only the exact known SHA256 `41fc9bb42f3a555dbcef05b8b6c4fb1fa63dfe80e844f66078d160d4f3235db9` before applying the new port. Any other preexisting overlay is rejected.

The slim stage now starts from the exact public bootstrap digest and contains only `source-pins.public.json`, with the private observer-node field removed before any image layer is made. It records the actual bootstrap reference, installed distributions and installed overlay hashes. The bootstrap was anonymously verified and pulled on the build host; only 208,097 compressed bytes belonged to public layers outside the observer's shared rootfs content.

## ExLlama binary provenance

The vLLM wheel completed CUDA and Rust compilation. A subsequent attempt to build upstream ExLlama exposed x86-only AVX detection/reduction sources and missing vendor header paths on ARM. The failed source attempt is preserved. The accepted build strategy retains the exact public bootstrap's ExLlama 0.0.43 package and GPU extension, with mandatory package-tree and extension SHA256 verification before and after installing the new vLLM. The extension links Torch/CUDA independently, with no vLLM native-library dependency. Actual new-stack import and GPU load/forward gates remain required.

Comparing packaged sources to reference c5d9c657 found 317 identical files, including the GPU implementation, and six differing ARM CPU AVX/reduction files. This is a source-file comparison, not a claim that the reused binary was freshly compiled or attested from unmodified upstream. ExLlama's native CPU all-reduce is unsupported on ARM; this recipe uses TP1 GPU EXL3 kernels and does not invoke that backend. `exllama-bootstrap-pin.json`, full inventory and source-comparison receipts preserve these distinctions.

## Latest build and qualification state (2026-09-11)

New source-built Jovian vLLM and pinned B12x are installed in the slim public-bootstrap candidate `sha256:4ec5267998b60eaefc8d72bd837c8897a71724b4a7968d4db3ff307efa98736e` (25,611,760,700 bytes). The exact image transferred to the isolated2384 lane. This is an experimental baseline, not accepted quality or published output.

Docker builds apply checked source overlays; native import tests run afterward with the actual NVIDIA host driver exposed and `CUDA_VISIBLE_DEVICES=`. BuildKit has no driver library; CUDA link stubs are not used to claim imports passed. All four final slim CPU gates passed, with `cuda_initialized:false`, and receipts are under `slim-*.json`. A standalone script bug calling an instance `get_name` method on the class was corrected before final qualification.

`slim-dependency-audit.json` preserves the actual nonzero pip-check result. Torch requires NCCL2.29.7 exactly, which is installed. DeepEPv2 needs2.30.4+ and is unavailable; thisTP1 recipe does not use it. Six standalone ExLlama application dependencies remain absent and cusparselt has an ARM metadata warning. The isolated serving path's actual native imports passed; this is not a general ExLlama application installation guarantee.

No K2 quality promotion is authorized. `precision-audit.md` separates quantization quality from runtime arithmetic parity, including PyTorch highestF32 matmul, B12x mHC high+low TF32 decomposition, BF16 absorbed MLA projections and packedFP8 KV. GPU strict draft load, target load, semantic output, numerical parity, image/video, sustained262144 context and performance remain distinct pending gates.

| New runtime gate | Observed failure | Correction |
|---|---|---|
| Draft probe1, before weights | EXL3 override missing from ModelConfig ordered registry | One-entry checked registration patch; actual EngineArgs262144 config then passed |
| Draft probe2, constructor | C4 pooled indexer rejects block64 | Use block256; still requires actual capacity profiling |
| Draft probe3, constructor | V2 draft auto backend has no GLM512-head packedFP8 implementation | Explicit draft `attention_backend:B12X` as well as targetB12X |
| Draft probe4, real source loading | Default automatic MTP NVFP4 head conversion violates native protected-weight policy | Controlled stop, noOOM; disable both targetMXFP8 and MTPNVFP4 head flags before retry |

R3 image `sha256:afb74c791853d438810b27bf28994128f5baa2a3bf5c9b50b9ecdad7b9afffd5` includes the checked EXL3 config patch and disables both protected-head conversions. Every failed probe and its frozen source/logs are retained on2384. Failed attempts are not runtime or quality acceptance.

## R3 actual serving and bounded profile (22:05 UTC)

R3 is now running on the isolated Spark2384 lane as `glm53-b12x-k2full-fp8mtp-attempt2`, container `95fa3d37087adf09ff2b3ed9518d234577b11bdbd2d4bddc416509cec03e9f42`. `runtime-gates-r3.json` binds its inspect, logs, actual strict draft loader receipt, output checks and profiler evidence. The earlier slim/R1 source state is superseded by R3 for this runtime; no public release is accepted.

| Gate | Observed result |
|---|---|
| Actual V2 native draft load | 891 indexed source keys across five shards; strict runtime closure; 288 draft experts; only expert weights FP8; source-exact protected embedding/head samples |
| Target plus draft load | 133 target shards plus five draft shards; 97.89 GiB model; 887.87 seconds |
| KV admission | 751,007 aggregate tokens; requested context262144; packed FP8 MLA; C1 configured |
| Memory profile | KV5.84 GiB, peak activations4.78 GiB, CUDA graphs0.13 GiB |
| Hybrid block geometry | Requested256; allocator chose8704 to accommodate recurrent-state pages |
| Fresh arithmetic | Answer42, normal stop, thinking enabled |
| Fresh counting with thinking | Correct list at256 tokens but length stop;512-token retry exhausted budget in reasoning with no final content |
| Fresh counting without thinking | Normal stop but reasoning and literal `</think>` leaked into content; strict formatting failed |
| Native speculative counters | 555 drafted,503 accepted across these smoke/profile requests; not an independent accuracy metric |
| Bounded profiler | Eight engine iterations completed; profile stopped; raw trace preserved |

The profiler sums actual kernel events separately from nested graph annotations. In its short prefill-plus-decode window, the EXL3 MoE kernel consumed441.26 ms (34.38% of summed kernel duration), and the two main BF16 WMMA variants consumed318.20 ms (24.79%) and263.31 ms (20.52%). B12x KDA decode consumed7.53 ms (0.59%). These percentages are attribution within this capture, not wall-clock shares or a throughput benchmark. Full trace and kernel names are under `profiles/`.

The server is healthy, but the output failures prevent semantic/quality acceptance. Matched numerical parity, real262144-context requests, end-to-end image/video, steady-state throughput and publication remain outstanding. No causal claim is made about quantization versus engine arithmetic from these smoke outputs.

## Staged selective-layer precision

`mixed-layer-precision/` adds method-local `layer_bits` handling without changing shared EXL3 configuration. Metadata uses default `bits:2` and `layer_bits:{"5":3,"32":3}`; all288 experts remain. Six CPU policy tests and an actual imported Exl3Config gate passed for all42 routed target layers and rankstack4. This is staged only: R3 remains unchanged and no mixed-weight GPU or quality result is claimed. Mixed artifacts require their own manifest validator and must not be presented as REAP-pruned checkpoints.
