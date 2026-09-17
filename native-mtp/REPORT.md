# Native GLM-5.3-Flash MTP audit and proposed implementation

Status: source-audited, CPU config transform verified, three metadata contract tests passed, proposal syntax compiled. No GPU allocation or serving claim. Production runtimes untouched.

## Existing support and exact problem

Pinned manager512 runtime image `sha256:4cc287d0555c9425ab2e483441ebbb2bb6efe984415230e06c5fd97d004d5bf7` already contains `Glm5NextMTP`, registry architecture `Glm5NextMTPModel`, native sparse-attention MTP forward with shared top-k indices, and speculative method dispatch. The main model loader skips layer45 intentionally: the speculative draft loads that layer separately.

A same-checkpoint `method:mtp` invocation implicitly inherits target EXL3 quantization. Its selective loader applies EXL3 to every RoutedExperts instance, so native BF16 MTP experts would receive incompatible packed EXL3 allocations. A pruned target config also has fewer routed experts than the retained288-expert companion. Neither issue requires rewriting the MTP forward.

Use an explicit `/mtp` draft path with no quantization_config and the original unpruned288-expert config. Keep the main REAP target's reduced expert count and EXL3 config separately. In SpeculativeConfig, explicit model path avoids the implicit target quantization inheritance; CPU transform verified glm5_next -> glm5_next_mtp/Glm5NextMTPModel with quantizationNone and288experts.

## Proposed minimal changes

`native-mtp-proposal.patch` contains two source changes:

1. MTP loader consumes real retained base embedding/head tensors into its temporary shared parameters. They are absent at MTP-specific checkpoint names. Native unquantized draft uses strict weight completeness checking BEFORE proposer shares parameters; without explicit loading these would fail or remain uninitialized. Existing proposer subsequently shares target embedding and lm_head, including per-MTP shared_head.head, releasing duplicate copies. No fake loaded flags or disabled completeness checks.
2. Add Glm5NextForConditionalGeneration to the multimodal proposer list reading image_token_id. Otherwise its fallback reads image_token_index, absent in the original GLM config. Target vision remains native; draft uses text-token proposals and target hidden state.

No EXL3 target-loader changes or MTP quantization required. These patches still require real load/forward/graph acceptance.

## Build and metadata staging

`prepare_native_mtp_view.py --source CANDIDATE --output FRESH_MTP_VIEW` uses candidate `source-config.json` when available; `--native-config ORIGINAL_CONFIG` is an explicit alternative. It verifies889 native companion tensors plus embedding/head, nativeBF16/F32 storage,288 router dimensions, and864expert projections. It creates only metadata and read-only-serving symlinks. Mount the same source directory at `/native-source`; mount the generated view at `/mtp`.

Candidate builder contract agreed with glm_candidate_quality: preserve source-config.json verbatim; candidate config has native_mtp_n_routed_experts288 and reduced target n_routed_experts; MTP45 untouched; expert indices and router rows reindexed consistently in layers3..44. No MTP expert pruning.

Build from this directory using `docker build -f Dockerfile.proposed -t local/glm53-reap-native-mtp:candidate .` on ARM64. `start-native-mtp-candidate.sh` provides the initial GPU launch candidate: TP1, EXL3 target, BF16/native MTP, FP8 KV,262144context, one speculative token, full-decode graphs capture2,2048chunk. Start conservatively with one MTP token, validate, then measure depths3/5/7 rather than assume more is faster. Launcher does not qualify artifact quality; run only after independent candidate sealing.

## Memory requirement

Read actual safetensors headers on557f:889 MTP tensors consume14,865,185,408bytes=13.8442827463GiB. Source MTP contains288nativeBF16 experts. Add this to target resident weights; it was skipped in the old89.89GiB K2 baseline. Initial embedding/head copies add~2.36328125GiB before proposer shares them. Runtime/KV/graph/workspace/vision costs are additional. Native MTP is substantially larger than the old DFlash draft. Plan REAP retention accordingly.

## Required next gates

- Rehash retained companion provenance against source; complete candidate structural audit.
- CPU full ModelConfig admission and loader-key coverage on actual candidate, not only the verified config override.
- Initial actual native MTP weight load; inspect no unexpected/uninitialized weights, exact original288experts, and shared embedding/head identities.
- Compare MTP-on and MTP-off deterministic completions; expose accepted/rejected draft token counts and finite logits.
- Validate target+draft full graphs, long262144context retrieval, paired image/video, and short/long prefill/decode.
- Confirm runtime memory reserve under long prefill and native MTP activation peak.

Existing source snapshots and hashes are under source/ and source-manifest.json. This proposal preserves the native architecture and uses the existing serving stack; the separate Transformers observation evaluator is not substituted as the server.

## REAP reduced-count and K3 loader qualification

Actual installed candidate image EXL3 source SHA256 `41fc9bb42f3a555dbcef05b8b6c4fb1fa63dfe80e844f66078d160d4f3235db9`. `test_reap_loader_cpu.py` extracts and executes its real methods, with actual Torch CPU tensors and meta-only full-size weight allocation, without GPU.

All four combinations176/192experts × K2/K3 passed:32packedparameters perlayer; gate/up trellis shape[N,2,256,32,K*16]; down[N,32,256,K*16]; actual copy-loader finalexpert writes acrossfourranks; actual RoutedExperts name mapper yields w13_rankR/w2_rankR and preserves expert/shardidentity; topk8 expands32 virtualroutes. Virtual expert counts704/768 computed dynamically. K3 uses48trelliswords; no hardcodedK2/288/1152allocation found. Only immutablearchivegeometryTP4 and512-width slices are hardcoded, appropriate for this source.

Fresh source-q3/config.json on557f confirms n_group1/topk_group1, topk8, sigmoid, noaux_tc, normalizeTrue, scale2.5. CPU execution of actual grouped_topk proves exact selectedoriginalexpertIDs and exactfloat32routeweights for reduced176/192reindexed candidates against the original288router with removedexperts masked. There are no multiple group boundaries to preserve; topk8 is not the groupcount. Both candidates' weights sum2.5. Fused CUDA router remains an inference acceptance gate; no structuralgroupsemanticsdefect observed.

Original launch guards hardcodedK2/133shards/583090tensors belong to old launch scripts, not loader. Newcandidate launcher avoids these artifact-specific counts. NativeMTP retains its separate288expertconfig and nativequantNone. No additional runtimepatch or Dockerfilechange was necessary for REAP/K3.

Evidence: reap-loader-cpu-evidence.json. These tests cover loader geometry, CPUcopy, routingmapping and CPUgroupsemantics; they do not claim fusedCUDAkernel or fullmodelacceptance.

## Prepared exact K2 native-MTP control

Created557f `/home/valentine/glm53-single-spark-release-20260911/native-mtp-k2-control` as fresh CPUmetadata/symlink view. Full SHA256 of five retained shards matched originalQ3filehashinventory and existingK2manifest;891protected keyclosure verified. Receipt `k2-control-protected-verification.json`. ExistingK2source untouched. Source/target: `/home/valentine/flash-experimental-staging-20260906/model`; draftconfigoriginalQ3native288/noquant. GPUlaunch/acceptance remains parentowned.

## Actual K2 attempt1 failure and r2 correction

Attempt1 CID9756d306b67f758c55bc90beb243d2ccda9ad524f057ab9f2557f17513cd2744 loaded all133targetshards and enabled42EXL3fusedMoElayers, then failed during native MTP load with missing `model.layers.45.mtp_block.mlp.experts.routed_experts.w2_weight`. NotOOM. Engine was neverready; no request sent. Preserved557f `native-mtp/failures/k2-attempt1-quant-leak.log`.

Rootcause: generic proposer `_create_draft_vllm_config` retained target model_config AND target quant_config even when separate draft ModelConfig identified native288/noquant. GLM constructors consume VllmConfig, therefore allocated EXL3packedMTP instead ofnativeBF16. R2 uses existingStep3.5pattern, gatedtoGlm5NextMTPModel: replace model_config withdraft andquant_config withget_draft_quant_config(base). This also protects REAP176target/native288draftgeometry.

R2image557f `local/glm53-reap-native-mtp:20260911-r2`, IDb49089b5db74da49ce4ce4605f3537171ac6a72cae314d5f5bed77aabaf44850. Actual MTP constructor plus891nativeheader-shapedMETAweights ranthrough real loader andstrictDefaultModelLoadertracking:27runtimeparams,allweightsMETA,native288,noquant,targetEXL3unchanged. 16MiBGPUbufferallocated; no weightpayloadallocated. Metadataharnesssetupfailures (classimport/currentconfig/defaultFP32attention) preserved inmeta-audit-r2*.log, resolvedtoactualBF16/FP8KVsettings. `native-mtp-keyclosure-r2.json` ispassedclosureevidence, notforwardacceptance.

R2K2servingreloadstarted2026-09-11T19:06:35Z CID1001952ca26da230945dc5a3133c456f9622078a75396b07eba62a75b815af51; log `native-mtp-k2-control-launch-r2.log`,262144,MTP1. No readiness or generationclaimyet.

Backendidentity: this baselineuseslegacyEXL3extension+FlashinferMLASparseSM120. It doesnot contain B12x; finalrequestedB12xqualification remainsseparate main-trackwork.

## Actual V2 dispatch correction (R3)

R2 serving failed at 19:17:26Z with the same missing native w2_weight. The earlier META closure covered the V1 proposer helper, while the production launcher sets VLLM_USE_V2_MODEL_RUNNER=1 and dispatches through v1/worker/gpu/spec_decode/eagle/utils.py. That independent loader still passed target EXL3 configuration into the GLM native constructor. R2 META closure was valid for its tested path but did not cover production dispatch; it was insufficient as a launch gate.

R3 additionally binds native draft model_config and get_draft_quant_config in the actual V2 load_eagle_model function, gated to Glm5NextMTPModel. Both V1 and V2 fixes are retained. Image local/glm53-reap-native-mtp:20260911-r3 has ID cd6ce6e10276f856d9ec617c702362ffc45114cad6f26ddcb6a39ef5a2ac7e37. R2 failure preserved in failures/k2-attempt2-v2-quant-leak.log. Before another full target reload, gates now require actual V2 META constructor/key closure and actual V2 native draft-only real weight loading with strict completeness and source-value comparisons. Neither gate alone claims forward or serving acceptance.

R3 actual V2 META closure passed (native-mtp-keyclosure-r3.json). Actual V2 draft-only native weight load also passed: five real retained shards, strict completeness26 runtime parameters/27 tracked;891 native index keys;288 native experts; target EXL3 versus draft quantization None; all draft parameters CUDA;17,419,522,048 allocated bytes. Embedding/head source samples matched exactly after loading. Source files remain immutable. First real probe lacked the engine current-config context and failed before loading; preserved real-load-r3.log. Corrected harness reproduces normal engine config context and succeeded in real-load-r3b.log,129.18 seconds loading. This validates real construction, loading and source-value checks, not generation.

Full-K2 R3 serving attempt started19:27Z on557f (wrapper PID1852938), log native-mtp-k2-control-launch-r3.log, retaining262144 configured context and MTP1. GPU memory admission, graphs, output and accepted-draft counters remain required.

R3 first full launch failed before weights at startup reservation:114.03GiB free versus114.39GiB requested at0.94. This is not intrinsic model/KV OOM. Parent confirmed CPU builder had already finished, so builder causation is not established. Preserved failures/k2-attempt3-startup-free-memory.log. An identical retry was stopped before engine/weights once parent clarified this. Launcher now exposes GLM53_MEMORY_FRACTION, default0.93. Current R3/m93 attempt PID1854920 uses identical262144/MTP1 and records native-mtp-k2-control-launch-r3-m93.log. No OS service or global-cache changes were made.

R3/0.93 completed actual target+native-MTP loading and profiling, then failed KV admission at19:44:08Z. Measured model memory103.86GiB; estimated graph memory2.01GiB; available KV memory negative1.13GiB. Encoder profile budget32242tokens/onevideo; hybrid attention block adjusted6912. Preserved failures/k2-attempt4-negative-kv-memory.log and experiment-ledger.json. No endpoint or fresh completion succeeded. The requested262144context is below source native maximum1048576. No layout-copy patch is needed: that step passed. Next candidate selection should use measured memory headroom and quality evidence. Draft external multimodal embeddings are currently not supported by the class; V2 warns it uses text-only draft inputs while target vision remains active. B12x port owner is auditing the proper embedding-merge protocol.

## K2 keep256 native-MTP capacity and text success

K2keep256 with all288 native BF16 MTP experts loaded95.82GiB (8.04GiB below fullK2). At0.93, requested262144context/C1 admitted693225 aggregate KV tokens using6.38GiB; graphs captured for target and MTP. Actual graph memory0.06GiB versus estimated0.97GiB. API startup completed20:02:27Z on557f127.0.0.1:18080.

Initial thinking-off smoke exposed reasoning as content and failed exact formatting, because the source template unconditionally opens<think> and has no enable_thinking:false implementation; that request flag disabled parser separation only. No template/model change was made. With thinking:true, fresh math/list responses were exactly42 and1–20, both normal stops with reasoning separated; native counters increased65drafted/63accepted. These are short correctness/counter checks, not speed benchmarks. Original failed smoke preserved; successful artifact native-mtp-k256-thinking-smoke.json. Frozen launch/runtime/log evidence: runtime-receipt-k256-r3-m93.json. Long-context, vision and sustained/concurrent quality/performance acceptance remain parent-owned gates. This legacy runtime does not contain B12x; final B12x qualification remains separate.

Parent took ownership of557f baseline bench after this proof. A separate, explicitly authorized2822 draft-only onlineFP8 routed-expert probe is now staged under this workspace; original protected files remain unchanged. MTPExpertOnlyFp8Config allowlists the single routed-expert module and explicitly leaves all nonexpert linear modules unquantized; real load, all-nonexpert dtype comparison and source embedding/head comparisons must pass before proposing serving integration.

## Optional routed-expert FP8 native MTP feasibility, real loading passed

Parent authorized an isolated2822 probe after reviewing user wording: expert-only quantization does not literally forbid MTP routed-expert quantization; all-nonexpert-native remains enforced. Original files remain immutable. Native MTP algorithm is unchanged; no target weights were allocated or quantized by this probe.

The existing online FP8 route is Fp8PerTensorOnlineMoEMethod, selected only for constructor prefix model.layers.45.mlp.experts. Runtime parameter namespace additionally contains mtp_block. The guard caught the initial prefix mismatch before loading; preserved expert-fp8-real-load.log. Corrected probe expert-fp8-real-load-r2.log loaded all5retained source shards/891native index keys in104.29s. Strict completeness28runtimeparams/29tracked passed. Both routed tensors retained288experts and became float8_e4m3fn; FP32 scales consume2304bytes. All24nonexpert parameters preserve baseline native dtype; source embedding/head samples match bitwise. Native attention/router/shared experts/embed/head remain unquantized.

Actual draft-only allocated bytes10,171,766,272 versus17,419,522,048 for all-native draft:6.74999GiB saved. TRITON FP8 MoE selected. This is a real load/precision-scope proof, not a forward, speculative acceptance, latency or full-target serving result. Distribution preservation from draft-only quantization still depends on the correct native speculative verifier and actual runtime validation. The source checkpoint is untouched and remains native; conversion happens on load. Receipt native-mtp-expert-fp8-real-load.json and probe policy mtp_expert_fp8.py.

2822probe exited and released GPU; parent owns resource hold and next experiment.557fK2keep256 baseline stays running for parent-owned acceptance/benchmark work. Final B12x image remains a separate build/qualification; its default MTP is still native BF16 until parent chooses otherwise.

Parent completed the keep256 baseline and released557f server ownership. Preserved final inspect/log, copied both baseline result directories, and retained immutable R3 image plus rollback-r3-native/rollback-k256-r3.sh. R4 adds explicit GLM53_MTP_EXPERT_FP8 opt-in (default0) to V1/V2 draft configuration; only original288-expert native GLM draft configs are accepted and previously quantized draft configs fail closed. CPU opt-in tests passed without CUDA initialization. R4 image4f06a26d872ddd6f268a1f99e85940ea630a3cbecc33c7cc8b0038faf8c0851b.

Full unpruned K2 target plus routed-expert-only FP8 MTP launch started on557f after stopping the completed keep256 experiment. Context262144, fraction0.93, maximum-video profile, C1 and graph settings are unchanged. Log native-mtp-k2full-fp8draft-r4.log. Actual full serving/forward/KV/MTP acceptance remains pending; original source weights untouched.

## Full original K2 plus expert-only FP8 native MTP: actual serving passed

R4 finished loading97.11GiB (6.75GiB below full target plus all-native draft). At.93, same32242-token video encoder profile, requested262144/C1 admitted460208 aggregate KV tokens,4.22GiB; actual graph.14GiB versus estimate.79GiB. API started20:41:08Z. Fresh math/list prompts produced exactly42 and1–20, both normal stops, with thinking correctly separated. Native MTP counters advanced65drafted/63accepted. A separate raw-completions contract returned20token_id:N logprob alternatives with top_k1 and exact tokenIDs. Frozen receipt runtime-receipt-k2full-fp8draft-r4.json binds immutable inspect/log/smoke/probe hashes. Parent took generation ownership after these gates.

This point keeps all288 original target experts at existing K2 precision and all288 draft experts, quantizing only draft routed weights online toFP8. Strict source/nonexpert native tests passed in the actual draft-only probe, and full serving now passed forward verification. It is still a legacy baseline withoutB12x; draft external image/video embeddings remain absent and must be qualified on the new overlay. Long-context, actual image/video, quality parity, throughput and concurrency remain separate acceptance gates.

Prepared CPU-only fixed launch controls and dynamic-depth source audit in launch-controls-audit.md. All36 fixed depth/slot/chunk combinations have explicit graphcapacity; three CPUtests passed includingactual upstream schedule lookup. Legacy V2 has a disconnected dynamic-depth control, so do not use it. NewJovian passes scheduled depth through V2 but dynamic mode disables fusedmultistepdrafting. No active runtime was changed by this audit.
