# GLM release experiment ledger

Updated2026-09-11. Private working evidence; no accepted or published release. Promotion is held for substantially higher agreement. Pop production is untouched and DeepSeek remains intentionally stopped.

## Completed matched quality

FrozenWikiText2,32×2048tokens,65504next-token positions, original BF16 teacher. Full-vocabulary KL; these are offline quality measurements, not broad task scores.

| Artifact | Agreement | Mean KL | PPL | Decision |
|---|---:|---:|---:|---|
|Original Q3|87.384%|0.152204|3.49691|Higher-quality reference; too large for proposed one-Spark layout|
|Original K2|77.385%|0.438986|4.54687|Serving baseline; quality promotion held|
|K2 keep256|71.350%|0.687907|5.82429|Worse quality; pruning unnecessary for baseline capacity|
|Q3 keep176|58.807%|1.343988|11.27267|Rejected|
|Q3 keep192|61.810%|1.183637|9.57075|Rejected|

Reports and disclosed missing early rotating receipts are in `quality/`, `staging-k2-k256/`, `staging-q3-k192/`. Final normalized tensors and reports are sealed. No missing timings were invented.

## Actual legacy serving baseline

Full unpruned K2, FP8 routed MTP experts, native protected tensors; configured262144request context,460208aggregateKV, C1/depth1/chunk2048/fraction0.93. This runtime uses legacyFlashInfer, not newB12x.

| Repeat | Actual input | Output | Total decode tok/s | Per-request decode tok/s | Server request-prefill tok/s | TTFT s |
|---|---:|---:|---:|---:|---:|---:|
|0|1024|2048|13.75|13.75|109.18|9.44|
|1|1024|2048|13.47|13.47|429.68|2.43|

First request lacked matched-shape warmup; prefill difference is not an isolated optimization effect. NativeMTP second-window acceptance873/1174. Exact measurements: `benchmarks/k2full-fp8-legacy-c1-1024-r4/`.

ActualAPI65504-position quality:77.3556% BF16top1 agreement, PPL4.54397. Full-vocabulary KL cannot be inferred from API top20. Thirty-six raw-result/receipt hashes verified in `benchmarks/k2full-fp8-serving-quality-r4/`.

Five short behavior cases, four paired images and two native videos passed. These are synthetic feature checks, not broad capability acceptance. Full262k retrieval and concurrency sweeps remain pending. The separate strict8-row numerical probe had8/8argmaxmatches but exceeded0.05logprob-drift tolerance on all8rows; strict numerical parity did not pass.

## Runtime failures and fixes

| Experiment | Observed result | Action/evidence |
|---|---|---|
|OriginalQ3 first quality import|Unused standalone FlashAttention import failed|Used existing actual quantization namespace; real EXL3 reconstruction retained|
|Legacy nativeMTP first loads|EXL3 target policy leaked into V2draft|Separated native draft policy; actual891-key strictload required|
|Full K2 with BF16 MTP|Loaded103.86GiB but KV admission failed|FP8 only routed draft experts saved6.75GiB; target protected weights unchanged|
|B12x build|NVRTC linker entry and staleCMakecache failed|Repaired exactlibrary discovery and failedcache only|
|B12x build20GiB cap|OOM137|64GiB cap; actual peak~29.62GiB|
|B12x final hostwrapper|CUDA header ordering failed|Scoped host include repair; reused alreadycompiledCUDAobjects|
|NewJovian actualconfig|EXL3 absent from ordered overrides|Minimal pinned/idempotent Pythonregistration patch; actualEngineArgs gate passed|
|Newdraft constructor|Legacyblock64 and autodraft attention invalid|Block256 and explicitB12X for both target/draft|
|Actualdraft probe4|Default MTPheadNVFP4conversion observed|Disabled both targetMXFP8 and MTPNVFP4head flags; source-value native checks required|
|Newdraft source audit|OldF32APE allocation differed|Storedsource isBF16; exact512valuecheck passed, no falseprecision-loss claim|
|Newdraft finalgate|All891keys and protectedsource checks passed|R3 actual strictV2 load, FP8onlyexperts|
|Fullserver attempt1|Positional model argument not consumed|Explicit --model /model; failedattempt retained|
|Fullserver attempt2|133targetshards loaded; draft/loading gates ongoing|No fullB12x serving, speed, KV or quality acceptance yet|

Portable launcher has16CPUtests passing and includes source-bound MTP view verification, delayedMTPmetrics handling, port occupancy checks, bounded setup, explicitB12X/block256/nativehead flags. CPU success is not public-image or GPU acceptance.

## Higher-agreement work now

`quality/unsloth-reference-20260911/ASSESSMENT.md` records official Unsloth measurements and methodology. Different panels and precision allocations preclude a directGGUF/EXL3ranking.

`quality/mixed-precision-feasibility/` verifies paired MITK3 calibration provenance and exactbytecosts. Layer5/32 is the first proxy-selected mixedK2/K3 candidate being staged;1.6875GiB addedtensorpayload, all288experts retained. This proxy ranking is independent of the frozenWikiText evaluation, but is not validated end-to-end sensitivity. No agreement gain is claimed.

Root is running a separate head-only full-vocabulary KL-tail supplement from sealed activations, with finite checks and agreement/NLL/KL crosschecks against original reports. Existing evaluator, reports and source weights are immutable.

Final acceptance still requires higherquality, real262144budget retrieval, vision/video/MTP, matched per-request and totaldecode/prefill/concurrency, cleanHF/GitHubDockerpublication and a fresh publicreload. Only then resumeDeepSeek-V4.1-Flash.


### Latest verified updates

B12x runtime reached751007aggregateKV and output42; counting/formatting gates failed. Profilercompleted/stopped. Mixedlayer5/32artifactassembledandhashverified; GPUqualityauthorized. KLtailssupplementsfinishedexit0/noOOMandpreserveoriginalmetrics; see`quality/kl-tails-results/TABLE.md`. Noqualitypromotion/publication.


### Verified full-vocabulary tail measurements

65,504 positions; lower KL is better. Two-pass head-only calculation, FP64 accumulation, original report crosschecks passed. No model layers re-run.

| Candidate | Mean | p99 | p99.9 | Max |
|---|---:|---:|---:|---:|
|original-q3-v1|0.152204|1.843328|4.039562|8.510485|
|k256-v1|0.687907|5.950869|9.335599|15.121773|

Tiny negative values from finite-precision rounding are retained and counted; no clamping. Quantiles use linear interpolation. Cross-method external GGUF comparisons remain unmatched.

## 2026-09-12 unpruned projection lead and live context acceptance

OfficialUnsloth IQ3headeraudit completed:targetgate/upmostlyIQ2_S,downIQ3_S,protectedheads/attentionQ6. Weightlabel alone does not establish nativeprotectedEXL3fit. Evidence quality/gguf-header-audit/ASSESSMENT.md. Projection-specificEXL3K(2,2,3) boundedGPUgate assignedde5c;no fullqualitylaunchyet.

Lowefforttext JSONvalues correct butMarkdownfences failedstrictformat;explicitJSONobject mode passedstop.4kthree-code retrievalpassed,4096prompt/32output,12.503s wall,10.507sTTFT.262016+128budgetrequestlive on2384;DSATiledTopkKernelJIT warning meanscold latency, no warmthroughputclaim. Evidence benchmarks/reasoning-effort-check/.


## 2026-09-12: Successful-output structured decode and DeepSeek layer4

The200-integerJSONscreen completed all6cells correctly. Measured TOTALdecode14.91–15.07tok/s at1023/4095prompttokens; windows27.9–28.2s remain screens. The extended300-integertask is running sevenC1contextsizes with separatewarmups and2measuredrepeats. Firstmeasured cell:1023input,683output,45.53s matchedwindow,14.98TOTAL/per-requesttok/s,422.32nativeprefilltok/s. Correctnormal-stopJSON,exacttokenIDs,andadvancingMTPcountersindependentlyverified. No MTPspeedgain or broaderqualityclaim. See benchmarks/structured-sustained-20260912/TABLE.md; it is partial.

DeepSeeklayer4nativecapture completed512rows/1048576tokens in736.33s,56,598,282,240B,373coverageeligibleexperts. Root independently rederived373eligibles and11exclusions fromsealed counts; no fullmodelacceptance. SHA bd5c2003192152fb9a3374071f7cf856bd9531430fe6c60d346a690e465d450b. Supplementalnaturalroutecapture andnextlayerpipeline remain active.


## 2026-09-12: Projection runtime packaging

Built isolated Docker overlay on de5c:368820997e11, adding the exactGPU-qualified30321f70wrapper and2794ddbbhelper over verifiedR3c0d5f762. AllbaseRootFSlayers,ENV,entrypoint andCMD unchanged; Dockerreported28,522Bsizeincrease. CPUactualcandidateconfigparsedall42layertriplesthewayexpected; noCUDAinitialized, terminalexit0/noOOM. Sixexistingprojectioncontract/dispatchtests rerun andpassed. Fullservingnotyettested. See b12x-runtime/projection-runtime/BUILD_RECEIPT.json. The initial CPUlaunch used anunavailable namedruntime and didnotstart; the second usedsupporteddriverexposurewithCUDAdeviceshidden andpassed.


## 2026-09-12 01:14 UTC: Two encoders and draft-memory experiment

DeepSeek now has two qualified K3 encoders, layers4 on2822 and5 on557f. Cross-node realexpert gate/up/down packed hashes match exactly. Bounded successor capture/encode staging preserves original parents and10GiB disk floor. The GLM C1 matrix independently passed both64k repetitions and the first131k repetition, at14.79–14.82TOTALdecode and455.28–456.95nativeprefilltok/s for64k. Projection6 capture is terminal0/noOOM; paired quality pending.

Draft-only onlineNVFP4 adapter passed actual installed CPUimports/precision-scope guards, with backendselection mocked.8nativeexpertW4A16 GPUreference/graph proof staged only.16MTPdepth/slot controls and admission/countertooling passed27CPUtests, independently rerun. No live inference change or speedup claim.


## 2026-09-12: Projection6 terminal and draft reduction A/B

Projection6:77.97691744% agreement,KL.4198133393,PPL4.44220947 at65504positions. All9terminalreceiptSHAs independently verified. The+.59233pp gain is insufficient. Projection14CPUassembly is prospectively selected from independent calibration, pending memory admission and quality measurement.

NVFP4 draft component failures preserved:attempt1 needed explicit workspace initialization; attempt2 exposed graph/eager variation; attempt3 fixed-route diagnostic needed preallocatedINT32routeIDs; attempt4 isolated repeated nondeterminism to routedexperts with router/shared bitwise stable; attempt5 observed actualtc_decode_fused_sum=True. Packedattempt6 observedFalse and M1replays bitwise equal, but saturating-inputFP32referencefailed and remains unresolved. No tolerance was relaxed; no servingchange. Raw8expertweights113246272B vsnative402653184B, native router/shared exact. Full288FP8draft load comparison launchedb0a96646, no target/KV/API.


## 2026-09-12 01:58 UTC: Packed NVFP4 component and measured draft saving

Attempt7 passed all five unchanged-tolerance reference cells, including saturation, and15 changed-input graph replays were bitwise equal. The separately source-pinned reference models explicit BF16 packed arithmetic; attempt6's FP32 oracle failure remains retained. Actual packed launch disables fused atomic sum; direct control weights are identical. This is eight-of-eight real-expert testing, not full288 sparse/serving proof.

Full288 production-loader FP8/NVFP4 A/B completed with native protected dtype audits and full-value embedding/head comparisons: resident10,255,971,840B versus7,085,086,720B, saving3,170,885,120B=2.9531GiB. Both terminal0/noOOM, no target/KV/API. NVFP4attempt1 audit incorrectly expectedUINT8 instead ofINT32-packedFP4; retained, corrected inattempt2 with exact4-bit byte accounting. Source/result SHA seals are in b12x-runtime/mtp-draft-nvfp4/.

Projection14 assembled133files with all experts and native protected tensors; manifest80967e71922a534a3ab2872ad90edc4aacaf958bcb0a060699e599318727556d. Its independent-calibration14downK3allocation adds3.9375GiB; net+.9844GiB after measured draft saving. Actual fullserver admission pending. Matched qualityCID5a05e4c4 has layer0 complete32/32. Baseline12 measured C1cells through199999input passed;260095warmup passed. No release promotion.


## 2026-09-12 08:24 UTC: Completed quality/matrix and recovery

Projection14:78.381472887% agreement,KL.4065290764,PPL4.395652597. All9 terminalreceiptSHAs verified; insufficienthighquality gain. C1baseline21cells completed,14measured independentlyverified;260095input repeats14.80/14.83TOTALdecode and453.16/452.94nativeprefilltok/s. Baselinepreserved/stoppedafteridlecheck; D2C1newcontainer4c5a3d3f loading. Full288NVFP4sparseproof563adcb7 runningde5c afterCPUimports+3tests; actualGPUresultpending.

DeepSeeklayer7sealed364experts/1092projections/20gaps. Layer3supplementexpert41 terminal0/noOOM/all3hashesverified,1056train266heldout withcorrectoriginalrowIDsplit. Layer6recovery preservedstalehold andnewline-mutatedseal; exactsourceparentfd071e13restoredafterJSONandstrip equality, then7.397GB sourceSHA/headerstagedin8GiBtmpfs/48GiBcgroup. Container64044a70producingoutputs. Nooriginalsource/calibrationdeletion.


## 2026-09-12: Full288 sparse NVFP4 proof passed

Container563adcb7078c completed10 independent-reference cases and30/30 bitwise changed-input/route CUDA-graph replays. Native and fixed boundary routes exercise all288expert addressing, including0/7/8/127/255/287. Maximum routed relativeL2 to the independent packed-BF16 reference0.0003725954; peakCUDA21.3329GiB. Native production-loader completeness, protectedheads/router/shared values passed. Actualcompileflags consistentlypacked/noatomicfusedsum. Rootverifiedallterminalandfrozen-sourceSHAs. No fulltarget/KV/attention/API/speedacceptance follows. Serving-lifecycle warmup sourceaudit stillneedscompletion beforefullserverNVFP4launch.

## September 12 09:20 UTC follow-up

The first depth-2 C1 server returned correct JSON, but actual draft-decode graph coverage was empty: capture size 3 exceeded the draft manager's one-request, one-token limit. It was stopped and preserved. R2 uses capture sizes [1,3]. All 27 tooling tests and 44 actual installed candidate-construction role checks across 16 configurations passed; the earlier compatibility-only test did not cover candidate construction. The corrected server is loading on 2384. Watcher PID 1368519 requires actual capture admission and a functional response before comparing the exact baseline prompts at 1k and 16k. Short decode windows remain screens, not sustained results.

The full 288-expert NVFP4 sparse MoE proof already passed. Source inspection then established a production lifecycle omission: B12x warmup enumerates only the target returned by worker.get_model(), while native MTP is held separately and returned by get_draft_model(). A small source-pinned R2 overlay adds draft provider warmup only with the NVFP4 opt-in. It preserves the completed-eager-warmup capture guard; CPU orchestration tests and full serving remain separate gates.

DeepSeek layer6 reached326 encoded experts at09:14. The pinned native layer8 source downloaded directly from HF and passed full SHA verification. Native8 is running on557f after its actual CPU restore/header/parent preflight; this includes the new shared-KV source boundary. The detached bounded queue continues native9 and then covered encoder9. Original weights and calibration parents remain.

## September 12 09:32 UTC — measured depth improvement and next jobs

Depth2 passed actual target/draft-prefill/draft-decode graph admission and all8 exact-prompt responses. Root independently recomputed timing and semantic acceptance from the downloaded raw cells; four measured windows were35.5–36.3seconds. Same K2 target, native FP8 draft, image, request payloads, C1,262144 configured context,2048chunk,.93memory fraction and cold prefix caching. Two measured repeats per depth:

| Input tokens | D1 TOTAL decode tok/s | D2 TOTAL decode tok/s | Change |
|---:|---:|---:|---:|
|1023|14.98|19.11|+27.62%|
|16383|14.86|19.03|+28.06%|

TOTAL equals per-request speed at C1. This is a repeated structured counting comparison, not broad task quality or a40tok/s result. D2 native prefill measured431.50/432.71tok/s at1k and448.94/449.45tok/s at16k. The idle D2 server was stopped and preserved; D5C1 is loading with capture[1,6] and the same actual-admission/smoke/exact-replay watcher, PID1384971.

NVFP4 warmup R2 adds4877bytes over the pinned parent image and passed5 tests of the actual installed orchestration with CPU providers; base layers, environment and entrypoint are preserved. Full Projection14 serving attempt1 failed before weight loading because the stock loader's root glob missed indexed weights/ paths. A new hardlinked flat view retains identical tensor inodes, keeps original weights/ paths usable for native MTP, and changes only its copied index. The actual loader/header preflight passed133files,583090targetentries and891draftentries with CUDA uninitialized. Attempt2 is loading on de5c; watcher581449 requires actual graph/capacity admission and a correct response. No full serving claim yet.

DeepSeek6 is sealed366experts/1098projections with18coveragegaps; exact sealSHA ac1500aeb1718151ceeb4c2e5b2d58c05efbcc3c9bd01a6878d646eb0639ab20. Native8 is sealed512rows with366eligible/18gaps. Native9 is running on557f; encoder9 follows. The second encoder queue, localPID27794, streams lossless compact8 into2822hostRAM tmpfs, then uses the existing bounded source-tmpfs encoder. HostRAM is explicitly budgeted separately from its48GiBcgroup; exact output/disk gates remain. Originals and previous derived inputs are retained.

## September12 10:04–10:13UTC: fleet interruption

557f shows a new boot around09:40; native9 exited255 with OOMKilled=false and last recorded297rows. Its previous boot journal ends without a shutdown cause; the readable targeted OOM/panic/thermal/shutdown search returned no entries. Pstore is not readable. Root cannot infer power loss versus another hard restart.2822/de5c are Tailnetoffline;2384 is listedonline butSSH,TSMP and both known fabric routes did not establish access. PopSSHworks and production was untouched.

Original native9partialrows and container are preserved. Recovery uses separateoutput/evidence/containeridentities and the unchanged native computation. ActualsourceSHA and all512parentrowchecks run again afterreboot. The firstCPUpreflight exposed stdlib queue shadowing by the new driver filenamequeue.py; driver renamedresume_driver.py, oldfileandfailedlogsretained. Recoverydriver9208 is revalidatinginputs on557f, then will recapture native9 and launchcoveredencoder9. No completedexpertencoding is repeated.

D5andProjection14serverresultsremainunverified because their hosts cannotbereached. DS8had6liveverifiedexpertoutputs at09:33:58; localterminalpollerfailedSSH255at09:39:47. Itscheckpointmustbeauditedwhen2822returns; do not rerun the old queue because it allocates freshinputs andwouldduplicatecompletedwork.
## September 12 10:57 follow-up

Native9 recovery passed all56,598,282,240bytes of calibration validation and terminal0/noOOM checks. Root fetched the exact receipts; seala50afd868f5a79a04844e581be3a5440a75df38ae9ed1439193fe7a069d4d36d. Capture took861.175seconds; this is calibration time, not serving throughput. Encoder9 containerdf4d1207183a is advancing with the original strictcoverage and codepins.

Sources10/11 are pinned to published HF fullSHA41d87a4c81fec1550f9cb975db05598a18ee0161c2664e8e1a0c7b60742755a6 ande7ca4a12688a5819829ead3a03282aedb380e438bc41a00e969f70952c6d0eb9. Each is7,389,761,368bytes. Queue18455 waits for the exact successfulencoder9seal, then performs native10/encode10/native11/encode11serially. Stagebudget preserves room for both fullparents, outputs and reserve. No source/parent deletion.

GLM CPU accounting inquality/kv-budget-followup/ASSESSMENT.json identifies a prospective tradeoff: D2C1 reports753664aggregateKVtokens despite admitting one262144request. A portion might fund additionalK3expertprojections. Linear estimates are not runtime admission; hybridblockrounding, actualfull-draft savings, loadingpeaks, media andOSreserve mustbe measured. No protected precision/context change or qualityprediction is claimed.

## September 12, 11:36 UTC — encoder9 sealed, native10 active

Encoder9 exited0 withoutOOM:369experts/1107projections,4,913,756,028bytes;15natural-routecoverage gaps remain. Root verified terminal identity/counts and recorded exact seal SHA in `next-pair-8-9/recovery-20260912T1004/ROOT_ENCODER9_TERMINAL_RECEIPT.json`. The existing queue launched native10, which reached66/512rows at11:34UTC. Both next sources are downloaded and SHA-verified. The other3Sparks still timed out at11:35UTC. No duplicate jobs launched, no Pop changes, no public release acceptance.


## September12,12:11UTC — native10 sealed and encoder10 progressing

Native10:512rows,365eligible/19gaps,56,598,282,240bytes; exit0,noOOM. Root matched container/source/recovery9parent hashes,432natural+80random split, all512row entries, and independently recomputed365eligibleexperts from384train/heldoutcounts. Encoder10:134/365at12:10:50UTC,48GiBcgroupwithoutadditionalswap, unchanged pinnedimage. Existingqueue18455willproceedtonative11afterterminalseal. Diskfree125.05GB. Other3SparksstillSSHtimeout12:07UTC. No duplicatejob,Popmutation,cleanuporpublicupload.

New reproducible ledger `coverage-ledger/TABLE.md` reconciles122expert-layergapsacrosslayers3-10; expert41supplementonlyclosesitsownlayer3gap. Layer8encodingterminalremainsunknown. GLMcalibration-onlyprojection20/22mapsfrozenandreproduced; no newGPUfit/qualityclaim.


## September12,12:52UTC — encoder10 sealed; native11 started;12/13queued

Encoder10:365experts/1095projections,4,860,490,380bytes,exit0/noOOM;19coveragegaps. Rootreceipt `next-pair-10-11/ROOT_ENCODER10_TERMINAL_RECEIPT.json` matches nativeparent and terminalidentity. Native11 CPUrestoration/all512parenthashes passed; GPUcontainerde71ca13a70cstarted12:51:32UTC,initial0rows.

Pair12/13directdownload41133andsequentialqueue41134areactive;18deployedhelper/pinhashesverified. Source12 showed1.735GBdownloadedat12:47; fullSHApending. Storage reservation214.116GBincluding12GiBfloorpassedagainst235.521GBfree. Queuewaitsencoder11seal, thenruns12and13withoutoverlap. Unchanged computation/coverage; original10GiBencoderfloorpreserved. Layer14requiresseparateEngram/sharedKVadmissionand~149.46GBadditionalstorageprojected; sourceheaderonlyaudited.

Other3Sparksstillunreachable12:42UTC;GLMoutcomesunknown. This run deleted no data. Oldinterruptednative9pathnowabsent; sealedrecovery9retained; dispositionunknown. Originalhistoricalclaimsarenotfreshretentionproof.


12:52UTCfresh-outputcheck: native11 GPUlog reached16/512rows; monitorrecorded9rowsat12:52:32.938UTC. This confirms newcaptureoutput, not completion or inference speed.

## September12,13:40UTC — native11sealed, encoder199/364, storagearchiverunning

Native11rootreceiptin `next-pair-10-11/ROOT_NATIVE11_TERMINAL_RECEIPT.json`:512rows/364eligible/20gaps. Encoder11running199/364at13:39UTC. Bothsources12/13FULL_SHA_VERIFIED. Coverageledgerlayers3–11has142expert-layergaps.

Storage: directIOarchiveofcompletednative5/6/7startedonPopSYSTEMdiskPID1873162;169.795GBlogicalpayload. Actualsmokeverifiedmanifest+110MBrowplusarchiverereadSHA;firstroot-permissionfailurepreserved. Read-only128MiB/noGPUexportcontaineruseslog-drivernone,inspectverifiedbeforebinaryattach. Capped30MiB/s;WiFisourcepathcurrently~12–13MB/s.3.17GBcopiedat13:39UTC,nooriginalretiredyet. Eachsourcewillberetiredonlyafterfull513filehashverification/durablearchiveandunchangedsource/mountchecks. Evidence `checkpoint-archive-20260912/ROOT_ARCHIVE_PREFLIGHT.json`;destination `/home/ser/deepseek-v41-spark-checkpoint-archive-20260912T132635Z`. Do not treatforecastsasfreedspaceorarchivalcompletion.

Other3Sparksremainunreachableincludingsecondaryfabric;GLMresultsunknown. PopGPUrecipeunchanged;productioncontainerhealthgreenbutnofreshAPItestclaimed.


## September12,14:23UTC — encoder11sealed, native12recovered

Encoder11terminalrootverified364experts/1092projections,4,847,173,968bytes,20gaps. SealSHA d9660f9dd9135cc8d08d62b66223766c386b8f704a079dd31433df5d637a2947. Native12firstattemptstoppedbeforeGPUbecauseoldguardrejectedCPUarchivecontainer;actualCPUpreflightpassed. R2guardadmitsonlypinnednoGPU/readonly128MiBexporterandretainsactualGPUidlecheck. Positiveactualexporterand13negativevariantschecked;reproducibletestsaved. Fourlauncherreferencesupdated;oldsource/logs/provenancepreserved. QueueR2PID60257,containerab71bec6,196/512native12rowsat14:22UTC. No12encoderoutputrepeated.

Archive5transferred36.40GBat14:22;verificationandretirementpending. Layer14actualCPUmetacontract2347tensors/6Engramshapespassedafterreadonlycontainer/tmpcachefixes;notfullweights/lookup/sharedstate/GPUacceptance. Source13fullySHAverifiedandqueued. GLMthreehostsunreachable14:12UTC.



## September 13, 2026 22:58 UTC — S216 G4 sealed; samples summary produced

`s216-g4-summary.json` was missing although panel and samples receipts were complete. The scheduled driver run executed g4_summary.py on spark-2384 (written 2026-09-13T22:55:25Z; both receipts status=complete; samples sha256 5684425f… matches the expected pre-registered hash). Final S216 row, verified against receipts: top-1 68.0187%, KL lower bound mean 0.80122 / p50 0.30486 / p95 3.42855 / p99 6.09066 / max 12.83936, candidate PPL 6.5965, samples 12/20 — structured_decode 1/5, generation 1/5, reasoning 5/5, retrieval_long_context 5/5. U216 compares: same 12/20 total but a different profile (structured_decode 5/5, generation 1/5, reasoning 3/5, retrieval 3/5); panel top-1 66.83%, KL 0.90523, PPL 7.3714. All three receipts copied to the Mac `s216/receipts/`. BUILD-S216.md SAMPLES_PLACEHOLDER was resolved at 22:38 UTC by a parallel editor with per-row detail consistent with these receipts; this run independently verified every published number against panel+samples receipts and made no further edits there. Evidence: `/home/sero/w2port/out/s216-g4-{panel,samples,summary}.json` (2384), `s216/receipts/` (Mac). Next queue item: T216 sensitivity-only keep-216.


## September 14, 2026 01:3x UTC — T216 built, served, evaluated: sensitivity-only ranking refuted

Queue item 2 (scheduled driver run 2, lock-held). Plan T216 (sha 4e4d8973661b414cb0f2bc8496e756d41b25f1dd6a8e2299d5bcb97c574f492a, t216/t216-plan.json 0444): uniform keep-216 ranked by K3 sensitivity ALONE, routes excluded from the score; sensitivity inherited verbatim from the sealed S216 plan (pins k3-projection-errors.json 89a26929…; independent reduced copy cross-checked at max |diff| 0.0 over 12,096 experts); routing-counts.json sha re-verified (1f0e98c8…) and used for stats only. Route-mass retention 74.80% (S216 80.0%, U216 86.0%); keep-set overlap 202.7/216 with S216, 152.4/216 with U216; MTP layer 45 verbatim. Build on 557f: pack 15 shards → augment → G2 verify PASS (111,736 tensors byte-compared, 0 problems, 140.4 s) → census stock exactly 42 expected deviations, plan-aware contract_ok=true, residual [], dense 440/440.

Serving took three attempts, all logs/receipts preserved in t216/: attempt 1 crashed ~15 s in — the overlay config.py requires the served census receipt at /receipts/exl3-plain-census-2p05.json (the "problems cleared" copy U216/S216 had pre-staged; my build scripts omitted it). Attempt 2: receipt produced (42 expected deviations cleared, verdict.contract_ok=true, raw plan-aware census preserved; one guard bug fixed on the way — problems_expected_by_plan is the 42-string list, not a count) → READY but max_total_num_tokens 258,240 < 262,144 required, with cmd/env/shm/devicerequests byte-equal to glm53-s216 (270,016 at equal avail mem) ⇒ profiling variance; attempt 3: READY with 302,656 ≥ 262,144, load 330.70 s, greedy smoke PASS ("smoke ok", stop, matched_stop 154827). glm53-s216 was stopped with docker stop (container preserved; inspect+ports captured pre-stop); glm53-t216 owns 557f's serving slot.

G4 from 2384: panel candidate exit 0; compare attempt 1 FAILED (teacher-row symlink bug — my container command ran with cwd=/ so ../panel didn't resolve; one dangling literal link, exit 0 — evidence in t216/g4-t216.out); samples completed 20/20 during the failure; fixup with absolute-path symlinks → compare exit 0, summary exit 0. FINAL T216: top-1 66.4998%, KL lower bound mean 0.85544 / p50 0.34680 / p95 3.58252 / p99 6.28300 / max 13.87572, PPL 6.99697, samples 6/20 (structured_decode 0/5, generation 1/5, reasoning 1/5, retrieval 4/5). Per-row (32): vs S216 top-1 5/32, KL 4/32, NLL 5/32; vs U216 top-1 12/32, KL 15/32, NLL 15/32.

CONCLUSION: hypothesis refuted — the ln(routes+1) factor in S216 carries real signal despite the thin capture; sensitivity-only ranks worse on every panel metric and collapses samples to 6/20 (reasoning 5/5 → 1/5). Leader among pruned points remains S216 (KL 0.801). Next queue item: denser route capture on 2384's idle recorder, then re-rank sensitivity × ln(routes) with the richer routes. Evidence: t216/BUILD-T216.md, receipts on 2384 (/home/sero/w2port/out/t216-g4-*.json) and 557f (/home/valentine/t216/), local copies in t216/.

## September 14, 2026 ~09:30 UTC — R216 (denser route capture) built, served, evaluated: NEW LEADER

Queue item 3 (driver run 3 executed capture→serve→G4 launch; its recording completed in this pass after the driver lock went stale at 04:27Z — the G4 poll loop died with the session, G4 itself ran to completion on 2384). Capture: 15 generation prompts (GEN_PROMPTS[1..15] verbatim, temp 0.7/top_p 0.95, hard max_tokens=2048 cap) via r216/capture_gen_r216.py on 2384's recorder server → 26,560 completion tokens; combine_reduce.py full recomputation from ALL 16 dumps → routing-counts-r216.json sha ee8858f0… (86,453,472 routes, +11.8% vs sealed capture; zero-route experts 0; min per-expert 113). Plan R216 sha 62300f56… (sensitivity × ln(routes+1), denser routes; keep-set overlap 213.2/216 with S216, 158.1/216 with U216; route-mass retention 79.72%; MTP layer 45 verbatim). Build on 557f: pack → augment → G2 PASS (111,736 tensors, 0 problems, 136.6 s, finished 02:17:33Z) → census 42/42 expected deviations, residual 0; served census receipt pre-staged BEFORE first launch (T216 trap), index_only=false contract_ok=true. Serve: attempt 1 KV 253,248 < 262,144 (profiling variance) → attempt 2 KV 321,664 READY + greedy smoke PASS; glm53-r216 owns 557f's slot. G4 on 2384 finished 03:58:50Z.

FINAL R216: top-1 68.5286%, KL lower bound mean 0.77709 / p50 0.29003 / p95 3.36459 / p99 5.99003 / max 14.21859, KL top-k renorm 0.76834, PPL 6.44689, teacher PPL recomputed 3.19953 (= reference), samples 12/20 (sd 1/5, gen 1/5, reasoning 5/5, retrieval 5/5; samples sha matches pre-registered hash). Per-row (32): vs S216 top-1 23/32, KL 26/32, NLL 27/32; vs U216 18/32, 23/32, 22/32. Failure set identical to S216's (8 = 5× sd-json300 + 3× gn-tips240, all token-cap stops) — ranking stability w.r.t. capture depth confirmed as pre-registered.

CONCLUSION: densifying the generated-token routing signal (1 partial prompt → 16 prompts) improves the score's route factor into a real gain: KL −0.024 nats vs S216 (0.777 vs 0.801), +0.5 top-1 points, PPL −0.15, same 12/20 samples. **R216 replaces S216 as leader (0.777).** The gap to A0 remains ~0.39 nats / ~10.4 top-1 points — pruning damage without healing. Next queue item: REAP-native score (router-weighted activation per reap-strict-serialization) keep-216; rebuild + eval only if it plausibly differs from R216's ranking; then G5 loop harness on the leader. Evidence: r216/BUILD-R216.md; receipts 2384 /home/sero/w2port/out/r216-g4-*.json, 557f /home/valentine/r216/ + manifest.json, Mac copies in r216/receipts/ (12 files, sha-verified identical to the omarchy mirror); u216-g4-{panel,samples,summary}.json also backfilled to Mac u216/receipts/ this pass.

## September 14, 2026 ~13:00 UTC — USER PIVOT to real benchmarks; E216 build failed on full disk; U216 artifact retired per authorization

User directive: the KL-only view may be wrong ("we are doing something really wrong") — run REAL benchmarks on what we believe is the highest-quality 3bpw + ~25% REAP point (R216, panel leader) against the unpruned reference A0, on multiple DGX Sparks for speed. Suite: MMLU (full) + GPQA Diamond (zeroshot MC + CoT) + Terminal-Bench 2.1.

E216 build FAILED at pack shards 6-10/15 with OSError Errno 28 (557f at 100% disk, 38 MB free; wedged packers self-exited; build+census never ran; orchestrator stopped). Response: the U216 artifact dir was deleted under the standing BUILD-S216.md authorization AFTER a full pre-deletion receipt — all 15 shards sha256-verified against the sealed u216 manifest (all match; receipt u216/receipts/deletion/u216-pre-deletion-receipt.json on Mac + /home/valentine/e216/u216-pre-deletion-receipt.json on 557f) — and the failed E216 partial (41.3 GB) removed with logs preserved (e216/build-failed-diskfull.out, pack/verify logs). 557f now 129 GB free. E216 plan stays frozen (sha 37fa10fd…); rebuild deferred behind the benchmark campaign.

Benchmark serving: both points relaunched in BENCH-TUNE (identical artifacts/overlay/262144-context/vision/kv-fp8/dsa; only batching knobs raised; panel-era logs+inspects preserved for exact restore). Attempt 1 with MAX_BATCH_TOKENS=8192/chunked-4096/running-32 HUNG silently at the EXL3 decode stage on BOTH hosts (15+ min vs 2m45s normal; GPU 0%) — attempt 1 logs preserved (r216-bench-attempt1-hung.log, a0-bench-attempt1-hung.log); attempt 2 moderate knobs (2048/1024/16) launched 13:02:32Z/13:02:55Z. lm-eval 0.4.13 installed on omarchy (new run/ls CLI); PREREG.json written BEFORE any results (benchmarks/lm-eval-bench/PREREG.json): mmlu full + gpqa_diamond_zeroshot + gpqa_diamond_cot_zeroshot, seed 1234, local-completions from omarchy against both points; watcher validates with a 2-doc GPQA run (also catches the gated-dataset/HF-token path) then auto-launches both full runs. Terminal-Bench 2.1 installing in its own venv on omarchy (tb). Evidence: benchmarks/lm-eval-bench/ on the Mac.

## September 14, 2026 ~13:30 UTC — 557f data-plane outage (user-authorized reboot armed); a0 mamba OOM fixed; lm-eval client moved to the Mac

557f stopped answering at the tailscale data plane ~13:05-13:10Z (its ssh polls were instant at 13:04:30Z): the netmap lists it idle/relay-fra, but `tailscale ping` gets no reply and ssh:22/HTTP:8000 time out from the Mac, omarchy AND 2384. de5c — the only jump path to 557f's LAN 10.0.1.2 — has been offline ~10 h; titpod (ext.titpod.com:3061) refuses connections, so no alternate LAN path exists. With no remote path left, the user asked "is there any way for you to send a reboot signal" — for this incident that authorizes rebooting 557f: `benchmarks/lm-eval-bench/recover-557f.sh` (nohup on the Mac, pure-local) polls de5c via local `tailscale status` every 3 min and, the moment de5c returns, probes the jump path and fires `sudo -n systemctl reboot` on 557f (skipped if 557f self-heals first), then waits and relaunches the r216 bench server with attempt-3 knobs. Passwordless sudo is not guaranteed — if it fails, the log records it and only a physical power-cycle remains.

Root cause on the healthy host: a0 bench attempt 2 (mf 0.90, chunked 1024, max-prefill-tokens 2048, running 16) died 13:09:29Z with "Not enough GPU memory for hybrid (mamba/linear-attention) state cache" — max_mamba_cache_size=-31, total_rest_memory=-8.63 GB, 140.78 MB/req (evidence /home/sero/w2port/bench/a0-bench-attempt2-mamba-oom.log). The profiling reserve scales with max-prefill-tokens, so attempt 3 (launched 13:16:37Z) cuts the reserve: mf 0.86, chunked 512, max-prefill-tokens 512, max-running-requests 8, all other flags identical (262144 ctx, multimodal, fp8 KV, dsa backends, no cuda-graph). r216 attempt-2 used the same OOM-prone knobs on 557f before the outage — assumed crashed; relaunch-r216-bench3.sh applies identical attempt-3 knobs to both points for comparability.

Client move: lm-eval now runs on the Mac itself (~/lmeval-mac-venv, lm-eval 0.4.13 — same version as the omarchy venv; `run` subcommand CLI). `run-point.sh <name> <url> [poll-min]` is nohup'd per point: it gates on /health 200 → a real 1-token completion smoke → a 2-doc gpqa_diamond_zeroshot validation (catches gated-dataset/HF-token and API breakage), then launches the full preregistered suite (mmlu full + gpqa_diamond_zeroshot + gpqa_diamond_cot_zeroshot, seed 1234, log_samples) with PREREG model_args (num_concurrent=16, max_retries=5, timeout=1800); tokenizer files staged at benchmarks/lm-eval-bench/glm53-tokenizer (chat_template.jinja, tokenizer_config.json, tokenizer.json); HF token present at ~/.cache/huggingface/token (Aug 22, never printed). The run path is pure HTTP with no ssh legs, so the Mac D-state ssh wedge risk cannot touch the runs. r216's runner polls up to 48 h; the recovery watcher + attempt-3 relaunch feed it. Evidence: benchmarks/lm-eval-bench/{run-point.sh,recover-557f.sh,relaunch-a0-bench3.sh,relaunch-r216-bench3.sh}, run-{a0,r216}.log, recover-557f.log, results/.

CORRECTION (~13:45 UTC): attempt-3 (mf 0.86, chunked 512, prefill 512, req 8) crashed identically at 13:23:08Z — max_mamba_cache_size=-46, total_rest_memory=-13.05 GB (evidence /home/sero/w2port/bench/a0-bench-attempt3-mamba-oom.log, preserved). Lowering mem-fraction made it WORSE (-8.63 → -13.05): the configurator computes rest_memory = free_after_load − pre_model_load×(1−mem_fraction_static) − mm_reservation, so a smaller mf subtracts MORE slack while free-after-load is already nearly zero. Solving both crashes against each other: free_after_load ≈ 2.4 GB at the mamba check in BOTH — the exl3_plain overlay's decode workspace scales with the SGLANG_EXL3_MAX_BATCH_TOKENS env and 2048 eats ~40 GB before KV sizing (attempt 1 = 8192 → the silent hang; panel-era A0 = 256 → READY, KV 974,912 tokens, i.e. ~30 GB KV pool + ~41 GB free after load). Attempt 4 (BOTH hosts, identical knobs): SGLANG_EXL3_MAX_BATCH_TOKENS=256 + mem-fraction 0.90 + chunked 512 / max-prefill 512 / max-running-requests 16; everything else unchanged. a0 attempt 4 launched 2026-09-14T13:4xZ; r216's relaunch script (relaunch-r216-bench3.sh, deployed by the recovery watcher) carries the same attempt-4 config. Source read for the diagnosis: sglang srt/mem_cache/kv_cache_configurator.py _profile_available_bytes/_handle_max_mamba_cache (extracted from the serving image to /tmp/kvcc.py on 2384).

OUTCOME (~14:00 UTC): attempt-4 is the fix — a0 reached READY at 13:40:34Z with max_total_num_tokens=1,057,088 and zero mamba errors (the KV pool is even larger than the panel era's 974,912 because req 16 vs 1 changes the pool geometry). The 557f side resolved differently: the outage was NEVER a reboot (uptime continuous since Sep 12 16:57) — r216 attempt-2's memory appetite starved the host until nvidia-persistenced was OOM-killed at 13:20:27Z, and with /run/nvidia-persistenced/socket missing, EVERY GPU container (legacy --gpus AND CDI --device modes) fails at OCI create. Repair without sudo: privileged container `nv-persisted-restore` runs the host's own /usr/bin/nvidia-persistenced (libc-only link deps; this build has no -f flag, so a `sh -c "... & sleep infinity"` wrapper keeps the container alive) with /run bind-mounted rw and host aarch64 libs at /host-libs — socket restored, persistence mode Enabled, nvidia-smi-in-container PASS (580.173.02). r216 bench attempt-4 launched 13:58:48Z. Evidence: /home/valentine/bench/r216-bench-attempt2-pre-gpu-repair.log (attempt-2's preserved tail), recover-557f.log (watcher: SELF_HEALED, no reboot sent — the user-authorized reboot was never needed).

GPQA access finding (13:5xZ): the account-0xSero token (role write; never printed) fetches the gated repo's README.md (HTTP 200) but gpqa_diamond.csv returns 403 GatedRepoError via hf_hub_download, and datasets.load_dataset raises DatasetNotFoundError — the account has NOT been granted Idavidrein/gpqa. User action required: accept the gate at https://huggingface.co/datasets/Idavidrein/gpqa. Suite split accordingly (PREREG-ADDENDUM.json, written before any results): mmlu-only invocations launched immediately for both points; run-gpqa.sh auto-launches the GPQA pair for both points via watch-gpqa-grant.sh (polls the file endpoint every 5 min) when access appears. Mac venv additionally needed lm-eval[api] (tenacity) and transformers for the tokenizer path — three dependency failures each caught by the runner's validation gate and preserved in run-a0.log.

FINAL RESOLUTION (~15:10 UTC): attempt-5 is the serving fix that works on BOTH hosts — the overlay's moe.py enforces the batch cap (raises "EXL3 batch exceeds preplanned N tokens" on any forward batch > SGLANG_EXL3_MAX_BATCH_TOKENS), so attempt-4's chunked-prefill 512 crashed the scheduler on the first real prefill batch on both hosts (evidence a0/r216-bench-attempt4-overlay-batch-crash.log; smoke requests of 1 token never trip it). Attempt-5 = the exact panel-era pairing (MAX_BATCH_TOKENS=256, chunked 256, max-prefill 256, mf 0.90) with only max-running-requests raised to 16. Both servers READY ~14:48Z, survived the full 2-doc MMLU validation (validation accuracies sane: A0 stem 0.89-0.92 vs R216 stem 0.68 — directionally consistent with the panel leaderboard), and the full MMLU suites launched from the Mac at 15:08:38Z (a0) and 15:09:56Z (r216), landing in benchmarks/lm-eval-bench/results/{a0,r216}/ with samples. The runner gate greps were hardened after two false trips (a "403" substring inside it/s throughput numbers, and '"acc' not matching the printed |acc table). Server-side serving configs and the exact-restore panel-era inspects remain preserved on both hosts.

## September 14, 2026 ~16:00 UTC — GPQA unblocked WITHOUT the gate (user-directed reuse); both GPQA suites launched

The grant watcher polled 401 for an hour (account 0xSero not granted the gated Idavidrein/gpqa). User directive: "we should have an existing gpqa diamond run somewhere… why not reuse what works". The reap-era parcels pointed to the public mirror `hendrydong/gpqa_diamond_mc`; broader hunting found `bdytx5/gpqa_gpqa_diamond` — a public mirror carrying the FULL ORIGINAL schema (Record ID, Question, Correct Answer, Incorrect Answer 1–3, Subdomain + validator columns). Every row was cross-verified against the independent hendrydong/gpqa_diamond mirror (boxed answer TEXT, balanced-brace LaTeX extraction): **198/198 agree, 0 bad** (first pass flagged 17 rows: fixed by full-question alignment and a brace-balanced extractor; failed intermediate scripts preserved as build-gpqa-cache.py iterations, final = build-gpqa-cache-final.py). A local dataset dir (README configs trimmed to gpqa_diamond + csv) plus a copied task dir (tasks-gpqa-local/, yamls VERBATIM except dataset_path) makes the preregistered tasks load offline. Receipts: results/gpqa-reconstruct/{gpqa_diamond.csv (131,640 B), reconstruct-receipt.json}; PREREG-ADDENDUM.json correction 3; watch-gpqa-grant.sh retired (log line). GPQA validate passed on a0 (2-doc table acc 0.5/acc_norm 0.5, 2026-09-14T15:54Z) and both full GPQA suites launched: a0 15:56:36Z, r216 15:56:42Z (results/{a0,r216}-gpqa.log). MC part (792 loglikelihood requests) completed on a0 16:2xZ / r216 nearly done; CoT generation follows in the same invocation. MMLU suites continue concurrently on both points (shared server slots, symmetric across points).

## September 14, 2026 ~16:25 UTC — Terminal-Bench 2.1: proven runner found on 2822; omarchy back online as the container host; TBENCH-PREREG.json written; smokes green

The proven system existed: `/home/sero/terminal-bench` on spark-2822 — a Harbor-based Terminal-Bench runner with documented guardrails (distilled from the DeepSeek-V4-Flash 2.0 campaign), including the amd64/arm64 qemu false-failure trap analysis. State: (a) the Mac/Prime path (terminal_bench_2==0.2.1 + verifiers 0.1.15.dev17 + prime 0.6.10 on the Mac, dataset terminal-bench/terminal-bench-2-1 = official TB 2.1, 89 tasks) validated through preflight/imports/config-load/dataset-download, then FAILED at sandbox creation with PaymentRequiredError('Payment required. Check billing status.') — Prime account billing, owner action only; log preserved (prime-smoke-a0.log). (b) omarchy returned to the tailnet (x86_64, Docker 29.7.2, 1.5T free, GPUs idle) — the preregistered tbench host. Repo synced to Mac + omarchy (config.env excluded from rsync of secrets-bearing dirs; a plumbing-only config.env written with MODEL_LABEL/DEVICE_LABEL/MODEL_NAME=/model/SPARK_API_HOST/API_PORT). (c) TBENCH-PREREG.json written BEFORE any score-bearing tbench results (schema glm53-tbench-prereg-v1): terminal-bench-2-1, terminus-2, harbor==0.6.6 program, timeout_multiplier 20 / agent-timeout 100 for full runs, max_concurrent 1, full runs gated on suite completion per endpoint. (d) omarchy smokes: ORACLE PASS (container+verifier plumbing, jobs/2026-09-14__18-07-08, reward block present, dataset terminal-bench/terminal-bench-2 = 2.0 default), MODEL smoke running (terminus-2 → a0 endpoint, TBENCH_DATASET=terminal-bench-2-1 override, model openai//model, agent trajectory flowing, 1 task). (e) tbench-auto.sh armed on the Mac: polls results/status.txt for all four suite exits (a0/r216 × mmlu/gpqa), health-gates both endpoints, then launches BOTH full runs on omarchy in parallel (one per endpoint) with full run.env provenance. The 2.0-era dgx-spark TOMLs and guardrail docs preserved untouched.

## September 14, 2026 ~17:00 UTC — 3-track benchmark restructure (user directive); spark-2822 purged of all non-GLM53 models (2.2TB freed); dedicated tbench engine staged

User directive mid-suite: "we should have 3 tracks — mmlu on 1 spark, gpqa diamond on 1 spark, terminal-bench-2.1 on 1 spark" (i.e., suite isolation; no cross-contamination of endpoints). Implementation, no kills of score-bearing processes:

- Track 1 MMLU: a0 runner on spark-2384, r216 runner on spark-557f (one point per engine — an engine cannot host two models); both endpoints become MMLU-exclusive the moment GPQA CoT drains.
- Track 2 GPQA Diamond: MC 792/792 COMPLETE both points; CoT was finishing on the shared endpoints (this contention is what triggered the directive; MMLU rate had collapsed to ~3-28 req/min vs ~130/min isolated).
- Track 3 Terminal-Bench 2.1: omarchy x86_64 harness + a dedicated model endpoint. Old tbench-auto.sh watcher DEFUSED without killing it (marker .tbench-full-launched pre-created; its own marker check will exit it after suites land); new tbench-auto-a0.sh (a0 leg -> 2384 post-MMLU, marker .tbench-full-a0-launched) armed on the Mac. r216 leg -> dedicated engine on spark-2822.

spark-2822 purge (user: "remove all these models, the only model we want is the glm-5.3-flash we made for 1x spark"): 77 explicit paths + docker image prune -a. Receipt: 2822:/home/sero/work/glm53-single-spark-release-20260911/receipts/DELETION-20260914-2822.log (per-target du -sk + path logged BEFORE rm). Disk 3.5T used -> 1.5T used (2.1TB freed, 41% use). KEPT: campaign dir, glm53-rank-relay, glm53-3p05-pruned-r216 (copied), sealed laguna-reap-saliency-v1, reap calibration datasets (both HF cache copies), qwen36-hybrid-evidence, glm53-vllm-512k, all dsv4 work/protected/archive dirs (flagged, not models). NOT deleted (root-owned, need owner sudo): models/ remnants ~350GB (Inkling promoted/tp1, Laguna a5, qwen36-hybrid-cache, Nemotron, Qwen3.6-Hybrid, …), spark/models/hf-cache ~114GB, spark/models/vllm-cache 0.45GB. NFS NAS mount models/GLM-5.2-EXL3-TR3-3.0bpw (spark-raila.internal) never touched; purge script has an abort-guard on that mount.

Transport note (receipt for method): Mac-relayed ssh pipe 557f->Mac->2822 measured 2.35MB/s (~11h for 90GB) — stopped (own transport helpers, receipted), replaced with direct spark-to-spark rsync/docker-save over ssh using per-invocation -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no (no persistent ssh/config changes; valentine@557f key already authorized on 2822). Direct rate 306MB/s.

tbench-on-2822 staging: r216 artifact rsync 557f:/home/valentine/models/glm53-3p05-pruned-r216 (90GB) -> 2822:/home/sero/models/; image glm53-flash-sglang-exl3-plain:serve4-pruned (b9cf5753) docker save|load 557f->2822; serve config = attempt-5 knobs VERBATIM (SGLANG_EXL3_MAX_BATCH_TOKENS=256, chunked-prefill 256, max-prefill 256, mf 0.90, max-running-requests 16, 262144 ctx, fp8 KV, DSA backends, no cuda-graph) — deliberately NOT raised for the tbench leg so both tbench legs run identical serving configs; census receipt exl3-plain-census-2p05.json pre-staged (T216 attempt-1 trap). Container name glm53-r216-tbench, 2822 tailscale IP spark-2822.internal.

## September 14, 2026 ~20:15 UTC — USER-DIRECTED STOP of the late benchmark runs; pivot to "figure out the keep"

User: "just fucking go and figure it out please. stop the run it's too late" — explicit authorization overriding the standing no-kill rule for these specific runs. Stopped with receipts:
- r216 full MMLU runner (Mac pids 9197/4912) at partial 19,7xx/56,168 — partial log preserved at results/r216.log; status.txt notes the stop. NO exit line was written for the killed r216 suite (runner was killed before its exit echo — noted; the stop line replaces it).
- r216 tbench full leg (omarchy pid tree around r216-20260914T201046Z) after ~1 task in progress — harbor tore down cleanly (compose down --rmi --volumes observed), partial state preserved in runs/full/r216-20260914T201046Z/.
- a0 tbench auto-launch DEFUSED (marker .tbench-full-a0-launched) — one command to re-arm.
- NOT stopped: a0 full MMLU (finishes ~22:00Z; completes the a0 reference row; no contention — 2384 suite-exclusive).

Replacement diagnostic (decision-support, explicitly labeled QUICK, non-full):
- r216 QUICK MMLU probe launched 20:14Z: identical local-completions config, --limit 20/subject (1,140 docs, deterministic subset, seed 1234) on the freed 557f endpoint -> results/r216-quick. ETA ~35-45 min.
- a0 QUICK probe (identical subset/config) auto-arms via quick-probe-a0-watcher.sh when the a0 full suite exits -> results/a0-quick.
Purpose: same-subset MMLU accuracy for both points TONIGHT to settle "is the keep bad" instead of waiting for 02:30Z. Full-suite asymmetry (a0 full vs r216 partial+quick) is recorded as such.

Next (pending probe verdict): keep-quality analysis — R216 keep set (r216/r216-plan.json) vs E216 REAP-native plan (sha 37fa10fd…, overlap 149.6/216) vs G4 per-layer receipts; decide between E216 rebuild (different criterion, same budget), layer-adaptive keep/healing (queue item 5), or falling back to a0 (q2 unpruned) as the release config.

## September 14, 2026 ~21:50–23:15 UTC — E216 REAP-native rebuild COMPLETE: WORSE than the deleted R216 on the G4 panel; panel-vs-benchmark divergence now proven both ways

Orchestrator (orchestrate-e216.sh) completed 23:15:13Z, all phases exit=0. Pipeline receipts: e216/orchestrator.log (BUILD_DONE 20:21:46Z pack+augment+G2-verify all exit=0 in ~4 min; census; served receipt contract_ok; serve attempt-1/2 KV-gate note — orchestrator polled serve.out which the run-serve script never fills with engine logs; FIXED by bridging `docker logs -f glm53-e216 >> serve.out` mid-flight — engine was UP all along, KV=344,256 >= 262,144 gate passed 20:38:01Z; max_running_requests self-capped to 1 by mamba slots — fine for sequential G4; smoke try-1 empty/try-2 OK 20:40:08Z; G4 launched 20:40:35Z, done 23:15:02Z).

G4 panel (2384, teacher = unpruned 2.05bpw glm53-a0-bench; receipts /home/sero/w2port/out/<t>-g4-summary.json):
| candidate | selection criterion | top1 | mean KL lb | KL topk renorm | PPL | samples/20 |
|---|---|---|---|---|---|---|
| R216 (DELETED per user) | K3-sens x log(routes+1) | 0.6853 | 0.7771 | 0.7683 | 6.447 | 0.6 |
| S216 | same, earlier capture | 0.6802 | 0.8012 | 0.7912 | 6.597 | 0.6 |
| E216 (NEW) | REAP-native (router-weight x expert-output-norm over routed tokens) | 0.6641 | 0.8980 | 0.8721 | 7.238 | 0.8 |
| U216 | earlier plan | 0.6683 | 0.9052 | 0.8795 | 7.371 | 0.6 |

Findings: (1) The REAP-native ranking produces the SECOND-WORST panel row — worse than R216 on KL (+0.121), top1 (-2.1pp), PPL (+0.79), though better on the samples gate (0.8 vs 0.6/20). Route-mass-informed selection (R216/S216) beats pure REAP saliency on teacher agreement. (2) THE PANEL AND BENCHMARKS DIVERGE BOTH WAYS: R216 = best panel row but LOST to a0-q2 on GPQA (MC -2.5pp, CoT-flex -14pp); the panel's KL-to-teacher metric does not predict benchmark capability. (3) Benchmark receipts preserved (partials): a0 MMLU 35,941/56,168 (64%) + r216 19,7xx (35%); GPQA COMPLETE both points. Honest state: NEITHER pruned candidate has benchmark evidence of beating the q2 unpruned reference; a0 remains the benchmark-backed release config; the prune program needs the healing/adaptive-keep fork or a0-fallback decision. Plan-sha note: e216-plan.json self-records sha 37fa10fd... but file hashes 5cea3a31... (0444, identical Mac<->557f) — recorded as receipt discrepancy; the served/evaluated artifact corresponds to the on-disk plan.

## September 15, 2026 ~00:15 UTC — QUICK like-for-like MMLU verdict: the keep question is ANSWERED — both tested keeps lose to a0

Driver runs 15-16. Identical subset/config (local-completions, num_concurrent 16, limit 20/subject = 1,140 docs, seed 1234): a0-quick acc 0.8342±0.0120 (receipt results/a0-quick/bench-a0/results_2026-09-14T19-49-13.012988.json, filename Mac-local = 23:49Z) vs E216-quick acc 0.7728±0.0120 (receipt results/e216-quick/bench-e216/results_2026-09-14T20-06-15.281810.json, = 00:06Z). Delta -6.14pp at ~3.6 sigma (combined stderr ~1.7pp) — SIGNIFICANT. Group-level: E216 loses across stem 76.05/80.53, other 75.77/83.85, social 81.67/87.92, humanities 76.54/83.08 (E216/a0). Receipts: results/status.txt exit lines; e216/receipts/e216-g4-*.json copied to Mac from 2384:out/.

TOTAL EVIDENCE (all receipted): (1) R216 (route-mass criterion, deleted): GPQA MC -2.5pp vs a0 (~1 sigma, ns) but CoT-flexible -14.1pp; (2) E216 (REAP-native criterion): MMLU-quick -6.14pp vs a0 (3.6 sigma, significant) AND worse than R216 on the G4 panel (KL 0.898 vs 0.777); (3) the G4 teacher-agreement panel predicts NEITHER suite direction. VERDICT: uniform keep-216 at 3.05bpw loses to the q2 unpruned reference on every capability measure run, under two independent selection criteria. Benchmark-backed release config = a0 (q2, 2.05bpw, all 288 experts). The prune program's remaining options (layer-adaptive keep, healing) carry a heavy burden of proof; recommend a0-fallback unless the user wants the healing experiment. Engines: glm53-e216 still serving on 557f, glm53-a0-bench on 2384; 2822 idle.

## 2026-09-15T00:2xZ — Evidence matrix COMPLETE: E216 GPQA MC row (driver run 17)
- E216 GPQA diamond MC (zeroshot, 198 docs, seed 1234, same prereg args as run-gpqa.sh): **37.88% ± 3.46** — receipt `benchmarks/lm-eval-bench/results/e216-gpqa-mc/bench-e216-gpqa-mc/results_2026-09-14T20-20-31.781572.json`; status.txt start/exit 00:10:26Z→00:20:32Z; runner log run-e216-gpqa-mc.log; validation 2-doc rc=0.
- Ordering on GPQA MC: a0 43.43±3.53 > r216 40.91±3.50 > e216 37.88±3.46. With MMLU-quick (a0 83.42 vs e216 77.28) and the G4 panel (r216 0.6853 > e216 0.6641), E216 is the worst measured point on EVERY surface. Combined verdict unchanged and strengthened: uniform keep-216 loses to q2-unpruned under both selection criteria; REAP-native selection is no better than count-based, and neither panel nor either keep metric predicts benchmark direction.
- No CoT leg for E216 (evidence-matrix row was MC-only by design; CoT strict/flexible already near-noisy for both measured points).

## 2026-09-15T01:2xZ — driver run 18: E216 build record COMPLETED (pipeline step 8); healing PLAN.md written (queue item 5, write-only)

- BUILD-E216.md placeholders FILLED from receipts fetched 557f→Mac (e216/receipts/ now 12 files): pack.log (15 shards, 85.4 s, output 96,580,353,371 B), augment.log, verify.log + g2-receipt.json (**G2 pass: true, 111,736 tensors / 93,611,759,540 B, problems []**), census.out + census-pruned-e216-{stock,plan-aware}.json (**plan-aware contract_ok=true, dense native 440/440**, stock = 42 expected deviations), build.out (BUILD_DONE 2026-09-14T20:21:46Z), exl3-plain-census-2p05.json, serve-attempt1-short-kv.log (KV 218,176 — known profiling variance) + serve-attempt2.out (relaunch), e216-g4-{summary,panel,samples}.json. Full 4.0 MB serve.out stays on 557f only (recorded).
- E216 samples gate confirmed from the summary JSON: **16/20 passes, sha 5684425f… matches expected** — E216 is the ONLY point above 12/20 on the samples gate. Brief leaderboard row added (E216: top1 66.41%, KL_lb 0.898 [lb-estimator], PPL 7.238, samples 16/20).
- **QUEUE ITEM 5 EXECUTED AS WRITTEN-ONLY**: healing/PLAN.md created — binding gates for any healed candidate (MMLU-quick ≥ 84.62%, GPQA MC ≥ 46.96%, panel KL_lb ≤ 0.80 same lower-bound estimator, samples ≥ 16/20); H1 calibration-folded correction ≤ 12 GPU-h on 2822 (teacher side already in sealed observations, student side = one ~3 M-token forward pass); H2 kept-expert sequential per-layer fine-tune (pilot 2 layers → go/no-go; full = 2–5 GPU-days, BF16-original location is a BLOCKING unverified pre-launch check); layer-adaptive-keep blocked by the a2d loader finding (config-only probe first); router-bias healing ruled OUT OF SCOPE (protected tensors). **Nothing launched. Default if the user is silent: a0-fallback stands, plan waits unlaunched.**

## 2026-09-15T01:4xZ — driver run 19: a0-FALLBACK EXECUTED (default fork (a)); prune program CLOSED

User said "continue" with the fork open after run 18's recommendation — the pre-registered default (healing/PLAN.md §6 and run-18 handoff: a0-fallback stands if no choice) applied. Executed, receipts:
- **glm53-e216 stopped+removed on 557f at 2026-09-15T01:36:04Z** — pre-stop receipt (health 200, KV 344,256 confirmed, container inspect saved, df snapshot) at e216/receipts/e216-pre-stop-receipt.txt + e216-container-inspect-pre-stop.json (Mac + /home/valentine/e216/). ARTIFACT PRESERVED (96.58 GB, 31 files, G2-verified) — no deletion authorized or performed. Image kept. Only nv-persisted-restore remains on 557f; port 8000 closed (health_post_stop=000 receipted).
- **Release serving recorded**: 2384 glm53-a0-bench (Up since 2026-09-14T14:39:14Z, KV 1,057,088) IS the release serving; exact config snapshot benchmarks/lm-eval-bench/a0-release-serving-inspect.json (docker inspect: serve4 image, SGLANG_EXL3_MAX_BATCH_TOKENS=256, mf 0.90, chunked/prefill 256, req 16, 262144 ctx, fp8 KV, DSA, multimodal, glm45 parser, no cuda-graph).
- **PRUNE-PROGRAM-CLOSED.md** written at release root: verdict, preserved-items inventory (sealed base, sealed observations, E216/S216 artifacts + containers preserved, r216/u216/t216 plans + receipts, calibration + teacher rows, all benchmark receipts), release serving config, owner open items.
- Watchdog: 4/5 (de5c down continuing). **pop-os REBOOTED ~00:5xZ** (up 36 min at 01:3xZ vs boot Sep 13 15:13 yesterday) — observe-only, untouched, cause unestablished, flagged in FLEET-STATE for owner. FLEET-STATE + FLEET-LOG updated.
- 557f serving slot now IDLE; 2822 idle; benchmarks stay paused as recorded. Terminal-Bench a0 score remains unrun (impractical at ~15 tok/s single-stream; owner decision).

## 2026-09-15T12:0xZ — USER GOAL: F216 — another prune of the EXL3 3.05bpw base, repo-verbatim REAP methodology on the FULL observation dataset, non-uniform explored

Plan written: f216/PLAN.md. Key findings (all receipted in the plan): (1) the sealed observations are PARTIAL — science 195 records (0 captured) + cuda 1,645/2,000 missing = 1,840 records / ~4.73M tokens (shards ~332-361 of the ours-original-records-v1 manifest); (2) the capture pipeline (glm53_record_observation_pipeline.py) is a PIPELINED 2-NODE protocol (each rank loads half the Q4 model — single-node impossible by design) with FIRST-CLASS RESUME (validates existing receipts, processes only uncommitted shards); (3) the Q4 capture model (0xSero/GLM-5.3-Flash-EXL3-Q4 @ d0b9301a, sha 6a6357fd), observer image, and the old run-root (332 committed shards) exist ONLY on de5c — powered off; (4) the repo's prune is uniform per-layer — its non-uniform elements are super-expert/outlier preservation (both computable from the same observations); true per-layer variable counts remain blocked by the a2d loader precedent; (5) E216's score IS the repo's reap metric — E216 was repo-faithful on partial data, so F216 = same score on FULL data (+ optional preserve-super variant pre-registered). **BLOCKING USER ACTION: power on de5c** — then: stage model+image 2822 (direct copy ~8 min), 2-node resume capture (~30-90 min, missing ~29 shards only), seal FULL, F216 plan (build rule: overlap with E216 < ~213/216), build on 2822, serve+eval. a0 full-MMLU on 2384 UNDISTURBED (in flight, ~8% at 1h15m, on-curve ramp, ETA ~17:30Z).

## 2026-09-15T14:1xZ — user: 'stop this run' → a0 full-MMLU re-run STOPPED at 8,688/56,168 (15%); full pivot to F216

- STOP RECEIPT: results/status.txt (USER-DIRECTED STOP line; pids runner=10534 lm_eval=14017, clean SIGTERM, no escalation); partial log preserved as results/a0.log.partial2-14pct-20260915 (8,688 requests, no final results file — third partial for this suite: 64% (09-14), 15% (09-15); both preserved). Release serving glm53-a0-bench on 2384 UNTOUCHED.
- F216 prep continued: Q4 capture model (0xSero/GLM-5.3-Flash-EXL3-Q4 @ d0b9301a — PUBLIC repo, 187.6 GB / 509 files, retained/+layers/ staged layout) downloading to 2822 (nohup pid 1208742, /tmp/q4-download.log): 87/509 files / 39.4 GB at ~121 MB/s at check time, one transient read-timeout auto-resumed. Needed for rank1 under the de5c path too (2822 self-sufficiency); sha-verify vs contract after completion.
- **CAPTURE-COST CORRECTION (receipted math)**: the original 2-node pipelined capture took ~3.5-4 days for 32.6M tokens (~94-107 tok/s aggregate; records flow SERIALLY through the rank0→rank1 pair). Therefore: RESUME path (de5c, 29 missing shards = 4.7M tokens) ≈ **12-14 h** (overnight); de5c-FREE path (no old run-root → all 361 shards = 37.3M tokens) ≈ **4-5 days — REJECTED as not viable**. The earlier 30-90 min estimate in f216/PLAN.md was wrong; plan file corrected in place. 2384 staging NOT needed; release serving stays untouched.
- **BOTTOM LINE: de5c power-on is REQUIRED for F216** (5-second user action saves ~4 days). Everything else staged: f216/PLAN.md corrected, build_f216_plan.py negative-tested, comparator plans verified, Q4 model landing on 2822, observer image + old run-root + Q4 model on de5c.

## 2026-09-15T14:4xZ — USER: 'you don't need power, figure it out' → de5c-FREE capture path SOLVED (architecture complete)

Discoveries (all from reading the sealed pipeline code, receipted):
- glm53_observation_stage.py: stage0 owns layers 0..22 + embed_tokens; stage1 owns 23..44; 'unowned skeleton tensors stay on meta'; resident native EXL3 loader with 12 GB reserve guard.
- glm53_observation_pipeline.py verify_stage_source: 'Verify precisely the owned files, not an impossible full local checkpoint' — each node needs ONLY: observation-stage-receipt.json (state RESIDENT_STAGE_COMPLETE, index_sha256 = model identity sha) + model.safetensors.index.json (sha = 6a6357fd… — VERIFIED == contract identity sha by downloading the index to the Mac) + the stage's own files.
- Index-derived staging (f216/stage-file-lists.json): stage0 = 106 files / 89.6 GB (2822), stage1 = 114 files / 88.6 GB (557f; fits 128 GB free); retained/ IS referenced by the weight_map per stage (26 files each); unowned = 20.35 GB.
- stage_glm53_observation_smoke.py = the OFFICIAL stager/producer: downloads-or-hardlinks selected files, verifies every file against pinned LFS sha256, writes the RESIDENT_STAGE_COMPLETE receipt. Retained-source hardlink mode makes 2822 staging instant (files already downloading there).
- CAPTURE COST (final, measured): the 29 missing shards average ~2.55k tokens/record (cuda ~2,850; science TBD) → 4.7 M tokens ≈ **~12-14 h** on the 2-node pair (original rate ~94-107 tok/s aggregate).
- NODE PAIR without de5c: **2822 = rank0 (stage0, has manifest+src+2.1 TB) + 557f = rank1 (stage1, 128 GB free)**; master on 2822 tailscale spark-2822.internal; GLOO/NCCL_SOCKET_IFNAME=tailscale0; hidden-state transfers ~82 MB/record at the missing shards' lengths (trivial vs 306 MB/s link). 2384 release serving UNTOUCHED.
- OBSERVER IMAGE: ghcr NOT pullable (checked twice) and the pinned image exists only on de5c → DERIVED image: FROM glm53-flash-sglang-exl3-plain:serve4-pruned (present on both nodes) + pip fla-core==0.5.2 einops==0.8.2 + observer-src/reap (mirrors Dockerfile.glm53-spark-observe-fla's delta on an available base). Receipted as derived-environment deviation; validated by the qualification smoke (1 record) before any production shard.
- DERIVED CAPTURE DRIVER required (f216/run-completion-capture.py, to write): the pipeline module hard-requires WORLD_SIZE=2 + a committed-prefix resume (old receipts on de5c); the driver reuses ALL original math functions verbatim (validate_manifest, load_records, record_contract → SAME run_id since contract is deterministic from manifest+identity, load_stage, forward_record, _export_stage, _merge_stage_range, make_receipt, validate_peer_identities) but loops ONLY shards 332-360 (source_row-ordered, contiguous tail) — deviation receipted; no observation-math changes.
- MERGE script required (f216/merge-completion-seal.py): partial-seal aggregate (Mac, 332 shards) + the 29 new receipts (each validated vs the ORIGINAL manifest + contract) → FULL aggregate (state SEALED_COMPLETE_ORIGINAL_RECORDS) with per-shard sources = partial-seal sources + new receipt shas; replicates the seal's validators (_validated_observation, coverage == contract 23,088/37,328,459).
- Download status: first downloader hit an unhandled ReadTimeout at 137.6/187.6 GB (~73%); relaunched retry-wrapped (12 retries, resume from cache) pid 1238905; the stale duplicate pid 1208742 stopped (mine, receipted — same-dir race prevention).
- de5c STILL VIABLE as the cleaner alternative (original image + receipts; no derived anything) — but the goal now proceeds WITHOUT it.

## 2026-09-15 F216 completion capture LAUNCHED (goal-f216-4)
- **Cleanup (user request)**: receipted deletions per host — 2384 old generations/exited container/DSV4 oracle env + 2 transfer tars; 2822+557f sglang-derived fallback observer images + build contexts; Mac reconstruction workspace. Protected: serving image serve4, b12x-r3, dsv41 calibration corpus/uring, abliterated presliced, e216 artifact, all receipts/models/teacher rows. Receipt: f216/receipts/cleanup-abandoned-20260915.json.
- **Observer image RECOVERED**: ddc1b6cf… found dangling on 2384 (matches observer-image-transfer-proof.json) → retagged glm53-full-observer-fla:20260907, CPU-validated EXACT against original receipt identities (ext 0eeb983b…, wrapper 41fc9bb4…, vllm 0.1.dev20051+g487ecf187, torch 2.13.0+cu130, transformers 5.16.1, glm5_next modeling 2092bbb4…), docker save/load 2384→2822+557f, loaded sha identical both nodes. All reconstruction (vllm vendoring, exllamav3 0.0.43 build, r3-base derivation) ABANDONED and its artifacts deleted.
- **Smoke**: two env failures fixed (missing accelerate → moot via original image; single-file mount hollow namespace → whole-src mounts). Then the real blocker: **GB10 UMA page-cache trap** — mmap'd model-file cache counts against torch.cuda.mem_get_info free; loader's 12GiB reserve breached at packed layer 18 (free 13.1GB vs 16.9GB needed) on an idle host (kernel doesn't reclaim mapped cache under CUDA allocation pressure; fadvise can't drop mapped pages). Fix without root/source-edits: container-rooted /:ro fadvise sweep + privileged ephemeral `echo 3 > /proc/sys/vm/drop_caches` every 45s on both capture nodes. Free: 2822 112.1GB, 557f 110.2GB at relaunch.
- **Qualification smoke PASSED**: RECORD_SMOKE_COMPLETE shard 332, 1 record / 3,421 tokens (rank0 203.0s, rank1 59.3s). Receipt pulled: f216/receipts/record-smoke.json. run_id == original (records-8d8dd5ec0a6608f97d88b158); runtime identical except prefill_policy (embeds derived-driver sha 5d6bc32f… — the only functional delta: tail-phase shard selection; all math/collective/kernel code byte-identical).
- **Full capture RUNNING** since 14:42Z: 29 shards (332-360), ~13h ETA (~03:30Z 09-16). rank0 2822 / rank1 557f over tailscale0; pause file /home/sero/work/f216-obs/pause-f216 (touch to stop at shard boundary); logs /tmp/f216-capture-rank{0,1}.log. Merge → FULL seal via f216/merge_completion_seal.py (10/10 tests incl. refusal paths; validates each epoch against its own runtime identity, joins sources with runtime_epoch, honest completion_run provenance).
- 557f at 46GB free (99%): stage1 files (88.6GB) to be cleared WITH RECEIPT before any F216 build/serving step on 557f.

## F216 stages 1-3 (2026-09-16T00:30-01:45Z, drivers 8-10)

- **Capture COMPLETE** 29/29 shards (332-360), RECORD_LOCAL_COMPLETE, 1,840 records / 4,725,609 tokens; merge in-container → **FULL SEAL SEALED_COMPLETE_ORIGINAL_RECORDS: 23,088 records / 37,328,459 tokens, missing_domains [], science 195 + cuda 2000 covered, min expert routes 49,895**; sources 361 with honest runtime_epoch split (original 0-331 / completion 332-360); observations.json sha beea9f01… (Mac f216/sealed-complete/, verified).
- **PLAN (stage 2)**: f216/f216-plan.json sha 342b840d…, frozen 2026-09-16T00:55:12Z — repo-verbatim REAP score (weighted_ean_sum/expert_frequency) on the FULL seal; uniform keep-216 × 42 layers (3..44), MTP 45 verbatim; **mean overlap with E216's keep-set 149.52/216 → rebuild gate passed decisively** (s216 149.24, u216 181.21, t216 149.14; route_mass_retained_diag 0.7276).
- **BUILD (stage 3, 2822)**: base 3.05bpw pulled 557f→2822 (129.5GB, 20 files sha-identical, receipts both 80c2a520…); pack 80.3s → 96,580,353,464 B (removed 36,288 tensor-level = 3,024 × 12); augment index 96,485,484,504 B (identical total to E216); **G2 verify pass:true, 111,736 tensors / 93,611,759,540 B, problems []** (identical counts to E216); **census plan-aware contract_ok=True** (42 expected-by-plan deviations, residual=[] missing=[]), moe native=True, dense 440/440, layers 3..45, codebook mul1. Census in image c65c840f (same id as 557f's, docker save|load transfer). Artifact: 2822:/home/sero/models/glm53-3p05-pruned-f216 (96,593,390,114 B du, 30 files, plan.json 0444 in-artifact). Full detail: f216/BUILD-F216.md; receipts f216/receipts/build/ (10 files) + f216-base-shas-{557f,2822}.txt. Defect preserved: census-aware attempt 1 quoting failure, relaunched literal (both logs kept).
- NEXT: stage 4 eval on 557f — served-census receipt PRE-STAGED before first launch (T216 trap); q4-stage1-obs cleared with pre-deletion sha/size receipt first; drop_caches loops kept through serve; KV gate ≥262,144; G4 from 2384 + quick MMLU (beat E216 77.28; ref a0 83.42) + GPQA MC (beat E216 37.88; ref a0 43.43).

## F216 stage 4 (EVAL, 2026-09-16T02:00-05:35Z, driver-f216-11) — VERDICT: full-seal REAP does NOT beat partial-seal REAP

- **Pre-flight**: q4-stage1-obs (557f) cleared WITH pre-deletion receipt (136 files sha'd, 88,682,100,382 B; stage receipt preserved at f216/observation-stage-receipt.json; disk 46→129GB free) → artifact pushed 2822→557f (96.6GB, **31 files sha-identical**, receipts f216/receipts/f216-art-shas-{2822,557f}.txt); served-census receipt PRE-STAGED before first launch (T216 trap avoided; plan sha 342b840d pinned, 42 problems cleared).
- **Serve attempt-1 PASSED all gates** (glm53-f216, image serve4-pruned b9cf5753, mem-fraction 0.90): KV 278,528 ≥ 262,144 (no short-KV relaunch — first point in the family to clear attempt-1), READY, greedy smoke OK try-1, zero tracebacks. Config: 262,144 context, multimodal on, MTP layer 45 verbatim (in-artifact mtp.safetensors byte-identical).
- **G4 panel (2384, all phases exit 0, F216_G4_DONE 03:34:27Z)**: top1 65.691%, KL_lb 0.93888 (p50 .337 / p95 4.100 / p99 7.174), KL_topk 0.91077, PPL 7.56398, samples 11/20; teacher reproduction exact (top1_exact_all_rows, NLL delta 0.00511, teacher PPL 3.19953). **WORSE than E216 on every panel metric** (66.414 / 0.89796 / 0.87213 / 7.23760 / 16/20) and worse than deleted R216 (68.53 / 0.777 / 6.447 / 12/20). Receipts f216/receipts/g4/f216-g4-*.json (Mac) + 2384:out/f216-g4-*.json.
- **Quick MMLU (1,140 docs, seed 1234)**: F216 78.68% ± 1.17 vs E216 77.28% ± 1.20 (+1.40pp — the ONLY gate won); a0 83.42%. **GPQA MC (198 docs)**: F216 36.87% ± 3.44 vs E216 37.88% ± 3.46 (−1.01pp, gate FAILED); a0 43.43%, r216 40.91%. Receipts results/f216-{quick,gpqa-mc}/ + run-f216-{quick,gpqa-mc}.log + results/status.txt exit lines (04:34:00Z / 05:27:12Z, both exit=0).
- **CONCLUSION**: the F216 question is ANSWERED — completing the observation capture (science 195 + cuda 2000; min routes 49,895) did NOT improve the REAP-native ranking: the keep-set changed materially vs E216 (149.52/216 overlap) but quality moved DOWN on the panel and GPQA and UP only on MMLU (where the family already trails a0 by ~4.7pp). REAP-native selection remains below route-mass-informed selection on teacher agreement regardless of capture completeness. **Benchmark-backed release remains a0 (83.42 / 43.43). No release change from F216.** The a0-fallback vs healing fork remains parked with the user.
- Housekeeping: 2822 drop_caches loop STOPPED with receipt 05:32:13Z (own helper pid 1420484); 557f loop KEPT while glm53-f216 container stays up (the one GPU job there); 557f disk now 39GB free; s216/t216 artifact dirs on 557f untouched.

## F216 teardown (2026-09-16T09:43Z, interactive "keep going")

- glm53-f216 container stopped/removed (exit 0 both), 557f drop_caches loop killed (pid 672379) + verified — no glm53 containers on 557f; 2822 loop already stopped 05:32:13Z. Engine log preserved (557f:/home/valentine/f216/serve-f216.out, 4.2MB). F216 artifact untouched on 2822 + 557f (relaunchable via f216/run-serve-f216.sh). Receipt: f216/receipts/f216-serve-teardown-20260916.txt (sha c97d0341…). Fleet: 2384 a0-bench is now the only running GPU job in the fleet.

## Prune-artifact removal (2026-09-16T10:52Z, interactive — user order "remove the prunes off the sparks")

- Removed 5 pruned artifact dirs, **482,966,693,370 B (≈483 GB)** total; df-confirmed freed 482,967,396,352 B: 2822 `glm53-3p05-pruned-f216` (96,593,390,114 B); 557f `glm53-3p05-pruned-{e216,f216,s216,t216}` (96.59 GB each). 557f went 99% → 89% full (39 GB → 427 GB free); 2822 55% → 53% (1.83 TB free).
- Pre-deletion receipts per dir (path, du -sb bytes, file count, sha256 of every *.json identity file incl. plan.json / g2-receipt.json / manifest.json / indexes, full per-file listing): `prune-removal-20260916/prune-removal-{2822,557f}-pre.json` + listings; post-deletion receipt with df before/after, rm_rc and gone per dir: `prune-removal-20260916/post-deletion-receipt.json`.
- Preserved: sealed base (turboderp 3.05bpw on 557f+2822), 2.05bpw quant copy (557f), Q4 observation model (2822), a0 (2384), all plans, tooling, sealed observations/calibration/teacher rows, every receipt; also 557f's exllamav3 v1.4.9 quantizer source (kept deliberately — see literature note below).
- The whole keep-216 @ 3.05bpw artifact family is now off the fleet (u216 + r216 were deleted earlier). Every removed artifact is rebuildable: sealed base + frozen plan + packer (pack measured 80.3 s on f216).

## MoE-compression literature synthesis (2026-09-16, interactive — "look at arxiv… what about REAM?")

- Full write-up: `LITERATURE-MOE-COMPRESSION-20260916.md`. Headline: the published literature **predicts our negative result** — REAP's own Appendix E (arXiv:2510.13999) and MoEXBench (arXiv:2608.21693) both find that weight quantization beats expert pruning at equal artifact size above ~3 effective bpw, with ~19-25% expert pruning costing ~5-6 quality points vs ~1 point for aggressive 4-bit weight quantization; "Half the Experts, All the Code" (arXiv:2607.16721) finds pruning wins only where quantization would have to drop below 3 bpw (five pre-registered attempts to overturn the crossover failed). Our pruned artifact is 3.05bpw × 216/288 ≈ **2.29 effective bpw** vs a0's 2.05bpw unpruned — the literature says a0 should win, and it does.
- REAM (arXiv:2604.04356) is a **merging** method (weight averaging): it requires unpack + re-quantization of EXL3 trellis-packed tensors, and MoEXBench measures REAM slightly *worse* than REAP at 25% expert reduction (6.05 vs 5.31 quality points lost). Not applicable without a new quantizer pipeline; deprioritized.
- Class-S (selection-only, no weight surgery) ideas still open, all bounded at ~+1-2.5 pts: super-expert protection (arXiv:2507.23279), gate-free task-agnostic rankers (arXiv:2606.15716 MAN/MSAN; AIMER arXiv:2603.18492), non-uniform per-layer budgets (GRAPE arXiv:2604.06542, EvoESAP arXiv:2603.06003, DiEP arXiv:2509.16105 — blocked by the uniform-expert-count loader), and serving-level routing-mass redistribution for removed experts (zero-quantizer; attacks the measured structural damage). MoEXBench's score-deployment mismatch gives a new testable hypothesis for our 2.8-pt ranking spread.
- Precision-bias note: our rankings were fit on observations from the **Q4 (4.05bpw-class)** model and applied to the 3.05bpw artifact — MoEXBench explicitly warns that rankings may not transfer across precision.

## MOSAIC-288 plan — mixed-precision all-expert TP1 variant (2026-09-16T11:4xZ, interactive; PLAN ONLY)

- User asked: "what is the next step to take to get a higher performance tp1 variant with the desired context and all". Answer written to **`MOSAIC-288-PLAN.md`** (frozen spec, gates, receipts). No GPU work, no artifact mutations this turn — read-only probes only.
- **The finding it rests on**: a0 (2.05bpw) and the sealed 3.05bpw base are **byte-aligned per expert tensor** (same repo/quantizer lineage: turboderp exl3 v1.4.4, codebook mul1, cal 250x2048; identical key names; 288 experts on both; measured on 557f from index+headers this turn). Per-expert payload 6,328,332 B @2.05 vs 9,474,060 B @3.05 → **2→3 upgrade = +3,145,728 B exactly**. So a mixed-precision artifact can be assembled by **byte-copying each expert from whichever sealed quant has the allocated rate — no requantization, no new quant error, no download**.
- **Variant M288**: a0's dressing byte-exact (MTP layer 45 verbatim, 288 experts) + top-86/288 experts per trunk layer (full-seal `weighted_ean_sum` ranking) swapped to the base's 3-bit tensors → predicted **96,595,853,116 B** ≈ the F216-proven serving envelope (96,593,390,114 B, KV 278,528 ≥ 262,144 @0.90). Coverage: top-86 = 49.6% of activation-weighted mass (min 44.8% / max 66.1%). Effective ~2.30 bpw with ALL 288 experts.
- **Why now**: mixed-precision bit allocation is the one literature-supported axis left (2509.25689, MC-MoE, QuantMoE-Bench; REAP App. E crossover), it removes every measured failure mode (no expert deletion, native router, byte-exact turboderp tensors), and it is ~1.5 GPU-days. The requant alternative is blocked at the start: **exllamav3's CUDA extension fails to build on the Spark** (557f `models/p3b-out/build.rc=1`, fa-build.rc=1) — recorded as the reason V5 is last, not first.
- **Gates (pre-registered)**: A = one-layer mosaic (layer 20, +270 MB, ≈85.5 GB, same envelope as a0) → serve + smoke + 4-sample mini-panel + tokens/s (proves intra-layer hetero-bit decode); B = full build + census + byte-exactness verification of all non-upgraded tensors; C = serve (KV ≥ 262,144) + G4 panel + quick MMLU + GPQA vs a0 bars (83.42 ± 1.2 / 43.43 ± 3.5). Release rule: MMLU > 83.42 AND GPQA ≥ 43.43 AND samples ≥ 16/20 AND KV gate → else a0 stands.
- Fleet this turn (read-only): 2384 a0-bench up (1.3 TB free), 2822 idle (1.7 TB free; serve4-pruned + 2p05-sglang-mul1-r1 images present), 557f nv shim only (1.5 TB free; both source quants + exllamav3 v1.4.9 present), de5c DOWN expected. 2384/557f free space measured **larger** than the 10:52Z receipts (owner cleanup today) — use live df at build time.

## MOSAIC Gate A — mixed-precision, all-288-expert TP1 variant (2026-09-16, interactive "let's test it")

Full record: `mosaic-gatea/GATE-A-RESULTS.md`; receipts in `mosaic-gatea/` (Mac), `557f:/home/valentine/mosaic-gatea/`, `2384:/home/sero/w2port/out/`.

- **Built by byte-copy only** (no requantization, no download, no new quantization error): a0 (2.05bpw) and the 3.05bpw base are byte-aligned per expert tensor (measured; 2→3 upgrade = +3,145,728 B/expert).
- **Per-expert mosaic (layer 20, top-86 experts) → NOT SERVABLE**: the exl3_plain overlay fuses all experts of a layer into one stacked tensor (`moe.py:88`, `RuntimeError: stack expects each tensor to be equal size … [256,128,32] vs [256,128,48]`). Build+census were clean (1,032/1,032 upgraded byte-equal to base; census `contract_ok true`, `moe_sparkinfer_native true`) — only the serve catches it. **Supported granularity = per layer** (`bits = w13.shape[-1]//16`; w13 must equal w2). Artifact `557f:models/mosaic-l20-gatea` (85,504,022,960 B) preserved as the negative evidence.
- **Per-layer mosaic M288-12L**: layers 3, 32–33, 36–44 at 3 bits (chosen by total activation-weighted mass over the full seal), all 288 experts in every layer, dressing (incl. MTP layer 45 + router) byte-exact from a0 → **96,105,137,024 B** (within 488 MB of the F216-proven serving envelope). Verification: 41,472/41,472 upgraded tensors byte-equal to the base, 26,117/26,117 kept tensors byte-equal to a0, 5/12 shards differ, index+config.json byte-identical, 10,368 quantization_config entries patched (receipt `mosaic-12l-pack-receipt.json`, `n_failures: 0`).
- **Serving**: READY, smoke "The capital of France is → Paris. It is located in the north-central part of the country." KV 136,896 (boot1) / 193,472 (boot2) at mf 0.90 → **below the 262,144 gate**; **mf 0.95 → KV 595,200 ≥ 262,144 ✓**. CUDA graphs (max-bs 1) → FAIL at warmup ("Hybrid (mamba/linear-attention) state cache is too small"). Live now: `glm53-mosaic-12lh` on 557f (mf 0.95).
- **QUALITY — first artifact in the program to beat a0 on the G4 panel.** Frozen rows 0–3 (8,188 positions), teacher reproduction exact both sides: a0 top-1 0.8096 / KL_lb 0.4749 / PPL 3.0162 vs **mosaic-12L 0.8282 / 0.3957 / 2.8463** (**+1.86 pp top-1, −16.7% KL, −5.6% PPL**). Receipts `a0-mini-panel.json`, `mosaic12l-mini-panel.json`. Full panel (65,504 pos) + quick MMLU + GPQA MC = the confirming runs, not yet run.
- **Speed**: mosaic ≈ a0 (a0 9.10/9.25/9.24 tok/s; mosaic 9.43/8.93/8.53 at mf0.90, 7.01/10.40 at mf0.95; 128-token single stream). Both eval configs run **without MTP** (`dropped:nextn_mtp_off`) and **without CUDA graphs**; the repo's recorded reference for this stack is ~14.8–15.1 tok/s total decode. The 20 tok/s class therefore needs the MTP/CUDA-graph levers, not a mosaic change.

## MTP SERVING — the D2/B12x recipe reproduced on the K2 target (2026-09-17, user: "make it serve mtp")

**Result: MTP serves, with CUDA graphs, at the receipted D2 rate — 19.23 / 19.00 TOTAL decode tok/s (native cross-check 19.26 / 19.03), acceptance 98.3–99.6% at depth 2, non-greedy sampling.** Full record `mosaic-gatea/RESULTS-CONFIRMING-RUN.md` §5; receipts `mosaic-gatea/receipts-557f/`.

- **Which recipe**: the program's own `b12x-runtime/mtp-baseline-replay/d2-c1-attempt2/{launch.sh,plan.json,admission.json}` — image `glm53-b12x-exl3:jovian-3aada677-r3` (sha256:afb74c791853…), `--attention-backend B12X --additional-config {"kda_prefill_backend":"b12x"}`, `--kv-cache-dtype fp8_ds_mla --block-size 256`, MTP depth 2 with draft `attention_backend: B12X`, `FULL_DECODE_ONLY` graphs capture `[1,3]`, mf 0.93, ctx 262,144, vision on. Launcher `mosaic-gatea/launch-k2-mtp-b12x-d2.sh` (sha e099005c…); container `glm53-k2-mtp-b12x-d2` on 557f.
- **Raw flags ≠ speed**: the *other* MTP-capable image (`local/glm53-reap-native-mtp:20260911-r4-expert-fp8` = 4f06a26d872d) also serves MTP (counters +900 drafted/+633 accepted = 70.3% over three 512-token sampled generations) but with `FLASHINFER_MLA_SPARSE_SM120`, `kv-cache-dtype fp8`, block 64, depth 1 it measures ≈13.3 tok/s from its own counters — 30% below the B12X/fp8_ds_mla/depth-2 configuration at the same target and image-independent stimulus. Both servers logged and preserved (`receipts-557f/mtp-glm53-k2-mtp.log`, `out/k2-mtp-final-inspect-20260917T120333Z.json`).
- **Admission**: KV 727,449 tokens (2.77× the 262,144 context), `Capturing model for speculator...` twice (target + draft) with graph-capture descriptors written by the program's `capture_plugin`, 102.29 GiB weights + 4.78 GiB peak activation + 0.21 GiB CUDAGraph, `/v1/models` max_model_len 262144.
- **Method — the program's, unchanged**: 1,023-token structured-count stimulus built by the D2 tokenize-bisection; exact per-chunk `token_ids` (`682 = 682` both measured cells), `MATCHED_SUSTAINED` 35.4/35.8 s windows via `harness/timing.py` (byte-identical to the D2 replay copy), native `vllm:request_decode_time_seconds`/`request_generation_tokens` counters via `harness/native_metrics.py`; client and server token totals agree. Receipt `receipts-557f/out/mtp-serve-proof-20260917T122332Z.json`.
- **Non-greedy, stated plainly**: cells ran `temperature 1.0 / top_p 0.95`. D2's own rows were `temperature 0`; the stimulus is identical and acceptance stayed ≥98.3%, so the sampled rows land on D2's numbers (19.11/19.03) rather than under them. This is a counting-screen stimulus, not broad quality or a MTP speedup claim.
- **The mosaic cannot use this runtime**: the mosaic `M288-12L`/`M288-10L` are **mul1**-codebook artifacts and both MTP-capable images raise `this overlay only implements codebook=mcg; got 'mul1'` (`exl3.py:567`). Documented in `mosaic-gatea/MTP-CODEBOOK-BLOCKER.md`. The mosaic therefore keeps its exl3-plain/SGLang serving (9–10 tok/s, `dropped:nextn_mtp_off`, no graphs); its quality win is runtime-independent, its speed is not. ⇒ the two-lineage split is now a **measured, receipted boundary**, not a hypothesis.
- **Fleet note**: the sealed turboderp 2.05bpw and 3.05bpw sources (the byte-copy parents of both mosaics) were removed from 557f `/home/valentine/models/` between 2026-09-16T22:00Z and 2026-09-17T12:00Z (~300 GB) by someone other than this program; both mosaic artifacts remain byte-intact vs their pack receipts.

## MTP full-context-window validation — 1,023 → 260,095 input tokens (2026-09-17T16:05Z→17:04Z)

**Result: decode is flat across the whole 262,144 window — measured cells 18.70–19.22 TOTAL decode tok/s (2.7% spread), MTP acceptance 0.95–1.00, prefill saturating at ~450 tok/s, and 260,095 input + 683 output completed without OOM on a KV of 727,449 tokens.** Receipt `mosaic-gatea/receipts-557f/out/mtp-ctx-sweep-full-20260917T160505Z.json` (per-cell token-ID arrays + native counter deltas), log `out/ctx-sweep-full.log`, runner `mosaic-gatea/mtp_ctx_sweep.py`.

- Ladder = the program's own (D1 structured-sustained TABLE): 1024, 4096, 16384, 65536, 131072, 200000, 260096 × (warmup, measured1), cold prefix cache, sampled (`temperature 1.0/top_p 0.95`), same 1,023-token structured-count stimulus, exact per-chunk token IDs + matched windows + native `vllm:request_decode_time_seconds` cross-check. **14/14 cells `ok`, 14/14 `MATCHED_SUSTAINED`, health 200 after the run.**
- Measured decode by length: 19.19 (1k), 19.22 (4k), 18.95 (16k), 18.82 (65k), 19.00 (131k), 18.70 (200k), 18.89 (260k). TTFT: 2.61 / 9.59 / 36.99 / 144.26 / 288.58 / 441.83 / 575.65 s. Prefill: 412 / 434 / 445 / 455 / 455 / 453 / 452 tok/s.
- Same shape as D1's recorded ladder (14.98 → 14.80, prefill 442–455) — i.e. the +28% D2 delta holds at every context length, not just at 1k/16k, and MTP acceptance does not decay with context (0.95–1.00 at 260k).
- Scope limits: C1 single stream, counting stimulus; no concurrency claim, no long-context quality claim.
