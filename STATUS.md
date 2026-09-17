# GLM-5.3-Flash and DeepSeek-V4.1-Flash release work

Updated September 12, 2026, 14:23 UTC. Neither new release is accepted or published. Keep all protected target weights native; prefer unpruned GLM. Pop production has not been changed.

## Current fleet state

A fleet interruption began around 09:40 UTC. Only Spark-557f has a confirmed reboot. Its previous native9 container exited255 with OOMKilled=false; last recorded capture progress was297/512rows. Readable journals did not establish a cause. Other hosts are unavailable, so their boot and workload states remain unknown. The user has been asked whether the fleet was restarted.

| Spark | Current verified state | Next action |
|---|---|---|
|557f|Encoder11sealed364experts/1092projections. Native12running196/512at14:22UTCafterR2guardrepair.|Queue10/11complete;R2queue60257runs12/13. Archiveongoing;no14launch.|
|2822|Unreachable over Tailnet and internal fabric; Tailnetlastseen09:40:11. Layer8encoder had6verifiedoutputs at09:33:58.|Inspect boot, surviving expert hashes and original checkpoints before resuming. Do not rerun its allocation queue blindly.|
|2384|Tailnetlistsonline, but SSH, TSMP and internal fabric access fail. D5C1 was loading at last contact.|Retrieve watcher/terminal receipts before claiming results or restarting.|
|de5c|Tailnetoffline,lastseen09:40:25. Projection14/NVFP4draft server was loading at last contact.|Retrieve actual startup and capacity results before another launch.|

All three unavailable hosts were rechecked by SSH at14:12UTC and still timed out. Evidence: `incidents/20260912T0940/`. Connectivity failures do not prove model OOM or explain the reboot. Sealed recovery checkpoints and terminal evidence are preserved. At12:46UTCthe old interruptednative9path was absent; its disposition is unknown. This run performed no deletion.

## Accepted GLM speed measurements

Unpruned K2 target; same pinned runtime, native FP8 MTP draft, C1,262144context configuration,2048chunk,.93memory fraction, cold prefix caching. Exact identical request payloads across depths. Two measured repeats per row, correct normal-stop JSON counting task. All reported decode windows exceed30seconds. These are task-specific throughput results, not broad quality. TOTAL equals per-request decode at C1.

| Input tokens | D1 native prefill tok/s | D1 TOTAL decode tok/s | D2 TOTAL decode tok/s |
|---:|---:|---:|---:|
| 1,023 | 424.32 | 14.98 | 19.11 |
| 4,095 | 444.31 | 14.92 | Not tested |
| 16,383 | 452.18 | 14.86 | 19.03 |
| 65,535 | 456.12 | 14.80 | Not tested |
| 131,071 | 455.18 | 14.81 | Not tested |
| 199,999 | 453.91 | 14.83 | Not tested |
| 260,095 | 453.05 | 14.82 | Not tested |

Depth2 improved decode27.62% at1k and28.06% at16k. Its native prefill was432.11 and449.20tok/s respectively. D1 and D2 containers are preserved and stopped; D5 has no accepted result. Higher concurrency has not been admitted/tested. Raw values, individual repeats, timing and task checks are in `benchmarks/structured-sustained-20260912/TABLE.md` and `b12x-runtime/mtp-baseline-replay/d2-c1-attempt2/exact-baseline-replay/`. No40tok/s claim.

The unpruned K2 baseline separately passed262016prompt+128reserve with all3retrievalcodes correct,4image cases and2video cases, and nativeMTPcounter advances. Combined near-limit vision, maximum media budgets and higher concurrency remain separate gates.

## GLM quality

Same frozen WikiText2 panel:65504positions; BF16teacherPPL3.1995295. These measure next-token fidelity, not broad task accuracy.

| Candidate | BF16 agreement | Mean KL | PPL |
|---|---:|---:|---:|
|Unpruned K2|77.384587%|0.438985908|4.546866|
|Whole layers5/32K3,restK2|78.099047%|0.417792191|4.455079|
|Projection6|77.976917%|0.419813339|4.442209|
|Projection14|78.381473%|0.406529076|4.395653|
|Unpruned Q3 reference|87.383977%|0.152204399|3.496914|

Projection14 retains288experts and native protected target weights. Its14K3down-projections add3.9375GiB overK2; gate/upremainK2. Selection used separate calibration. Its gain is insufficient for the high-quality release. Earlier pruning candidates performed worse and are comparisons, not the preferred path.

Unsloth's120.37GBunprunedGGUF uses mixed expert types and also quantizes attention, embeddings, output head and MTP; the separate vision file is excluded from that total. Its published quality panel differs. Header audit: `quality/gguf-header-audit/ASSESSMENT.md`. The lead supports an unpruned investigation but does not prove an equivalent native-protected EXL3 memory/quality result.

## NVFP4 draft and runtime gates

- Matched full288expert loads measured2.9531GiB lower resident CUDA allocation with draft-onlyNVFP4, preserving native protected heads/embeddings. No target/KV was present in that comparison.
- The full288MoEcomponent passed10reference cases and30bitwise changed-input/route graph replays; maximum routed-reference relativeL2was0.0003726. This is not full-server acceptance.
- WarmupR2 adds the separate nativeMTP draft to eager B12x warmup before capture. Five actual installed CPU orchestration tests passed; image6c6581ea1454 preserves parent layers/env/entrypoint. GPUstartup remains to be recovered.
- Full-servingattempt1 failed stock-loader discovery of nested weight paths. An identical-inode flat view passed actual CPUloader/header checks for133files,583090targetentries and891draftentries. Attempt2 was loading when connectivity was lost. Original model/index remain untouched.
- Depth2attempt1 returned correct text but lacked actual draft-decode graphs. Capture sizes[1,3] fixed it; correctedD2passed actual admission and measured tests. D5uses[1,6] but is unverified.

Details and all failed attempts are preserved under `b12x-runtime/mtp-nvfp4-serving/`, `mtp-nvfp4-warmup-r2/`, `mtp-full-sparse-proof/`, and `mtp-baseline-replay/`.

## DeepSeek checkpoints

Experts-onlyK3,sourcefb2764a5cf321eaa5070ca8f9e892818f477c16d. Native nonexperts remain protected. Strict natural-route gates remaintrain>1024andheldout>128; no forced routing.

| Layer | Verified encoder state | Coverage gaps |
|---:|---|---:|
|3|374experts including independently verified supplementalexpert41|10|
|4|373experts/1119projections sealed|11|
|5|373experts/1119projections sealed|11|
|6|366experts/1098projections sealed;4,873,806,792bytes|18|
|7|364experts/1092projections sealed|20|
|8|Native512rows sealed,366eligible. Encoder reached6verifiedoutputs before access loss; terminalunknown.|18|
|9|Recovery native512rows sealed. Encoder9 sealed369experts/1107projections; terminalexit0,noOOM.|15|
|10|Native512rows and encoder365experts/1095projections sealed; rootterminal/pinchainchecks passed.|19|
|11|Native512rows and encoder364experts/1092projections sealed; rootterminalandparentchainverified.|20|

Layer8's lossless compact inputs were SHA-verified in2822hostRAM tmpfs, with a separately SHA/header-verified7.405GBsource shard inside bounded container tmpfs. Volatile staging may be lost if that host rebooted; retained native8 on557f enables recovery. Do not assume the old READYreceipt proves current tmpfs contents.

Recovery9 retains originalpartialfiles and uses new output/container identities. Source and all512parent rows were revalidated after reboot. The first recovery preflight caught stdlibqueue shadowing by a driver namedqueue.py; rename fixed that before GPU launch. Evidence: `/Users/sero/sessions/deepseek-resume-20260911/next-pair-8-9/recovery-20260912T1004/`, remote `/home/valentine/deepseek-v41-exl3/layer9-recovery1-evidence/`.

## Remaining release gates

1. Recover authenticated access and inspect interrupted jobs/checkpoints; continue native9 and encoder9 on557f meanwhile.
2. Improve unprunedGLMquality materially, prove actual262144context/media/nativeMTP, qualify capacity and sustained speeds across admitted concurrency.
3. Finish allDeepSeekexperts with coverage, then validate two-Spark B12x/NVMeEngram/DSpark/vision,400kcontext and40tok/s, followed by REAP observations.
4. Publish only accepted HFweights and reproducible Docker/GitHub recipes, with clean attribution and an independent fresh reload.

`EXPERIMENTS.md` is the historical evidence ledger; `GOAL.json` holds coordination identities. No GPU power changes, arbitrary process killing, or removal of original model/calibration data.

## September 12 10:57 follow-up

Native9 recovery passed all56,598,282,240bytes of calibration validation and terminal0/noOOM checks. Root fetched the exact receipts; seala50afd868f5a79a04844e581be3a5440a75df38ae9ed1439193fe7a069d4d36d. Capture took861.175seconds; this is calibration time, not serving throughput. Encoder9 containerdf4d1207183a is advancing with the original strictcoverage and codepins.

Sources10/11 are pinned to published HF fullSHA41d87a4c81fec1550f9cb975db05598a18ee0161c2664e8e1a0c7b60742755a6 ande7ca4a12688a5819829ead3a03282aedb380e438bc41a00e969f70952c6d0eb9. Each is7,389,761,368bytes. Queue18455 waits for the exact successfulencoder9seal, then performs native10/encode10/native11/encode11serially. Stagebudget preserves room for both fullparents, outputs and reserve. No source/parent deletion.

GLM CPU accounting inquality/kv-budget-followup/ASSESSMENT.json identifies a prospective tradeoff: D2C1 reports753664aggregateKVtokens despite admitting one262144request. A portion might fund additionalK3expertprojections. Linear estimates are not runtime admission; hybridblockrounding, actualfull-draft savings, loadingpeaks, media andOSreserve mustbe measured. No protected precision/context change or qualityprediction is claimed.

## September12,12:11UTC additions

Coverage ledger: `/Users/sero/sessions/deepseek-resume-20260911/coverage-ledger/TABLE.md`. There are122expert-layer calibration gaps across layers3-10; other layers are outside this audit. Layer3expert41is closed through its separately verified supplement, not by changing original seals. Train>1024andheldout>128remain strict; deficits are route lower bounds, not token budgets.

Native10 root receipt: `/Users/sero/sessions/deepseek-resume-20260911/next-pair-10-11/ROOT_NATIVE10_TERMINAL_RECEIPT.json`; sealSHA3787c54d1cb23886ad04de3186144be3845d955e070a04182e86eccacfa4a086. Diskfree125.05GBat12:10UTC; queuednative11needs56.60GBplusremainingencoderoutputs. No new pair downloaded or original data removed.

GLM20/22down-projectionK3 designs are frozen under `quality/kv-budget-followup/candidate-designs/`. Both keep288experts and native protected weights; only independent paired calibration selected layers, checked against prior14map and full42×288expert coverage. Generator reproduced identical SHA on a second invocation. Extra targetbytes overK2are5.625/6.1875GiB. They are not built, GPUadmitted or qualitytested; allocation estimates are not actual KVcapacity. No new GLM results claimed while its hosts remain unreachable.

## September12,12:52UTC continuation

Encoder10 sealSHA `c40f574f3e54010e5f5539fe165bd049564429320b6d275b5d2f77fe9b1f4f26`;365experts/1095projections,4,860,490,380bytes,19gaps. Native11 actualCPUpreflight passed all512parenthashes/headers and restore, then launchedcontainerde71ca13a70c at12:51:32UTC. At launch0rows; no native11completion claim.

Directdownloads12/13 use pinnedofficialshards15/16; process41133. Sequentialqueue41134waitsencoder11terminalbefore native12/encoder12/native13/encoder13. Eighteen deployedsource/pin files matchedrootSHAs. Launches retain native sourceprecision, strictnaturalroutecoverage and idleGPUguard. Storage check reserved214.116GBof235.521GBfree for alreadyqueuedcaptures/encoders/sourcesplus12GiBfloor;12downloadshowed1.735GBof7.390GBat12:47UTC. Neither12nor13sourcefullSHApassed yet.

Layer14admission plan: `/Users/sero/sessions/deepseek-resume-20260911/layer14-preflight/ADMISSION.md`. Header-onlyHTTP206auditconfirmsnativeFP8Engram+scales101.54GBshard48and7.406GBshard17. About149.46GBadditionalspaceprojectedbeyondremainingaftercurrentqueue; fullsource/parity/sharedKVboundarychecksrequired. Queueends13. No14GPUallocationororiginaldataremoval.

GLMhostsremainunreachable, so no newruntime,qualityorpublicationclaim. Unpruned20/22projectionmapsremainfrozenonly.

12:52UTCfresh-outputcheck: native11 GPUlog reached16/512rows; monitorrecorded9rowsat12:52:32.938UTC. This confirms newcaptureoutput, not completion or inference speed.

## September12,13:40UTC — native11sealed and archiveactive

Native11:512rows,364eligible/20gaps,56,598,282,240bytes,675.934879seconds calibration. Root matched terminalidentity,source/native10parentSHA,432natural+80randomsplit and independently recomputed eligibility. SealSHA4bba19869394c9a9ae3c270555e5caca78e361255b22e90fe7872c81547c6a2a. Encoder11running199/364at13:39UTC. Coverage ledger now scopeslayers3–11andcounts142expert-layergaps. Source12/13fullSHAreceiptsmatchofficialpins.

Three completednativecheckpoints5/6/7will be archived toPopSYSTEMdrive, which had571.73GBfree. Their169.795GBlogicalpayloadremainsontheSparkuntil each513filearchivepassesitssealedSHAchecks,fullarchiverereadSHAandfsync. A directIO110.55MBsmokepassedmanifest+rowhashes. Firstsmokefailedonroot-owned0600rows; nooriginalremoved. Revisedread-only128MiB,noGPUcontainerexports withlog-drivernoneverifiedbeforebinaryattach. DirectO_DIRECTread/writeandrecheck,30MiB/slimit; hostkey-pinnedhomeLAN. Worker1873162at `/home/ser/deepseek-v41-spark-checkpoint-archive-20260912T132635Z`;3.17GBcopiedat13:39,nonefreedyet. Archive5/6/7sourcesareallowlisted;retirementchecksunchangedinodes/sizes/mtimesandwaitsforoverlappingcontainermountstorelease. Fullverification/retirementnotyetaccepted.

SparkhomeLANtrafficusesWiFiwlP9s9;wiredLANenP7s7hasNOCARRIER. Actualtransfer~12–13MB/s,soallthreearchivesroughly4hoursincludingverification. Zstdlevel1testedononeexisting110.55MBsmokecompressedto84.205MBin0.154sbutnotdeployed;currentstablearchiveformatremainsrawtar. No networkconfigurationchanges. Secondary10.10.20peerSSHportsalsotimedout. Poplocal-studio-llmstillreportshealthy,22hoursuptime;memory/ioPSIaverages0at13:36;GPUrecipeunchanged. This is process/pressureevidence,notfreshAPIacceptance.

GLMnewruntime/qualitytestresultsremainunknownbecauseitsthreehostsareunreachable. No publication. Layer14planningremainsgatedonactualfreedspaceandnativeEngram/sharedKVparity.

## September12,14:23UTC — layer12recovered;archivecontinues

Encoder11sealSHA `d9660f9dd9135cc8d08d62b66223766c386b8f704a079dd31433df5d637a2947`:364experts/1092projections,4,847,173,968bytes;exit0/noOOM;20gaps. Queue10/11complete14:06:53UTC. Layer12firstattemptpassedCPUsource/parentchecksbutoldlane_guardallowedonlya historicalcompact6container,soitrejectedthenewCPUarchiveexporterandstoppedbeforecreatinganynative12output.

R2guard `next-pair-12-13/lane_guard_archive.py` admitsonlyexactarchivehelperSHA/image/emit-rootlayers5–7,readonlysinglemount,noGPUdevices,explicitGPU-invisibleenv,noNetwork,128MiBnoswap,1CPU,andlog-drivernone;globalnvidia-smicomputePIDcheckremains. Actualexporterpassed;13invalidvariantsrejectedandreproducibletestsaved. Onlyfourlauncherreferencepathsandnewguardchanged;failedlogsandoldprovenanceretained. R2queuePID60257recheckedCPUpreflightandlaunchednative12ab71bec6at14:16:37UTC;196/512freshrowsat14:22:07UTC. Source13ready. No repeatedencoderwork.

Layer14meta-contractpassedactualnativeconstructor2,347tensors,6Engramtensors,pinnedsource17/48headers,CPUonly. FirsttwoauditattemptsfailedbecausehardenedreadonlycontainerlackedtemporaryandTileLangcachedirectories;boundedtmpfsfixedenvironment,allfailedlogsretained. Thisdoesnotloadfullweights,checklookupvalues,restoresharedstateoradmitGPUserving. `layer14-preflight/META_CONTRACT.json`recordsconfig/nativecode/headerSHAs.

Archive5transfer36.40GBat14:22UTC;full169.795GBjobstillongoing,nooriginalretirementaccepted. Popproductioncontainerstillhealthy23hoursuptime;noGPUrecipechanges. GLMthreehostsstillSSHtimeout14:12UTC;nonewquality/runtime/publicationresults.
