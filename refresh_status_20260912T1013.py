from pathlib import Path
import json,statistics,datetime
p=Path(__file__).resolve().parent
b=json.loads((p/'benchmarks/structured-sustained-20260912/verified.json').read_text())['records'];d=json.loads((p/'b12x-runtime/mtp-baseline-replay/d2-c1-attempt2/exact-baseline-replay/verified.json').read_text())['records']
rows=[]
for n in sorted({x['prompt_tokens'] for x in b}):
 x=[r for r in b if r['prompt_tokens']==n and r['repeat'].startswith('measured')];y=[r for r in d if r['prompt_tokens']==n and r['repeat'].startswith('measured')]
 assert len(x)==2
 avg=lambda a,k:statistics.mean(r[k] for r in a)
 d2=f"{avg(y,'total_decode_tok_s'):.2f}" if y else 'Not tested'
 rows.append(f"| {n:,} | {avg(x,'native_prefill_tok_s'):.2f} | {avg(x,'total_decode_tok_s'):.2f} | {d2} |")
text='''# GLM-5.3-Flash and DeepSeek-V4.1-Flash release work

Updated September 12, 2026, 10:13 UTC. Neither new release is accepted or published. Keep all protected target weights native; prefer unpruned GLM. Pop production has not been changed.

## Current fleet state

A fleet interruption began around 09:40 UTC. Only Spark-557f has a confirmed reboot. Its previous native9 container exited255 with OOMKilled=false; last recorded capture progress was297/512rows. Readable journals did not establish a cause. Other hosts are unavailable, so their boot and workload states remain unknown. The user has been asked whether the fleet was restarted.

| Spark | Current verified state | Next action |
|---|---|---|
|557f|Reachable after reboot. Native9 recovery container4773699c41bb is running after source SHA and all512parent-row checks passed. Recovery driver9208.|Complete separate native9 capture, seal it, then start covered encoder9.|
|2822|Unreachable over Tailnet and internal fabric; Tailnetlastseen09:40:11. Layer8encoder had6verifiedoutputs at09:33:58.|Inspect boot, surviving expert hashes and original checkpoints before resuming. Do not rerun its allocation queue blindly.|
|2384|Tailnetlistsonline, but SSH, TSMP and internal fabric access fail. D5C1 was loading at last contact.|Retrieve watcher/terminal receipts before claiming results or restarting.|
|de5c|Tailnetoffline,lastseen09:40:25. Projection14/NVFP4draft server was loading at last contact.|Retrieve actual startup and capacity results before another launch.|

Evidence: `incidents/20260912T0940/`. Connectivity failures do not prove model OOM or explain the reboot. Interrupted outputs and containers are preserved.

## Accepted GLM speed measurements

Unpruned K2 target; same pinned runtime, native FP8 MTP draft, C1,262144context configuration,2048chunk,.93memory fraction, cold prefix caching. Exact identical request payloads across depths. Two measured repeats per row, correct normal-stop JSON counting task. All reported decode windows exceed30seconds. These are task-specific throughput results, not broad quality. TOTAL equals per-request decode at C1.

| Input tokens | D1 native prefill tok/s | D1 TOTAL decode tok/s | D2 TOTAL decode tok/s |
|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

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
|9|Originalcapture interrupted; separate recovery running on557f. Encoder not yet launched.|Not yet sealed|

Layer8's lossless compact inputs were SHA-verified in2822hostRAM tmpfs, with a separately SHA/header-verified7.405GBsource shard inside bounded container tmpfs. Volatile staging may be lost if that host rebooted; retained native8 on557f enables recovery. Do not assume the old READYreceipt proves current tmpfs contents.

Recovery9 retains originalpartialfiles and uses new output/container identities. Source and all512parent rows were revalidated after reboot. The first recovery preflight caught stdlibqueue shadowing by a driver namedqueue.py; rename fixed that before GPU launch. Evidence: `/Users/sero/sessions/deepseek-resume-20260911/next-pair-8-9/recovery-20260912T1004/`, remote `/home/valentine/deepseek-v41-exl3/layer9-recovery1-evidence/`.

## Remaining release gates

1. Recover authenticated access and inspect interrupted jobs/checkpoints; continue native9 and encoder9 on557f meanwhile.
2. Improve unprunedGLMquality materially, prove actual262144context/media/nativeMTP, qualify capacity and sustained speeds across admitted concurrency.
3. Finish allDeepSeekexperts with coverage, then validate two-Spark B12x/NVMeEngram/DSpark/vision,400kcontext and40tok/s, followed by REAP observations.
4. Publish only accepted HFweights and reproducible Docker/GitHub recipes, with clean attribution and an independent fresh reload.

`EXPERIMENTS.md` is the historical evidence ledger; `GOAL.json` holds coordination identities. No GPU power changes, arbitrary process killing, or removal of original model/calibration data.
'''
(p/'STATUS.md').write_text(text)
print('Refreshed current status; historical experiments retained.')
