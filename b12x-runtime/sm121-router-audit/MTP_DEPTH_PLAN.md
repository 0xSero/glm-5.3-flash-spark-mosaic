# Fixed native MTP depth qualification plan

Increasing draft depth1→2→3→5 can amortize the target's large native BF16 weight reads over3/4/6 verification tokens instead of2. This may matter much more than the measured0.1–0.2ms router saving. Extra draft passes also read the native vocabulary head and draft weights; rejected drafts waste that work. No depth>1 GLM GPU acceptance has been established, so this is a prospective plan.

## CPU source support

The installed Jovian V2 route supports repeated native MTP: `Glm5NextMultiTokenPredictor.forward/compute_logits` selects `spec_step_idx % num_mtp_layers`, reusing the single native layer45. `AutoRegressiveSpeculator.propose` validates depth against configured maximum, generates remaining steps, advances positions and bounds draft length by max model length. B12X MLA metadata declares `supports_draft_decode_metadata_update=True`; its updater refreshes GLM selector accepted-token metadata for subsequent steps. This permits the fused multistep branch in principle, but actual branch choice/capture must be checked in the new runtime.

MTP step0 computes its own top-k; later steps reuse compacted final token indices. `on_prefill_begin` resets skip mode, `on_prefill_end` compacts last-query rows, and multistep begin/end hooks toggle reuse. They run at capture and replay construction, with the end hook in a finally block. The pool-index table is scratch used by the indexer; final expanded token indices are what the skipped-indexer attention path consumes. Do not alias these draft buffers to target buffers or change their lifetime while evaluating depth.

Keep batch-size dynamic scheduling and acceptance-length adaptation disabled for fixed comparisons. Dynamic schedules disable fused multistep draft execution and introduce another variable. There is no established request-level depth knob or supported hot-update API; depth changes require a controlled candidate reload after the active baseline sweep.

## Concrete settings

Preserve image/source precision,262144 max context, packed fp8_ds_mla, explicit target and draft B12X, requested block256, actual hybrid block size recorded, memory fraction0.93, chunk2048, native protected head flags0, vision budgets and all sampling settings. Initially keep C1/one slot. Do not combine the staged router patch with the first depth comparison.

| Fixed depth | Target verification length/request | C1 target graph sizes | C8 target graph sizes covering every active slot count |
|---:|---:|---|---|
|1|2|[2]|[2,4,6,8,10,12,14,16]|
|2|3|[3]|[3,6,9,12,15,18,21,24]|
|3|4|[4]|[4,8,12,16,20,24,28,32]|
|5|6|[6]|[6,12,18,24,30,36,42,48]|

For each point, set `speculative_config={"method":"mtp","model":"/mtp","num_speculative_tokens":D,"attention_backend":"B12X"}`. Use FULL_DECODE_ONLY/custom_ops all; capture `(D+1)*n` for every n1..slots and set max capture to `(D+1)*slots`. Draft decode itself processes one token per active request per extra step; its manager must capture the associated one-token request capacities. Verify actual target and draft descriptors, absence of unintended eager fallback, and the fused-multistep selection after load.

The earlier launch_controls helper supports1/3/5 only; the new immutable `mtp-depth-controls.json` includes depth2 explicitly. It does not edit that helper or launch any job.

## Resource budget and admission

Weights do not multiply with depth. Hidden-state scratch at fixed chunk2048 is2048×4096×2=16MiB; multimodal input-embedding scratch is another16MiB when enabled. These are chunk-dependent, not multiplied byD. The draft token buffer is8×slots×depth bytes. Optional probabilistic-draft logits scale as slots×depth×154880×dtype-size; record the actual configured draft sampling method and dtype rather than assuming they exist. FP32 vocabulary scratch per target verification token is619520B, so C8/depth5 could need about28.36MiB for48 token rows if fully materialized. Graph/private-pool workspaces and attention/rejection state may dominate these small buffers and must be measured.

Depth adds lookahead reservations. Per-request page rounding can exceed the raw extra-token bytes, particularly with the observed hybrid block8704. Preserve enough free memory for graph capture and long-context lookahead; do not claim the depth1 KV capacity remains unchanged. Record model bytes, graph bytes, peak activation bytes, actual KV blocks/tokens and process/host memory after each launch. The current B12x depth1 receipt (97.89GiB model,0.13GiB graph,751007 aggregate KV at262144 configured context) is a measured baseline only. It supersedes the older legacy460208-token receipt.

Do not raise memory fraction, reduce max context or shrink vision during the depth sweep. Reject an admission regression that breaks262144 per-request context. At higher concurrency, report actual simultaneous residency: eight slots do not imply eight full262144-token sequences fit the aggregate KV pool.

## Matched success and speed protocol

After the root's seven-context baseline completes, qualify depths in order2,3,5 against unchanged depth1. Use the existing accepted structured-output task and a separately frozen prose task. Keep structured and prose results in separate tables; freeze prompt token IDs, generation settings, expected-success verifier and output limits before the sweep. A fast request that loops, truncates, emits malformed structured output, leaks reasoning contrary to the requested mode or fails its task is a failed cell, not a speed win.

Start with1024 and8192-token prompts, C1, two unreported warmups and at least three measured successful repeats per task/depth. Count exact emitted token IDs from reasoning plus content and preserve normal stop reason and verified output; do not count SSE chunks. Report TTFT, prefill timing, request/aggregate decode tok/s, total decode tokens, successful request counts and latency distribution. Keep cached-prefix policy and input lengths identical. Expand only passing useful depths to the baseline's seven context sizes and then C2/C4/C8 after each slot/KV admission gate. Preserve failed cells and stop reasons alongside successful results.

Capture isolated before/after speculative counters scoped to served model and engine: number of drafts, drafted tokens, accepted tokens and accepted tokens by draft position. Preserve the position label; the current acceptance collector sums equal metric names and therefore loses per-position detail. Add a separate future collector/snapshot rather than altering the running frozen harness. Compute accepted fraction=accepted/drafted and mean accepted output length=1+accepted/number_of_drafts, matching upstream semantics. Require advancing counters, no resets and no unrelated requests in the interval. EOS and limited remaining output can shorten actual draft counts; do not demand every iteration draft exactlyD tokens.

For each depth, separately capture a short shape-recorded/stack-recorded mapping trace and a low-overhead formal trace at matched context/task. Attribute target verification, draft0, subsequent draft passes, LM heads and rejection. The useful comparison is accepted output tokens divided by total target+draft iteration time; a higher acceptance percentage alone is insufficient. A simple planning expression is `(1+E[accepted drafts])/(Ttarget(D+1)+Tdraft(D))`, populated with measurements, not assumed independent acceptance probabilities.

Finally validate long-context page boundaries, partial acceptance, EOS, transient active slot counts and fresh image/video output with MTP counters. Preserve external multimodal embeddings and graph lifetime behavior. No running server was restarted and no GPU job was launched for this plan.
