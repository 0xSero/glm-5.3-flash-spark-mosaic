# Native MTP experiment controls

Scope: CPU source audit after the successful full original K2 target + routed-expert FP8 MTP R4 serving gate. No active runtime changed by this audit. All new capacity/speed points below remain untested.

## Independent fixed-depth comparisons

Use `launch_controls.py --depth {1,3,5} --slots {1,2,4,8} --chunk {2048,4096,8192}` to emit JSON controls for the final B12x launcher. The existing legacy launcher is deliberately unchanged at depth 1, slot 1, chunk 2048.

| Draft depth | Query tokens per request | Maximum graph tokens, C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 4 | 8 | 16 |
| 3 | 4 | 4 | 8 | 16 | 32 |
| 5 | 6 | 6 | 12 | 24 | 48 |

For fixed depth `d` and active slot limit `s`, capture sizes are `(d+1)*n` for **every** `n=1..s`, not only the sweep's power-of-two request counts. This covers transient batches while requests finish. Set `max_cudagraph_capture_size=(d+1)*s`, `max_num_seqs=s`, and `max_num_batched_tokens=chunk`. Keep `max_model_len=262144`, FP8 draft expert opt-in, native nonexperts and the same multimodal limits across matched tests.

V2's actual `ModelState.num_new_sampled_tokens_per_step` defaults to 1 (`model_states/interface.py:265`), and `model_runner.py` sets `decode_query_len=num_speculative_steps+num_new_sampled_tokens_per_step`. Jovian `cudagraph_utils.py:345–451` computes `max_decode_tokens=max_num_reqs*decode_query_len`, rounds capture descriptors by their query lengths, and rejects descriptors exceeding max requests or max capture tokens. The current legacy runner has the same fixed-depth query-length contract (`model_runner.py:385–388`). The configured target has one MTP layer; depth 3/5 autoregressively reuses it, rather than requiring three/five checkpoint MTP layers.

These are decoder token capture sizes, **not** the prefill chunk size. Increasing a chunk raises activation workspace and can reduce admitted KV capacity. Increasing slots raises request state/graphs/profile costs. A requested 262144 per-request context does not imply `slots*262144` aggregate KV; record actual KV admission after each launch. More incoming requests than `max_num_seqs` demonstrate queuing, not active concurrency.

## Dynamic draft depth: distinguish current and new V2

There is no audited request-level native MTP depth override in sampling parameters, and no hot-update API has been established. Do not invent one or mutate scheduler internals during serving.

**Legacy vLLM487ecf187:** config and scheduler expose `num_speculative_tokens_per_batch_size`, and graph code mentions dynamic SD. However, actual V2 `model_runner.py` never consumes `num_spec_tokens_to_schedule`; `AutoRegressiveSpeculator.propose()` always loops over `self.num_speculative_steps`. This is an incomplete production dispatch path. Do not use it for a dynamic-depth performance claim.

**Pinned Jovian3aada677:** the source path is complete enough to qualify on GPU. Scheduler selects `num_spec_tokens_to_schedule` from scheduled batch size (`scheduler.py:1916–1925`), V2 passes it into `speculator.propose()` and slices returned drafts (`model_runner.py:2152–2194`), and the autoregressive proposer validates `1<=depth<=configured_max` then generates that depth (`autoregressive/speculator.py:261–419`). Graph manager captures the reachable query lengths (`cudagraph_utils.py:365–409`). Example startup config:

```json
{"method":"mtp","model":"/mtp","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,1,5],[2,2,3],[3,8,1]]}
```

This changes depth automatically as the **scheduled** batch changes (including mixed prefill/decode scheduling), without a model reload. It is an optimization policy to validate after fixed controls; it does not independently compare depths at the same concurrency. It also disables fused multi-step draft decode (`autoregressive/speculator.py:134–136`), so no speed advantage should be assumed. Jovian additionally supports acceptance-length adaptation via `adaptive_speculative_tokens_window`; leave that off during independent fixed-depth comparisons.

For dynamic max-depth5/C8, a conservative candidate graph list is the sorted union `{(d+1)*n : d in {1,3,5}, n in 1..8}`, with max capture48. Extra descriptors consume memory and require admission profiling. Static depth comparisons remain the clean reference.

CPU evidence: `test_launch_controls.py` passes three tests covering all36 fixed launch configurations, rejected out-of-scope controls, and **the actual upstream** dynamic schedule validation/lookup (no copied implementation, no CUDA allocation). These are configuration/source checks, not serving or speed results.

## Current measured point

R4 full unpruned K2 / FP8 draft experts at depth1/C1/chunk2048/memory.93: model97.11GiB, actual graph.14GiB, aggregate KV460208, configured context262144. Fresh text exact answers passed and native MTP counters advanced65 drafted/63 accepted. Raw logprobs API contract returned20 `token_id:N` alternatives despite `top_k=1`. Details/hashes in `runtime-receipt-k2full-fp8draft-r4.json`. Root owns subsequent benchmark and feature acceptance; this legacy runtime does not contain B12x and its draft does not consume external multimodal embeddings.

R4 FP8 draft log warns no tuned MoE config exists for `E=288,N=2048,device_name=NVIDIA_GB10,dtype=fp8_w8a8`. It uses TRITON defaults. This is a concrete later tuning opportunity, not a demonstrated speed gain.
