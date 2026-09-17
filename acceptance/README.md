# GLM single-Spark inference acceptance

These scripts implement the acceptance procedure; dated result directories establish what has actually run. They use an isolated
OpenAI-compatible `/v1/chat/completions` API, the matching vLLM `/tokenize` endpoint,
and Prometheus `/metrics`. Python standard library is sufficient except prompt
preparation, which needs the pinned Transformers/tokenizer environment.

The methods follow Local Inference Lab's matched-window and reporting skills.
`native_metrics.py`, `behavior_acceptance.py`, `vision_acceptance.py`, and the six small paired fixtures are
reused from the existing `spark-model-releases-20260905/glm-5.3-flash-dgx-spark`
release. Inspect that release's source/license attribution before publication.

## Runtime evidence before requests

The runtime operator must create a normalized JSON receipt from actual config,
container identity, completed CUDA graph capture, readiness, and allocated KV logs:

```json
{
  "model": "THE_ACTUAL_SERVED_ID",
  "context_limit": 262144,
  "max_num_seqs": 1,
  "kv_capacity_tokens": 262144,
  "prefix_caching": false,
  "cuda_graphs": true,
  "speculative_config": {"method": "mtp", "num_speculative_tokens": 1},
  "evidence_sha256": {"container_inspect": "64 lowercase hex characters", "frozen_engine_log": "64 lowercase hex characters"}
}
```

Values above illustrate the schema, not measurements. Record the actual allocated
KV capacity and slots; never set them to convenient targets. Hash frozen snapshots
rather than a log that continues changing. The harness checks these fields against
CLI settings and binds the receipt SHA, but the operator must audit the underlying
evidence and exclusive ownership. Do not run against Pop's interactive service.

## Matrix

Prepare the requested grid without any network calls or tokenizer load:

```bash
python3 run.py --plan-only --model THE_ACTUAL_SERVED_ID \
  --max-num-seqs 1 --kv-capacity-tokens 262144 --output planned-grid
```

Execute only once the accepted runtime is reserved and idle:

```bash
python3 run.py --base-url http://127.0.0.1:18080/v1 \
  --metrics-url http://127.0.0.1:18080/metrics \
  --model THE_ACTUAL_SERVED_ID --tokenizer /models/candidate \
  --runtime-receipt runtime-receipt.json \
  --max-num-seqs ACTUAL_SLOTS --kv-capacity-tokens ACTUAL_KV_TOKENS \
  --output matrix-run-001
```

Default prompt targets:1024,4096,16384,65536,131072,200000,260096 chat tokens.
Default concurrency1,2,4,8; three repeats. The near-limit target is
**262144 total context minus2048 reserved output tokens**, including the chat
template. Actual server tokenization must match local counting before generation.
Actual usage must remain within the limit. A262144-token input plus extra output
is never treated as an admissible262144-context request.
Prompt preparation allows up to four tokens below a requested target; tables
report each actual request length, and all rate calculations use actual usage.

Capacity checks conservatively reserve `C × (prompt + output reserve)` KV tokens.
Unsupported active-slot, total-KV, or request-context cells remain explicit rows.
These cells are not zero-speed results and queued incoming requests are not
mislabelled active concurrent streams. C1 executes before higher concurrency.
A runtime/token-accounting failure stops additional load; remaining planned cells
become `NOT_RUN_AFTER_FAILURE`.

For a first bounded screen use `--lengths 1024 --concurrencies 1 --repeats 2`.
Requests use temperature0, thinking enabled, one structured content class, and
2048 maximum output tokens. `finish_reason=length` is valid for a bounded speed
screen but is not semantic acceptance. No quality claim comes from these prompts.

### Table meanings

`TABLE.md` contains distinct columns for:

- **TOTAL matched decode tok/s** and mean per-request matched decode tok/s.
- Shared decode-window duration; under30 seconds remains `MATCHED_SCREEN`.
- Server request-prefill tokens/s, from matching isolated Prometheus request
  durations and token totals. Concurrent duration sums overlap; this is not total
  GPU-exclusive prefill throughput.
- Effective prompt-tokens/TTFT, which includes queue/tokenization/network effects.
- TTFT p50/p90, a separately labelled client decode estimate, and total output
  tokens per complete end-to-end wall second.

The matched window is `(latest first token arrival, earliest last token arrival]`.
Exact `choices[i].token_ids` from `return_token_ids:true` are counted on their
arrival timestamps. Every speculative bundle remains intact. Every stream must
emit in the common interval, with no gap exceeding five seconds by default.
Missing IDs, inconsistent emitted-ID/usage totals, silent streams, or no common
interval yield **unavailable**, never retokenized or SSE-chunk-derived tokens/s.
Raw client timing remains explicitly an estimate when exact matching is unavailable.
Native histogram deltas are allowed15 seconds to become visible; contamination or
unmatched counts invalidate accepted matched rates and stop the sweep.

Each run saves source hashes, settings, raw per-request token IDs/arrival events,
actual usage, individual cells, complete matrix JSON, and its table. URLs and API
keys are not put in run metadata; use `MODEL_API_KEY` if needed. Raw server errors
can contain private details and must be reviewed before public release.

## Near-limit semantic acceptance

```bash
python3 long_context.py --base-url http://127.0.0.1:18080/v1 \
  --metrics-url http://127.0.0.1:18080/metrics \
  --model THE_ACTUAL_SERVED_ID --tokenizer /models/candidate \
  --runtime-receipt runtime-receipt.json --output near-limit-001
```

Four random registry records are positioned at approximately5%,35%,65%,95% of
context. The model must return all values exactly in naturally terminated JSON.
The script preserves prompt, marker positions, expected answer, tokenizer counts,
server counts, usage and output. This establishes the scoped retrieval/capacity
case, not broad long-context reasoning or a full context-length quality benchmark.

## Image and native video semantics

```bash
python3 vision_acceptance.py --base-url http://127.0.0.1:18080/v1 \
  --model THE_ACTUAL_SERVED_ID --kind all \
  --template-json '{"enable_thinking":true}' --output vision-001.json
```

Four images test changing OCR strings, object counts and colors; two native MP4s
test direction and beginning/end colors. Fixture SHA checks fail closed. Expected
answer objects are withheld from requests. Repeated success proves these paired
semantic cases; it is not a claim that maximal video duration/resolution works.
Run this while the same MTP-enabled full-context configuration remains resident.

## Native MTP execution

Short text, parsed tool calls, Arabic, Chinese and Polish exact natural-stop
checks can be replayed with `python3 behavior_acceptance.py --endpoint
http://127.0.0.1:18080 --model glm-5.3-flash --output behavior-001.json`.
The tool case requests a synthetic weather function and never executes it.
Thinking is explicitly enabled; these checks do not measure broad coding quality.

```bash
python3 mtp_proof.py --runtime-receipt runtime-receipt.json \
  --cell matrix-run-001/cell-001.json --output mtp-001.json
```

Pass requires native `method=mtp`, positive speculative depth, a completed cell
with matched isolated request counters, and positive advancing drafted/accepted
token counts with valid acceptance fraction. A configured flag alone fails.
A counter reset, absent metric, different speculative algorithm, or zero accepted
tokens does not establish native MTP operation. The proof does not claim an MTP
speedup: that requires an otherwise matched on/off benchmark.

## CPU tests

```bash
python3 -m unittest discover -s tests -v
```

Tests cover bundle-aware matched arithmetic, output-reserved context/KV bounds,
missing-ID and silent-window rejection, near-limit semantic scoring, MTP evidence,
fixture integrity, and the network-free plan CLI. No model endpoint has been called
by these tests.

## Verified chat-template behavior

The pinned Q3 and K2 templates always open the assistant response with `<think>`; neither implements an `enable_thinking=false` template branch. Passing that flag nevertheless disables the glm47 reasoning parser, which exposes genuine reasoning as ordinary content. The acceptance defaults therefore use `enable_thinking=true` and count both reasoning and final-answer tokens in throughput. Strict semantic checks inspect the separate final content. A changed template or thinking-off mode requires separate validation.
