# Fixed MTP depth sweep — staged, not run

This freezes one guarded launch at a time for depths **1, 2, 3, 5** and slot limits **1, 2, 4, 8**. It reuses `../start-b12x-candidate.sh` and the existing `../../acceptance/{run,timing,native_metrics}.py`; their hashes are pinned in `dependencies.json`. It never stops an existing server, edits an image, or changes frozen baseline files. The existing launcher's occupied-GPU and duplicate-container guards remain intact. No depth>1 runtime or speed improvement is claimed by these tools.

All variants retain 262144 per-request context, chunk2048, memory fraction0.93, BF16 protected heads, target and draft B12X, packed FP8 MLA KV, requested block256, prefix caching disabled and the original vision configuration. Native MTP expert-only FP8 remains enabled. The actual hybrid block size, KV token capacity, model bytes, peak activation and graph memory must be collected after **each** launch; neither C8 nor a large aggregate KV pool establishes eight simultaneous full-context requests.

## Generate and validate without launching

Run from the release root. This is the safe default; generating a plan performs no Docker calls.

```sh
python3 -m unittest discover -s b12x-runtime/mtp-depth-sweep/tests -v
python3 b12x-runtime/mtp-depth-sweep/source_contract.py
python3 b12x-runtime/mtp-depth-sweep/launch.py --depth 2 --slots 1 --output /tmp/glm-depth-d2-c1-plan
```

All 16 controls are CPU tested. `source_contract.py` executes the pure graph-selection helpers extracted from the **audited installed source snapshot**, not reimplementations of those selectors. Target graphs cover every active request count. Jovian's draft-prefill graphs intentionally use power-of-two request capacities and pad intermediate counts. Depth>1 additionally requires captured one-token draft-decode coverage. Capture completion is not proof that every live batch actually selects a graph: actual eager fallbacks and fused-multistep dispatch remain trace/functional acceptance checks.

## Controlled launch after the current baseline is released

Use a fresh directory and immutable image ID/digest on the intended Spark, with the existing verified model and native MTP view already present. Keep the previous container/image/source available for rollback. Do not execute this while another owner is using the GPU.

```sh
python3 b12x-runtime/mtp-depth-sweep/launch.py \
  --depth 2 --slots 1 --output /absolute/new-experiment-d2-c1 \
  --image sha256:VERIFIED_64_HEX_IMAGE_ID \
  --model-root /absolute/verified-model --mtp-root /absolute/native-mtp-view \
  --container glm53-depth-d2-c1-attempt1 --execute
```

Execution checks the exact installed depth/capture/MTP/B12X source hashes inside a bounded CPU-only container before launch. It requires image plugin/PYTHONPATH settings that it can preserve without ambiguity. It records model/index/native metadata hashes; complete weight-file integrity remains the prerequisite supplied by the existing verified artifact, not a claim inferred from those metadata hashes.

A small **vLLM general plugin** records actual target/draft descriptors only after successful graph capture. It changes no kernels or model arithmetic and is mounted read-only in the experiment. It ignores memory-estimator sample captures for admission. The plugin's source-level hook tests pass; actual worker plugin discovery/capture receipts must pass in the new candidate. If the plugin fails to load or misses a manager, admission fails rather than assuming graph success.

After startup finishes:

```sh
python3 b12x-runtime/mtp-depth-sweep/admission.py \
  --run /absolute/new-experiment-d2-c1 --endpoint http://127.0.0.1:18080
```

Admission requires the exact running image, command, native precision flags, read-only mounts, actual graph descriptors, model/activation/graph/KV resource logs, at least262144 aggregate KV tokens and a fresh `/v1/models` identity. Evidence files and source metadata are hashed. Every benchmark snapshot rechecks the container start and process identities, catching worker restarts even when the Docker container survives. Admission does **not** declare image/video quality, long-context correctness, speed or full-model acceptance.

## Matched structured and prose requests

Freeze a task JSON separately for each output class before measuring. Use the same task, prompt seed, matrix order, tokenizer revision, sampling settings and output limits at every depth. The wrapper preserves the original filler/tokenization/streaming/timing helpers; deterministic per-length nonces replace random run nonces so matched sweeps reproduce input token IDs. Every request records its exact prompt token IDs and output token IDs, including reasoning. Prefix caching stays disabled. A changed matrix order changes prompt ordinals and must not be called an exact matched comparison without comparing recorded IDs.

```json
{"name":"structured-answer","content_class":"structured","prompt":"Return only the JSON object {\"answer\":42}, with no markdown.","validator":{"type":"json_equals","expected":{"answer":42}}}
```

The example is a short semantic smoke, **not** a sustained speed workload. For prose use `content_class:"prose"` and `validator:{"type":"exact_text","expected":"..."}` with a frozen copy task long enough to measure. This is an exact-output functional check, not open-ended prose quality. Long requests must stop naturally and satisfy their declared validator; reaching `max_tokens`, malformed JSON or incorrect output fails the cell. Structured and prose runs remain separate.

```sh
python3 b12x-runtime/mtp-depth-sweep/measure.py \
  --run /absolute/new-experiment-d2-c1 --task-json /absolute/frozen-task.json \
  --prompt-seed glm53-depth-v1 -- \
  --base-url http://127.0.0.1:18080/v1 --metrics-url http://127.0.0.1:18080/metrics \
  --tokenizer /absolute/verified-model --lengths 1024,8192 --concurrencies 1 \
  --repeats 3 --output-reserve 2048 --template-json '{"enable_thinking":true}' \
  --output /absolute/new-experiment-d2-c1/structured-measured
```

Use two separately retained warmups first; exclude them from reported measured rows. The reused harness reports total matched decode tok/s, mean/request matched decode, request-native prefill, TTFT and end-to-end rates. Its default matched window is30seconds; a shorter successful smoke remains `UNAVAILABLE` for that window, never an invented throughput value. Only capacity-admitted cells execute. Counters do not prove successful output by themselves.

## Position-preserving counters

`counters.py snapshot` and `diff` can also run independently. Snapshots preserve every engine and position label and exact model identity, rather than summing all positions into one counter. Deltas reject changed processes, changed label sets, missing positions, counter decreases, changed creation timestamps, invalid numbers, inconsistent accepted-prefix accounting and non-advancing drafts. `_created` is optional because multiprocess Prometheus can omit it; process identity and counter monotonicity still apply. An externally reset counter that recovers above its old value without any observable identity/creation change cannot be proven from two snapshots.

Each final benchmark cell retains both snapshots and per-engine position deltas. Accepted fraction is accepted/drafted tokens; mean acceptance length is `1 + accepted/draft_iterations`, matching upstream semantics. EOS/remaining-output limits may shorten draft batches, so drafted tokens need not equal configured depth times iterations. Native request-counter matching and idle-before/after checks remain required to reject unrelated traffic. A position-accounting failure erases that cell's speed summary and stops the sweep.

The launcher does not blindly run all16 configurations. Start D1/D2/D3/D5 at C1 with identical tasks, retain useful passing depths, and qualify C2/C4/C8 with fresh graph/resource evidence. Then expand prompt sizes and run image/video, long-context, partial-acceptance, EOS and transient-slot graph checks. Shape-recorded attribution and low-overhead timing traces remain separate experiments. Preserve every failed admission/output and do not reduce context or vision budgets to turn failures into passes.
