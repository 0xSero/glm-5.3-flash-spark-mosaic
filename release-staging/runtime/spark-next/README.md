# Single-Spark B12x launch candidate

**Experimental baseline only; K2 promotion is on hold pending higher-agreement quality work. No accepted B12x image or speed result is claimed here.** Supply the final independently validated image by digest. This package preserves the full original K2 target and converts only the native MTP companion's routed experts to FP8 during loading. All target nonexperts and all MTP nonexperts stay in their native precision.

The checkpoint is pinned to [0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw/tree/35b4b580debe2a1510a2eb042d9d6d4cd7fb3a6b), commit `35b4b580debe2a1510a2eb042d9d6d4cd7fb3a6b`. It contains 133 weight files totaling 111,352,026,456 bytes. The serving ID is `glm-5.3-flash`.

## Launch

Requires one DGX Spark GB10, Docker with the NVIDIA Container Toolkit, and Bash. Python, Hugging Face download tools, the patched native MTP loader, EXL3, vLLM and B12x must already be inside the supplied image. No host Python ML packages are installed.

```bash
export GLM53_IMAGE='your-registry/validated-image@sha256:REPLACE_WITH_64_HEX_DIGEST'

# Download into an empty directory, or resume this exact pinned download.
./run.sh "$PWD/model" "$PWD/state" --download

# Alternatively, validate and use an existing self-contained model directory.
# Omit --download to mount existing weights read-only throughout setup.
./run.sh /path/to/model /path/to/separate-state
```

Use **one** launch command. Model and state directories must be separate and non-nested. Existing Hugging Face snapshot symlinks that point outside the supplied model directory are rejected; use a complete local-directory download. The public download needs no credentials.

`run.sh` checks for occupied GPUs before heavy setup and again before launch. It uses an ephemeral CPU-only setup container limited to 2 CPUs and 3 GiB RAM with no additional swap, then one persistent GPU container. Setup checks SHA256 and size for every weight, hashes all pinned metadata, checks all 583,090 tensor names against their shard headers/index, verifies payload offsets, and creates a separate 891-key native MTP view. It never edits source tensors. The persistent container mounts the model read-only. On automatic restart it rechecks model metadata hashes, recorded weight sizes/modification times, and every generated native MTP view file hash/symlink target. A changed draft view fails closed; use a fresh state directory to regenerate it. Manually rerunning setup performs the full weight hashes again.

Launch refuses an existing container of the same name or any occupied GPU. It never stops unrelated jobs. The default name is `glm53-spark-next`; override `CONTAINER_NAME` when intentionally creating a distinct experiment.

```bash
docker logs -f glm53-spark-next
docker inspect --format '{{.State.Health.Status}}' glm53-spark-next
```

Health becomes `healthy` only after the API answers a fresh math prompt correctly and native MTP drafted/accepted counters advance. The requested bind port is checked before the engine starts, so an existing unrelated API cannot satisfy startup readiness. Startup has a bounded 1,800-second engine-readiness window plus a 180-second text check. Metrics are then polled for up to 15 seconds to allow delayed export; finite deltas must satisfy `0 < accepted <= drafted`. `state/fresh-text.json` preserves the actual request, response and counters. `state/launch.json` records arguments. Container logs preserve loading, graph and KV admission. Default endpoint: `http://127.0.0.1:18080/v1`. Explicit `BIND_HOST=0.0.0.0` exposes the API on the host network.

The container uses `restart: unless-stopped` semantics for process exits. An unhealthy Docker status alone does not trigger a restart. A startup failure remains visible in logs; automatic restarts are not a guarantee of uninterrupted inference. `docker stop glm53-spark-next` stops this container and suppresses automatic restart.

## Controlled settings

| Variable | Initial value | Allowed experimental values |
|---|---:|---|
| `MTP_DEPTH` | 1 | 1, 3, 5 |
| `ACTIVE_SLOTS` | 1 | 1, 2, 4, 8 |
| `PREFILL_CHUNK` | 2048 | 2048, 4096, 8192 |
| `MEMORY_FRACTION` | 0.93 | Greater than 0, at most 0.95 |

Other settings are explicit: B12X attention for both target and draft, `fp8_ds_mla` KV with block size256, `kda_prefill_backend=b12x`, TP1, native MTP with only its routed experts FP8, no prefix caching, full-decode graphs, and 262,144 requested tokens per request. The source native context maximum is 1,048,576; this recipe targets the requested 262,144. Image/video processing configuration and native source weights are retained. Both `VLLM_MXFP8_LM_HEAD=0` and `VLLM_MTP_NVFP4_LM_HEAD=0` are forced at launch to prevent Jovian defaults from quantizing protected output heads. The Python API-server entrypoint receives `--model /model` explicitly. These constructor/precision requirements follow the strict native draft-load gate; full target B12x serving remains a separate acceptance. No profiler/debug routes are enabled by this package.

Capture sizes are `(MTP_DEPTH+1)*n` for every active request count `n=1..ACTIVE_SLOTS`. For depth5/slots8 the largest capture is 48 tokens. More slots and larger chunks require new memory admission and correctness/performance validation. All alternate settings are **untested**, and the new B12x initial point also requires its own acceptance. A queue of eight requests with one active slot does not prove C8 throughput. The actual aggregate KV capacity must come from startup logs; no multi-request full-context capacity is promised.

## Validation boundaries

Sixteen CPU tests cover hash failures, metadata corruption, extra shards, missing index targets, payload offsets, path containment, launch flags, delayed/invalid metrics, port occupancy, stale readiness, draft-view edits, and the ordering/resource caps of the shell wrapper using a fake Docker executable (no GPU calls). A separate 583,090-key/133-shard scale fixture uses the exact public index names with synthetic zero-byte tensor payloads to measure CPU process memory; it is not a real model-weight acceptance or Docker cgroup-memory measurement (`validation-scale-evidence.json`). Eight tokenizer/processor/template metadata files were also downloaded anonymously at the immutable public revision and matched expected hashes (`metadata-pin-verification.json`). These checks do not replace actual GPU acceptance with the supplied image.

Fresh short text is the launch gate. Full requested-context generation, actual image/video input, output-quality parity, concurrent throughput and sustained stability remain separate release gates. The release's evidence directory should contain those results before this candidate is promoted. The existing all-native-source checkpoint remains reusable independently of the optional on-load FP8 draft policy.
