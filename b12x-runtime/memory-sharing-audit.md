# R3 target/native-MTP storage sharing audit

Read-only audit of installed source in the running Spark2384 container. No runtime attachment, GPU allocator queries, inference requests, restart, or patch was performed. `memory-sharing-source-snapshot.json` preserves exact installed files and SHA256 hashes. Runtime identity is bound by `runtime-gates-r3.json`.

## Large protected tables already share

The installed V2 `load_eagle_model` in `v1/worker/gpu/spec_decode/eagle/utils.py` resolves the target's language model, then:

- Lines99–117 replace the draft embedding module with the target embedding at pipeline-parallel size1. GLM MTP has no `has_own_embed_tokens` flag, so `_should_share` returns true.
- Lines119–137 replace the draft top-level head and every `layer.shared_head.head` with the target head. With the R3 `VLLM_MTP_NVFP4_LM_HEAD=0` setting, `Glm5NextMTP.has_own_lm_head` is false and sharing occurs. The actual forward computes logits through that same per-layer head.
- `Glm5NextForConditionalGeneration` marks `language_model` through `_mark_language_model`; `SupportsMultiModal.get_language_model()` retrieves it. The target language model owns both `model.embed_tokens` and `lm_head`.
- The separate draft-head-copy switch `VLLM_GLM53_MTP_DRAFT_HEAD` is absent from the container environment and defaults to `bf16`; its factory returns `None`. Therefore it does not create another vocabulary projection.

The original source headers confirm each table is BF16 with shape[154880,4096]:

| Table | Bytes | GiB | Additional reclaimable bytes from repeating alias |
|---|---:|---:|---:|
| Embedding | 1,268,776,960 | 1.181640625 | 0 |
| LM head | 1,268,776,960 | 1.181640625 | 0 |
| Both | 2,537,553,920 | 2.36328125 | 0 |

These are the executed source path's guaranteed module rebindings under the recorded configuration. The server's live tensor storage pointers were not independently inspected. A later instrumented fresh launch could record `is` and storage-pointer equality without changing arithmetic; no such live receipt is claimed here.

The standalone draft probe deliberately passed a dummy target with no embedding or head. Its9.55 GiB therefore includes both tables. It cannot be added directly to the full server's model allocation to infer duplication. Subtracting these tables gives about7.19 GiB as a rough incremental draft allocation; this is not a separately measured whole-server allocation delta.

## Smaller indexing buffers remain separate

`Glm5NextModel.__init__` creates target top-k buffers as local variables and passes them to target layers; it does not assign `self.topk_indices_buffer`. The generic V2 sharing guard `hasattr(target_inner,"topk_indices_buffer")` consequently does not find them. Each MTP layer separately allocates a token-selection and pooled-selection buffer in `mtp.py`.

| Buffer at max_num_batched_tokens2048 | Shape/type | Bytes |
|---|---|---:|
| Token-selection IDs | [2048,2051],INT32 | 16,801,792 |
| Pooled-selection IDs | [2048,512],INT32 | 4,194,304 |
| One duplicate MTP set | | 20,996,096 (20.0234 MiB) |

This is an upper bound for a straightforward scratch-sharing investigation, not an approved patch or measured saving. `set_skip_topk` explicitly reuses MTP selections across speculative iterations, so rebinding these buffers without checking scheduling/lifetime could corrupt draft state. The generic sharing loop also matches only `topk_indices_buffer`; pooled IDs require a separate, shape-correct binding. Saving around20 MiB does not buy another0.84375 GiB K3 routed layer.

The pooled indexer also owns per-layer FP8 query scratch, score arrays, block tables, tail state and tail snapshots. Some scratch has equal shapes, but tail/index/cache state is layer-specific; equality of shapes does not make it duplicate data. Its `_index_cache` is already an `as_strided` view into the packed MLA cache, not a separately allocated redundant index table. No other obviously duplicate native weight table was found in the inspected target/MTP path.

## Decision

Do not patch embedding/head sharing: it is already present. Preserve the current runtime. Treat scratch reuse as a separate small optimization requiring an instrumented fresh-run lifetime and parity test; it is not a solution to the higher-bit capacity requirement.
