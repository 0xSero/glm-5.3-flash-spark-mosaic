# Mixed K2/K3 feasibility and independent evidence

Read-only audit, 2026-09-11. No new quantization, GPU allocation, runtime patch, or source mutation occurred during this audit. This directory is private operational evidence.

## Completed quality result

Q3 keep192 completed all45 trunk layers and the matched65,504-position comparison, exited0, and passed both final-result seals. It measured61.809966% top1 agreement, KL1.1836369571 and perplexity9.570748215. This is substantially worse than unpruned Q3 (87.383977%, KL0.1522043987, perplexity3.496913656). Keep192 is not a suitable higher-agreement release. Its exact report is in `../../staging-q3-k192/candidate-quality-report.json`; report SHA256 `6e59b342cab221120aab0ce32b85fd9a5e1464e0c2a48ed108624aaaca5c91af`. All32 final normalized rows are verified. Raw rotating receipts0–2 were unavailable and disclosed; receipts3–44 were retained.

## Exact incremental cost

Headers from the completed K2 and K3 artifacts were inspected for all48 packed fields of one logical expert (3 projections ×4 archive ranks ×4 fields). Their shapes match except trellis K; source manifest totals independently agree with the derived cost across all42×288 experts. `header-byte-audit.json` records the actual header evidence.

Upgrading one complete logical expert from K2 to K3 adds exactly3,145,728 tensor bytes (3 MiB). Upgrading a complete288-expert layer adds905,969,664 bytes (0.84375 GiB). Native attention, router, shared experts, embedding/head, vision and MTP are unchanged. These increments do not include allocation/workspace changes.

| Full layers upgraded | Routed expert average bpw | Additional tensor GiB | Payload GiB including original native MTP/protected tensors |
|---:|---:|---:|---:|
|0|2.00000|0.00000|103.63273|
|1|2.02381|0.84375|104.47648|
|2|2.04762|1.68750|105.32023|
|3|2.07143|2.53125|106.16398|
|4|2.09524|3.37500|107.00773|
|8|2.19048|6.75000|110.38273|
|21|2.50000|17.71875|121.35148|
|42|3.00000|35.43750|139.07023|

`byte-frontier.json` includes exact tensor and source-file byte increments. For the first two proxy-ranked layers, files add1,811,943,584 bytes; tensors add1,811,939,328 bytes. Metadata headers account for the difference. Original native-MTP payload accounting must not be confused with the smaller live FP8-MTP allocation.

The sealed legacy full-K2 + FP8-MTP runtime reports97.11 GiB model loading,460,208 KV capacity,4.22 GiB KV,262,144 configured context,0.14 GiB actual graph pool, and5.59 GiB profiled activation peak. Its vision weights are configured, but those figures do not prove full-context vision acceptance. Proportional KV math suggests roughly1.82 GiB above the262k floor, enough for approximately two whole-layer increments with little margin. That is only a planning estimate: the hybrid attention cache has multiple components, GiB logs are rounded, B12x may change workspace, and no mixed model has passed admission. Start with one layer, then test two only after real262k + vision + MTP admission. There is no measured B12x KV receipt in this audit.

## Source compatibility and availability

The existing pinned K2 source remains `0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw` at `35b4b580debe2a1510a2eb042d9d6d4cd7fb3a6b`; its protected49 shards match the older pinned Q3 source exactly. All logical expert IDs, tensor names, rank0–3 decomposition, hidden4096/intermediate512 archive slices and MCG field layouts are compatible.

For allocation evidence, the important newly confirmed source is the **paired MIT-lineage K3**, not the older public uniform-Q3 control. It is present at `/home/sero/glm53-tr3/release-tr3-3bpw` on Omarchy:84 weight parts,115,567,478,208 file bytes. All84 observed sizes and declared hashes match the source references embedded in the paired proxy records; status `ENCODE_COMPLETE`,42 routed layers,288 experts,600 calibration rows. Encoded-status SHA256 `4a4402ab7b47f91145db841ce0105b54c3f7fcc3663a110fa712ceb85d5d1f0f`; historical independent-audit manifest is present. This audit checked availability and metadata binding, not a new full115GB SHA pass. Any selected parts must be fully hashed before building.

The K2 paired source hashes match the pinned K2 manifest. The paired K3 should be used if selecting from these proxy records. Do not apply the paired proxy scores as if they were measured for the older public Q3 encodings. Both use the same sealed BF16 model and protected head, but quantization provenance and tensors differ. The paired uniform K3 has no new end-to-end quality result from this campaign.

## Calibration evidence and allocation limits

`paired-proxy-inventory.json` binds84 local sidecars covering all12,096 logical experts. Each record has K2/K3 proxy error sums over12 projection/rank slices, measured error reduction and natural-route Hessian coverage. The original source calibration metadata was fetched from the exact pinned K2 repository and its existing declared SHA256 hashes were checked:

- Manifest `77393cc1862c069d1d770887b8c83f0ba5aab4b8dfdd7cb18447dbb621c2c80e`.
- Token file `479581ca7bebb14bc4b41d375cd74e66cab40a58a0662efd297c41566543bbad`.
- Verification `291250dea453739f1f1a9f0ab69d6a3c49c4dcc459b580eb850449b1291249c9`.
-600×2048 tokens, natural top8, no forced expert activation. Final mix:90 C4,156 code,25 multilingual,92 random,23 technical,18 tiny,132 wiki, plus64 private-session rows across code/agentic, inference infrastructure, writing, and research/reasoning.

Fresh raw token-row hashing on Omarchy verified the exact calibration/evaluation file identities and zero equal2048-token rows between the600 calibration rows and32 frozen WikiText test rows. This is exact-row exclusion only; it does not prove semantic or substring deduplication. No held-out WikiText outcomes were used for the proxy ranking.

A simple sum of independent calibration K2-minus-K3 proxy reductions ranks layers5,32,6,37,36,31,7,35 first. This is **not measured end-to-end layer sensitivity**. Proxy scales, error propagation and route frequency can change the preferred global allocation, and the historical builder only compared experts within each layer. Before freezing a multi-layer allocation, a bounded one-layer substitution pilot on separate calibration rows should compare activation/logit residuals for candidate layers; do not optimize on the WikiText regression panel. Upgrading only two layers increases average bpw to2.04762; a large agreement increase cannot be promised.

## Runtime implementation boundary

Current `b12x-runtime/exl3-port/exl3.py` assigns global `quant_config.bits` to each MoE method. Packed tensors allocate `k_words = bits *16`, and the fused call uses one `layer._exl3_k`. Therefore merely mixing files or changing a global bits label will fail shape checks or be incorrect.

The smallest extension is a strict per-layer bits map resolved from the MoE prefix in `get_quant_method`. Each method gets its own immutable K2/K3 setting before allocation; the existing fused kernel continues to see uniform K within that layer. Preserve the global default and fail on malformed/unused/non-routed overrides. Keep the separate FP8-MTP draft precision contract untouched. CPU tests should cover override isolation, name resolution, source-shape mismatches and layer45 exclusion; actual K2/K3 fused loading and graph replay still need GPU acceptance.

A no-requantization mixed artifact can reference immutable selected K2/K3 shards under distinct directory prefixes, rebuild the weight index and record a complete layer-bits ledger. Keep all288 expert IDs and router rows unchanged. Hash every selected source part and protected closure. The full model then needs quality, actual API parity and262k/vision/MTP admission before selection.

Per-expert mixed K within one layer is not currently supported by the fused path. It would need separate packed buckets, route-to-bucket mapping, multiple K-specialized launches and correctly accumulated down-projection outputs, with graph-safe scratch. It could spend the same576-expert budget across more layers, but is a larger correctness and performance change. Do not implement it as a ragged tensor padding shortcut.

## Numerical seam audit

The frozen workers explicitly disable CUDA matmul TF32 (`evaluate_low_bpw_quality.py:235`, `release_route_capture_v2.py:305`). Native parameter copying preserves BF16 versus FP32 source dtype, and the candidate adapter reconstructs EXL3 weights to BF16 before the same transformer path. There is no demonstrated TF32-caused quality loss in these runs.

The private serving-reference preparation independently reproduced all32 BF16 teacher NLL sums exactly using the same2047-position BF16 head GEMM and4096-vocabulary chunk geometry; max delta0.0. That confirms the head/position arithmetic but does not revalidate the historical teacher trunk. Real serving still differs from offline reconstruction at EXL3 fused FP16 intermediates, attention implementation/FP8 KV and graph behavior. Engine agreement at identical weights must remain separate from quantization agreement to BF16. No runtime flag or dtype change is justified from the external llama.cpp anecdote alone.
