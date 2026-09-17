# Why the mosaic cannot use the MTP runtime today — codebook `mul1` vs `mcg`

Written 2026-09-17 UTC from live launches on spark-557f. Nothing was modified in any image or artifact.

## Symptom (receipted)

Launching `M288-12L` (artifact `557f:/home/valentine/models/mosaic-12l-gatea`) on the program's
MTP-capable runtime with MTP ON
(`mosaic-gatea/launch-mosaic-mtp.sh`, shipped sha256 `bbfbc38aecee80b69b1dc6a5fe92ac3d3bdf97b2c3e1fda2fd93a752bb4d9a28`,
log `557f:/home/valentine/mosaic-gatea/mtp-glm53-mosaic12l-mtp.log`) fails at config validation, before
any weight is read:

```
(APIServer pid=1) INFO [model.py:679] Resolved architecture: Glm5NextMTPModel
(APIServer pid=1) INFO [speculative.py:1256] Overriding draft model max model len from 1048576 to 262144
(APIServer pid=1) Traceback (most recent call last):
(APIServer pid=1) pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig
(APIServer pid=1)   Value error, this overlay only implements codebook=mcg; got 'mul1'
```

## Where the gate is (file:line, both MTP-capable images)

`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py`

```
$  sed -n '555,572p' .../exl3.py          # image local/glm53-reap-native-mtp:20260911-r4-expert-fp8
        codebook: str = "mcg",
        ...
        if self.codebook != "mcg":
            raise ValueError(
                f"this overlay only implements codebook=mcg; got {self.codebook!r}"
            )
$  grep -n 'codebook' .../exl3.py         # image glm53-b12x-exl3:jovian-3aada677-r3
555:        codebook: str = "mcg",
568:        if self.codebook != "mcg":
570:                f"this overlay only implements codebook=mcg; got {self.codebook!r}"
610:            codebook=str(config.get("codebook", "mcg")),
```

Both the native-MTP runtime (`glm53-reap-native-mtp:…-r4-expert-fp8`, id `4f06a26d872d`) and the B12x
runtime (`glm53-b12x-exl3:jovian-3aada677-r3`, id `afb74c791853`) carry the same single-codebook gate.

## Why this separates the two artifact lineages

| lineage | artifacts | codebook | MTP runtime? | quality evidence |
|---|---|---|---|---|
| turboderp EXL3 (the shipped line) | **a0** (85.1 GB), 3.05bpw base, **M288-12L/10L mosaic** | **`mul1`** | **no** (`ValueError`) | full G4 panel + MMLU + GPQA (all in this directory) |
| staging K2/K3 | `flash-experimental-staging-20260906/model` (103.70 GiB ckpt), K2-keep256, K3 | `mcg` | **yes** (R4 receipt: load 97.11 GiB, KV 460,208, target+draft graphs captured, MTP counters 65 drafted/63 accepted) | leaderboard rows, task benchmarks |

The whole mosaic program built on the turboderp/mul1 line because that is what a0 ships and what the
sealed 3.05bpw base is; the MTP program built on the mcg staging line because that is what the
MTP-capable vLLM overlay implements.

## What closing the gap would require (options, honest costs)

1. **Implement `mul1` in that overlay** — the gate above is the entry point, but the codebook is not a
   label: it selects the decode math (trellis codebook tables used by `LinearEXL3`). Doing it means
   porting the mul1 codebook path and numerically qualifying it against the exl3 reference decoder.
   Multi-day, GPU-qualified work; no shortcut found in the images.
2. **Re-quantize the mosaic to `mcg`** — needs a working exllamav3 quantization build; on Spark that
   build currently fails (`557f:/home/valentine/models/p3b-out/build.rc=1`, `fa-build.rc=1`).
3. **Accept the split**: mosaic = quality artifact on the SGLang/exl3-plain path (no MTP, ~9–10 tok/s);
   K2/K3 = MTP artifact on the vLLM path (~19 tok/s class). Two artifacts, two purposes, until (1) or (2) lands.

Interim measured fact from the SGLang path (this session): enabling decode CUDA graphs alone (no MTP)
on the smaller 10-layer mosaic lifted single-stream decode from ~9.0–9.4 tok/s to **11.24/11.41/11.52 tok/s**
(`557f` container `glm53-mosaic-10lg`, KV 718,400 at mem-fraction 0.95, graph capture 72.5 s / 3.27 GB).
## Resolution (2026-09-17, same session): option 3 stands, and it is now measured on both sides

The mcg lineage was taken all the way to a served, receipted, non-greedy measurement on the program's own
D2/B12x recipe — so the split is a boundary with numbers on both sides, not a hypothesis:

| side | runtime / artifact | MTP | CUDA graphs | ctx | measured decode |
|---|---|---|---|---|---|
| **mcg** | `glm53-b12x-exl3:jovian-3aada677-r3` + K2 target + native draft (depth 2, B12X attn, `fp8_ds_mla`, block 256) | **yes**, 98.3–99.6% acceptance | **yes** (`FULL_DECODE_ONLY`, target + draft) | 262,144 (KV 727,449 = 2.77×) | **19.23 / 19.00 total decode tok/s** (warmup 19.11), sampling on |
| **mul1** | `glm53-flash-sglang-exl3-plain:serve4` + M288-12L / 10L mosaic | no (`dropped:nextn_mtp_off 3508 tensors`) | no in the eval config (the 10L could capture at max-bs 1 → 11.24–11.52 tok/s) | 262,144 (KV 595,200 @0.95 / 718,400 @0.95 on 10L) | 9–10 tok/s; 11.2–11.5 with graphs |

Same-session cross-check on the *other* MTP-capable image (`local/glm53-reap-native-mtp:20260911-r4-expert-fp8`,
`FLASHINFER_MLA_SPARSE_SM120`, `fp8`, block 64, depth 1): MTP also served (counters +900 drafted / +633
accepted over three sampled generations, 70.3%) but the cells compute out to ≈13.3 tok/s — i.e. on this
hardware the B12X + `fp8_ds_mla` + depth-2 configuration is worth about +45% over the r4 configuration at
the same target. That gap is a runtime/flag gap, orthogonal to the codebook gate.

Full receipts: `RESULTS-CONFIRMING-RUN.md` §5, `receipts-557f/out/mtp-serve-proof-20260917T122332Z.json`,
`receipts-557f/mtp-glm53-k2-mtp.log` + `out/k2-mtp-final-inspect-20260917T120333Z.json`.
