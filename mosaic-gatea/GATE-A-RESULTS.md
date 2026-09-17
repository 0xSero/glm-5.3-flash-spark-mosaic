# GATE A RESULTS — mixed-precision mosaic (2026-09-16, interactive "let's test it")

Everything below is measured this session; receipts in `mosaic-gatea/` (Mac) and
`557f:/home/valentine/mosaic-gatea/` + `2384:/home/sero/w2port/out/`. Artifacts live on 557f only.

## 1. What was tested

Two mosaics of the sealed turboderp quants (byte-copy only, no requantization, no new quantization error):

| Variant | Structure | Artifact | Result |
|---|---|---|---|
| **M288-GateA** (per-expert) | a0 dressing + **top-86 of layer 20's** experts at 3 bits (47.30% of layer-20 activation-weighted mass) | 85,504,022,960 B | built+verified; **serve FAILED** (see §2) |
| **M288-12L-GateA2** (per-layer) | a0 dressing + **12 whole layers at 3 bits** (3, 32–33, 36–44; all 288 experts everywhere) | **96,105,137,024 B** | built+verified; **serves; beats a0 on the panel** (see §3–§4) |

## 2. The per-expert granularity is not representable in the served form (definitive)

Build/census were clean (pack: 1,032/1,032 upgraded tensors byte-equal to the 3.05bpw base, 12,960/12,960
others byte-equal to a0; census: `contract_ok true`, `moe_sparkinfer_native true`, bits `{2: 36894, 3: 258}`)
— then the loader failed:

```
/opt/exl3-plain-overlay/exl3_plain_sglang_overlay/moe.py:88
    torch.stack([param.payloads[(e, projection)] for e in range(EXPERTS)])
RuntimeError: stack expects each tensor to be equal size, but got [256, 128, 32] at entry 0 and [256, 128, 48] at entry 1
```

The overlay **fuses all experts of a layer into one stacked tensor** per (group, field), so every expert in a
layer must share one bitrate. Confirmed by source read: `bits = int(w13.shape[-1]) // 16`, and
`w13` (gate+up) must equal `w2` (down) — i.e. **the supported granularity is whole-layer**. Note the census's
`moe_sparkinfer_native` check does not catch this; only a serve attempt does. (Evidence: `build-gatea.log`,
`census-mosaic-gatea-stock.json`, `mosaic-gatea-pack-receipt.json`, container log quoted above.)

## 3. The per-layer mosaic: build, serving, gates

**Build verification** (`mosaic-12l-pack-receipt.json`, `build-12l.log`):
- 41,472/41,472 upgraded tensors byte-equal to the 3.05bpw base; 26,117/26,117 kept tensors byte-equal to a0
  (`n_failures: 0`; the receipt's `FAIL` verdict is only my exact-delta check: +10,706 B of header-offset
  drift across the 5 rewritten shards — the tensor-level checks are strictly stronger and all passed).
- Exactly 5/12 shards differ from a0 (the ones holding upgraded layers); `model.safetensors.index.json`
  byte-identical; `config.json` byte-identical; `quantization_config.json` +1 B (10,368 tensor_storage
  entries patched from the base's own config).
- Size 96,105,137,024 B — 488 MB inside the F216-proven serving envelope.

**Census** (`census-mosaic-12l-stock.json`): `contract_ok true`, `problems []`, bits `{2: 26784, 3: 10368}`,
layer 3 = all 3-bit, layer 20 = all 2-bit, layer 44 = all 3-bit, `moe_sparkinfer_native true`.

**Serving**:
- mf 0.90, `--disable-cuda-graph` (the family's eval config), max-running-requests 1 → READY, smoke
  `The capital of France is → " Paris. It is located in the north-central part of the country."`
  **KV 136,896 (boot 1) / 193,472 (boot 2) — below the 262,144 release gate.**
- CUDA graphs on (`--cuda-graph-max-bs 1`) → **fails**: `Hybrid (mamba/linear-attention) state cache is too
  small to serve a…`. Graphs are not usable for a 96 GB artifact on this stack at 0.90.
- **mf 0.95, no graphs → READY, `max_total_num_tokens = 595,200` ≥ 262,144 (release gate satisfied)**;
  memory: avail-after-load 13.84 GB, pool-end avail 6.49 GB, usage ~99 GB (`serve-12lh.out`).
- Throughput (128-token greedy, single stream): a0 **9.10/9.25/9.24 tok/s**; mosaic (mf0.90) **9.43/8.93/8.53**;
  mosaic (mf0.95) **7.01/10.40**. The mosaic is speed-neutral vs a0. The repo's recorded reference for this
  stack is ~14.8–15.1 tok/s total decode; both eval configs here run **without MTP** (`dropped:nextn_mtp_off`)
  and without CUDA graphs — the two known speed levers are therefore *not* exercised by either endpoint.

## 4. Quality: first artifact in the program to beat a0 on the G4 panel

Frozen rows 0–3 (8,188 positions), identical prompts/teacher for both endpoints, teacher reproduction exact:

| variant | top-1 | KL lower bound | PPL |
|---|---:|---:|---:|
| a0 (2.05bpw, all 288 experts) | 0.8096 | 0.4749 | 3.0162 |
| **mosaic-12L** | **0.8282** | **0.3957** | **2.8463** |
| delta | **+1.86 pp** | **−16.7%** | **−5.6%** |

Receipts: `a0-mini-panel.json`, `mosaic12l-mini-panel.json` (both 8,188 positions, `teacher_top1_exact=true`).
Caveat: 4 of 32 rows; the full panel (65,504 positions) + quick MMLU + GPQA MC are the confirming runs.

## 5. State and next

- **Live now**: `glm53-mosaic-12lh` serving `/home/valentine/models/mosaic-12l-gatea` on spark-557f
  (mf 0.95, KV 595,200). The per-expert artifact `mosaic-l20-gatea` (85.5 GB) is preserved as the negative
  evidence for §2 — deletable on request.
- **What changed in the plan**: the mosaic route works, but the *allocation granularity is per layer*, not
  per expert. That caps the spend resolution at 906 MB per upgraded layer (288 experts × 3,145,728 B) and
  hence the number of upgraded layers (12 at this size budget).
- **Open items**: (a) full G4 panel + quick MMLU + GPQA MC vs a0 on the 12L mosaic (the confirming run);
  (b) a variant that fits mf 0.90 safely (e.g. 8–10 layers) so the release config keeps its OS headroom;
  (c) recover per-expert granularity via an overlay patch that groups experts by bitrate (days of work,
  needs GPU qualification) — worth it only if the per-layer result confirms the direction; (d) the speed
  question: MTP/nextn and CUDA graphs are the two levers, neither exercised here.