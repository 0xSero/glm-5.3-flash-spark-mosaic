# Speed path to ~20 tok/s on the exl3-plain stack — root cause + what's missing

Written 2026-09-16 (UTC) from live recon of image `glm53-flash-sglang-exl3-plain:serve4` (id `c65c840f1908`),
container files. Read-only: nothing was modified.

## 1. Where the 20 tok/s expectation comes from (receipted)

`EXPERIMENTS.md`, section "September 12 09:32 UTC — measured depth improvement and next jobs"
(receipt: `benchmarks/structured-sustained-20260912/TABLE.md`; D1/D2 raw cells under
`b12x-runtime/mtp-baseline-replay/d2-c1-attempt2/exact-baseline-replay/`):

| Input tokens | D1 TOTAL decode tok/s | D2 TOTAL decode tok/s | Change |
|---:|---:|---:|---:|
| 1023 | 14.98 | **19.11** | +27.62% |
| 16383 | 14.86 | **19.03** | +28.06% |

Conditions recorded there: same K2 target, **native FP8 MTP draft**, "Depth2 passed actual
target/draft-prefill/draft-decode graph admission", C1, 262144 configured context, 2048 chunk,
.93 memory fraction, cold prefix caching, two measured repeats, windows 35.5–36.3 s.
So **19–20 tok/s is real on this hardware — with MTP speculative decoding + CUDA graphs active.**

## 2. Why the current exl3-plain servings run at 9–10 tok/s

Two capabilities are switched off in this serving path, both visible in every serve log
(`mosaic-gatea/serve-12lh.out:52`, same line in `e216/`, `r216/`, `t216/` attempts):

```
dropped:nextn_mtp_off                  3508       1.936        0.000
...
Engine startup timings (s): ... cuda_graph={prefill=0.00, decode=0.00, target_verify=0.00,
                                             draft_prefill=0.00, draft_decode=0.00, draft_extend=0.00}
```

- **MTP/nextn draft is not loaded.** 3,508 tensors (1.936 GB) of layer 45 are counted and dropped.
- **No CUDA graphs at any stage** (`--disable-cuda-graph`; an attempt to enable them at this artifact
  size fails: `Hybrid (mamba/linear-attention) state cache is too small to serve a ...`).

Measured consequence (steady-state, 128-token greedy single stream, this artifact family):
a0 9.10/9.25/9.24 tok/s; mosaic-12L 9.43/8.93/8.53 (mf 0.90) and 7.01/10.40 (mf 0.95).
Mosaic ≈ a0 ⇒ **the mosaic costs nothing; the stack is the ceiling.**

## 3. The exact blocker in the overlay (file:line receipts)

`/opt/exl3-plain-overlay/exl3_plain_sglang_overlay/loader.py`:

```python
def wrap_load_weights(original):
    def wrapped(self, weights, is_nextn: bool = False):
        if is_nextn:
            raise NotImplementedError("NEXTN/MTP on exl3_plain is a follow-up experiment; launch with MTP off (see LAUNCH.md)")
```

and in the same function, for the *target* load, layer-45 tensors are skipped by design:

```python
if source_name.startswith(f"model.language_model.layers.{NEXTN_LAYER}."):
    acct.add("dropped:nextn_mtp_off", tensor, 0)
    continue
```

`contract.py:22`: `NEXTN_LAYER = 45  # num_nextn_predict_layers == 1 -> one draft layer after the 45 text layers`

The sglang side of the draft already exists in this image —
`/sgl-workspace/sglang/python/sglang/srt/models/glm5_next_nextn.py`:

```python
class Glm5NextForConditionalGenerationNextN(DeepseekV3ForCausalLMNextN):
    @classmethod
    def get_hf_to_sglang_mapper(cls, config):   # model.layers.45.* -> model.decoder.*
    def load_weights(self, weights):            # filters model.language_model.layers.45.*
        return Glm5NextForConditionalGeneration.load_weights(self, nextn_weights, is_nextn=True)
```

plus `_resolve_nextn_quant_config()`, which keeps a *checkpoint-declared unquantized* NextN block in
BF16 (returns `None`). Our checkpoints declare **quantized** MTP (a0: `mtp_bits 2`), so that escape
does not apply — the draft would go through the EXL3 quant method, i.e. straight into the
`NotImplementedError` above.

## 4. What closing the gap requires

1. **Overlay change**: implement the `is_nextn=True` branch instead of raising — reuse the existing
   expert path (`exl3_w13_*`/`exl3_w2_*` params + `weight_loader(expert_id=..., projection=...)`) with the
   decoder's parameter names (`model.decoder.*` after the WeightsMapper remap), and let `parse()` accept
   layer-45 names in the nextn mapping. The draft's experts are the **same 2-bit MCG payloads** that the
   target already decodes today (a0 MTP tensors are kept verbatim by the mosaic build), so **no new decode
   capability is needed** — only plumbing.
2. **Serving flags**: `--speculative-algorithm NEXTN` (num-steps/num-draft-tokens per the D2 receipt) with
   the draft on the same checkpoint; CUDA graphs for target + draft.
3. **Memory re-budget (the real coupling)**: draft weights ≈ 1.9 GB + draft graphs + graph pools. At
   mosaic-12L size (96,105,137,024 B) CUDA graphs **already fail** at mf 0.95
   (`Hybrid state cache is too small`), pool-end headroom only 6.49 GB. So MTP+graphs on the *mosaic*
   artifact most likely needs a **smaller mosaic** (≈8–10 upgraded layers ≈ 92–94 GB) or a0 itself.
   i.e. the quality win (+1.9 pp top-1 measured) and the speed win are competing for the same ~10 GB.
4. **Qualification**: smoke + arithmetic check that draft accept/reject is correct with an EXL3 draft
   (the D2 route used a native-FP8 draft), then a like-for-like tok/s measurement on the same harness.

**Ordering implication:** the mosaic's quality evidence and the speed work are independent; the mosaic
does not block MTP, and MTP does not require rebuilding the mosaic. But a *release* that is both
+1.9 pp and ~19 tok/s will need the 8–10-layer variant to make room.