# M288-12L confirming run — full G4 panel + task benchmarks vs a0

Run 2026-09-16/17 UTC, user-directed ("okay do it"). Artifact: `M288-12L-GateA2`
(`557f:/home/valentine/models/mosaic-12l-gatea`, 96,105,137,024 B, plan_sha256 7938939c…), served on
spark-557f as container `glm53-mosaic-12lh` (image `glm53-flash-sglang-exl3-plain:serve4` = c65c840f1908,
mem-fraction 0.95, KV 595,200, max-running-requests 1, ctx 262,144, vision on, fp8 KV, dsa, no CUDA graphs).

Receipts: this directory (Mac copies) + `2384:/home/sero/w2port/out/{mosaic12l-full-*,a0-full-*,…}.json` +
`557f:/home/valentine/mosaic-gatea/`.

## 1. Full G4 panel — 65,504 positions, 32 rows (the release panel)

Both sides measured **the same day, same harness, same image, same teacher rows**, teacher reproduction
exact on both (`top1_exact_all_rows: true`). a0 was re-measured (not taken from the leaderboard) so the
comparison is like-for-like.

| Metric | a0 (2.05bpw, reference) | **M288-12L mosaic** | Δ |
|---|---:|---:|---|
| top-1 agreement | 0.78902 | **0.80842** | **+1.94 pp** |
| KL lower-bound mean | 0.38378 | **0.32522** | **−15.3%** |
| KL p50 / p95 | 0.1044 / 1.797 | **0.0811 / 1.540** | better |
| KL p99 / max | 3.849 / 13.73 | 3.424 / 14.23 | ~ |
| candidate PPL | 4.28692 | **4.03563** | **−5.9%** |
| samples (20 pre-registered) | 15/20 (re-run after the §4 crash) | 14/20 | −1 |

**Per-row: the mosaic wins 32/32 rows on top-1, 32/32 on KL, 32/32 on NLL.** No row regresses.

Receipts: `mosaic12l-full-panel.json` / `mosaic12l-full-summary.json`, `a0-full-panel.json` /
`a0-full-summary.json` (both on 2384; Mac copies in `receipts-2384/out/`). Panel fixture manifest
`46255621…`, teacher manifest `a1604015…` — identical on both sides; all three panels report
`teacher_reproduction {top1_exact_all_rows: true, max_nll_sum_abs_delta: 0.005112852444881355, ok: true}`.

## 1b. Task benchmarks — quick MMLU + GPQA MC vs a0 (both release bars met)

| Benchmark | a0 (reference) | **M288-12L mosaic** | Bar |
|---|---:|---:|---|
| MMLU (quick, 57 subjects × limit 20, seed 1234) | 0.8342 ± 0.0105 | **0.8360 ± 0.0104** | > 0.8342 |
| GPQA diamond zeroshot (MC, 198 docs) | 0.4343 ± 0.0353 | **0.4394 ± 0.0354** | ≥ 0.4343 |
| samples (20) | 15/20 | 14/20 | ≥ 16 → **not met** |

The mosaic is the only artifact in the program to beat a0 on the panel *and* not regress on either
task benchmark (e216 0.7728 / 0.3788, f216 0.7868 / 0.3687 are −6.1/−5.6 pp MMLU and −4.6/−6.6 pp GPQA
below a0 on the same harness).

Receipts: `benchmarks/lm-eval-bench/results/mosaic12l-quick/bench-mosaic12l-quick/results_2026-09-16T20-09-27.917455.json`,
`results/mosaic12l-gpqa-mc/bench-mosaic12l-gpqa-mc/results_2026-09-16T20-24-02.421014.json`,
`results/a0-quick/bench-a0/results_2026-09-14T19-49-13.012988.json`,
`results/a0-gpqa/bench-a0/results_2026-09-14T12-55-54.578035.json`, driver log `results/status.txt`
(`2026-09-16T21:44:55Z start mosaic12l-quick` → `2026-09-17T00:09:37Z exit=0`; GPQA
`00:13:08Z` → `00:24:02Z exit=0`).

## 2. 10-layer sibling (`M288-10L-GateA3`) — built for memory headroom

Not a quality candidate; a serving-headroom variant prepared because the 12L's +11 GB pushes the
262,144-token KV gate over `--mem-fraction-static 0.90`.

- Plan frozen: `mosaic-10l-plan.json`, plan_sha256 `2dd29c133f52318d8d1e49b92265aa49e4a5b1a539adc67941d9c36c575bb474`,
  layers **[3, 33, 37, 38, 39, 40, 41, 42, 43, 44]** = top-10 by FULL-seal mass (46.99% of mass; the two
  dropped vs 12L are #11 layer 36 and #12 layer 32). Ranking recomputed from `f216/f216-plan.json`
  `weighted_ean_sum_by_layer`; it reproduces the 12L selection exactly for the top-12.
- Build: artifact 94,293,197,696 B; **34,560/34,560 upgraded tensors byte-equal to the 3.05bpw source,
  33,029/33,029 kept tensors byte-equal to a0, 5 shards differ, index byte-identical** (`shas-*-10l.txt`).
- Census (stock route): **contract_ok true**, `moe_sparkinfer_native true`, 0 problems,
  expert bits {2: 28512, 3: 8640}, layer 3 = 3 bits, layer 20 = 2 bits, layer 44 = 3 bits.
- Packer verdict was `FAIL` on the exact-delta assertion only: delta 9,059,707,346 vs expected
  9,059,696,640 = **+10,706 B**, byte-for-byte the same header drift the 12L showed (+10,706 B, same five
  rewritten shards). `n_failures: 0`. `rc=2` aborted the build script before its sha/index steps, which
  were then run separately (this is why both mosaics carry a FAIL receipt that is not a defect).
  **`mosaic_pack2.py`'s exact delta equality check produces false FAILs; the per-tensor checks are the
  load-bearing ones.**

### 2b. 10L full panel — 65,504 positions, 32 rows (measured 2026-09-17T10:09:57Z→10:10:01Z)

| Metric | a0 | M288-12L | **M288-10L** |
|---|---:|---:|---:|
| top-1 agreement | 0.78902 | 0.80842 | **0.80278** |
| KL lower-bound mean | 0.38378 | 0.32522 | **0.34264** |
| KL p50 / p95 | 0.1044 / 1.797 | 0.0811 / 1.540 | 0.0856 / 1.639 |
| candidate PPL | 4.28692 | 4.03563 | **4.10959** |

Clean dose–response over the mosaic idea: **0 layers (a0) 78.90 → 10 layers 80.28 → 12 layers 80.84**
top-1, monotone in KL and PPL as well. The 10L still beats a0 by +1.38 pp / −10.7% KL / −4.1% PPL
(artifact 94,293,197,696 B = 9.1 GB above a0's 85.2 GB, 1.8 GB below the 12L's 96,105,137,024 B).
Receipts `receipts-2384/out/mosaic10l-full-panel.json` (sha `8e1b50343127d27e6f9c9cac3626c29061340244f281dfc7f03bbd3248009cf5`),
`g4/panel-mosaic10l-full/` (32 `cand-row-*.pt`); teacher reproduction exact; served at **mf 0.90 / KV
209,024** (the boot whose KV misses the 262,144 gate — the panel is valid at 2,048-token inputs).

## 3. Speed — MTP now serving; the gap was the runtime, not the artifact

See §5 for the full serving proof. In one line: on the **B12x runtime** the same K2 target with the
program's native MTP draft + B12X attention + `fp8_ds_mla` + depth 2 serves structured decode at
**19.23 / 19.00 token/s** (sampling on), i.e. the D2 receipt class; the mosaic itself is a
**mul1-codebook** artifact and that runtime's exl3 overlay refuses `codebook != 'mcg'`
(`MTP-CODEBOOK-BLOCKER.md`), so the mosaic stays on the SGLang exl3-plain path at 9–10 token/s with
`dropped:nextn_mtp_off` and no CUDA graphs. The two facts are independent: the mosaic's quality win
does not depend on which runtime serves it, and the fast path does not accept its codebook.

Labelling note: the 9–10 / 11.2–11.5 token/s mosaic and a0 figures were taken **before** the standing
"no greedy" instruction, with the simple probe that sets `temperature: 0` (`speed_probe.py` — the same
convention the program's earlier acceptance client used). Every MTP number in §5 is sampled
(`temperature 1.0 / top_p 0.95`). No apples-to-apples non-greedy mosaic figure exists yet; if one is
wanted, re-run `speed_probe.py` with a sampling payload against a relaunched mosaic server
(`run-serve-12lh.sh`).

## 4. Fault record — a0 baseline server died mid-samples (NCCL)

The first a0 samples leg is **invalid**: 16 of 20 samples failed with
`ConnectionResetError(104, 'Connection reset by peer')` because the a0 server on 2384 crashed at
**2026-09-16T21:41:41Z** with `torch.distributed.DistBackendError: NCCL error … unhandled system error`
(`ncclSystemError`), after which SGLang's SIGQUIT handler ran `kill_process_tree` (container
`Exited (0)`). The a0 **panel** row from the same run is valid (it completed before the crash, with
teacher-exact reproduction). The container was restarted with its original flags (ctx 262144, chunked
256/prefill 256, max-running 16, mf 0.90, no graphs) and the samples leg re-run as
`a0-full-samples-r2.json` / `a0-full-summary-r2.json` (`run-a0-samples-r2.sh`, on 2384).
## 5. MTP is serving — the D2/B12x recipe reproduced (2026-09-17, user: "you need to make it serve mtp")

Two servers were run on 557f, both with CUDA graphs ON (`cudagraph_mode: FULL_DECODE_ONLY`, capture
sizes `[1,3]`/`[2]`) and MTP ON (`--speculative-config method=mtp`). No greedy call was used anywhere.

**(a) r4-variant — `glm53-k2-mtp`** (image `local/glm53-reap-native-mtp:20260911-r4-expert-fp8` =
4f06a26d872d, `FLASHINFER_MLA_SPARSE_SM120`, `fp8`, block 64, depth 1, KV 565,065):
MTP served — spec counters advanced **+900 drafted / +633 accepted** (70.3%) across three 512-token
sampled generations; server log + inspect preserved in `receipts-557f/`. Client-side token counting in
that first probe was invalid (SSE chunks coalesce several tokens), which is why the program's harness
counts token IDs, not chunks; from the counters the cell is ≈13.3 tok/s. Replaced, not deleted.

**(b) D2/B12x replay — `glm53-k2-mtp-b12x-d2`** (image `glm53-b12x-exl3:jovian-3aada677-r3` =
sha256:afb74c791853…, the program's own
`b12x-runtime/mtp-baseline-replay/d2-c1-attempt2/{launch.sh,plan.json,admission.json}` recipe:
`--attention-backend B12X --additional-config {"kda_prefill_backend":"b12x"}`,
`--kv-cache-dtype fp8_ds_mla --block-size 256`, spec depth 2 with `attention_backend: B12X`,
mf 0.93, ctx 262,144, vision on):

| cell | window | TOTAL decode | native decode | acceptance | tokens | TTFT | prefill |
|---|---:|---:|---:|---:|---:|---:|---:|
| warmup | 35.64 s | 19.11 tok/s | 19.14 tok/s | 98.3% | 682 | 5.93 s | 226 tok/s |
| measured1 | 35.41 s | **19.23 tok/s** | 19.26 tok/s | 99.6% | 682 | 2.63 s | 408 tok/s |
| measured2 | 35.84 s | **19.00 tok/s** | 19.03 tok/s | 98.3% | 682 | 2.48 s | 435 tok/s |

- **Sampling ON** (`temperature 1.0`, `top_p 0.95`) per standing instruction. D2's own rows were
  `temperature 0`; the stimulus here is the same 1,023-token structured-count prompt the D2 replay used,
  and acceptance stayed 98.3–99.6%, so the non-greedy rows land on the D2 numbers (19.11/19.03) rather
  than below them.
- Accounting is the program's, not a new invention: exact per-chunk `token_ids`
  (`682 = 682` on both measured cells, `MATCHED_SUSTAINED` ≥ 30 s), matched-window math from
  `harness/timing.py` (sha `91d413a267…`, byte-identical to the D2 replay copy), and native
  `vllm:request_decode_time_seconds` / `request_generation_tokens` counters via `harness/native_metrics.py`
  (`vllm:request_*_count` deltas equal the request count, i.e. the vLLM-side and client-side token
  totals agree).
- Admission evidence: `GPU KV cache size: 727,449 tokens, Maximum concurrency for 262,144 tokens per
  request: 2.77x`; `Capturing model for speculator...` twice (target + draft) with
  `Graph capturing finished` both times; six capture receipts written by the program's own
  `capture_plugin` into `receipts-557f/d2-profiles/capture-receipts/`; weights 102.29 GiB, peak
  activation 4.78 GiB, CUDAGraph 0.21 GiB; `/v1/models` → `glm-5.3-flash`, `max_model_len 262144`.
- Receipt: `receipts-557f/out/mtp-serve-proof-20260917T122332Z.json`
  (sha `f14dd44b77e227c74cb40bac5cea677a45989af49a8d0296733988d5cd6d12ad`), log
  `out/mtp-serve-probe-b12x-20260917T122134Z.log`, launcher `launch-k2-mtp-b12x-d2.sh`
  (sha `e099005c2d4cb4a163f46dcb0eeb482df594cb2192914bb6cbe9c56e73b08ab3`), probe `mtp_serve_probe.py`
  (sha `3a7fe1ca2acf638fe1dec7285f626f86bb9f11675e2d275a021c025ac4bfc79b`).

### 5b. Full-context-window validation — 1,023 → 260,095 input tokens (2026-09-17T16:05:05Z → 17:04:16Z)

One warmup + one measured cell at each length on the program's own ladder (the one the D1
structured-sustained TABLE used), cold prefix cache, same sampled payload, same accounting. **14/14
cells `ok`, 14/14 `MATCHED_SUSTAINED`, server healthy (200) after the run.**

| input tokens | repeat | window s | TOTAL decode tok/s | native cross-check | prefill tok/s | TTFT s | out tokens | MTP acceptance |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1,023 | warmup | 35.35 | 19.24 | 19.26 | 333.6 | 3.13 | 681 | 0.993 |
| 1,023 | measured | 35.54 | 19.19 | 19.21 | 412.1 | 2.61 | 683 | 0.991 |
| 4,095 | warmup | 38.74 | 18.56 | 18.57 | 379.0 | 10.87 | 720 | 0.952 |
| 4,095 | measured | 35.49 | 19.22 | 19.23 | 433.6 | 9.59 | 683 | 0.998 |
| 16,383 | warmup | 36.45 | 18.93 | 18.94 | 411.9 | 39.86 | 691 | 0.976 |
| 16,383 | measured | 36.41 | 18.95 | 18.96 | 444.8 | 36.99 | 691 | 0.987 |
| 65,535 | warmup | 38.37 | 18.56 | 18.55 | 446.8 | 146.90 | 713 | 0.955 |
| 65,535 | measured | 36.40 | 18.82 | 18.81 | 455.0 | 144.26 | 686 | 0.978 |
| 131,071 | warmup | 37.45 | 18.67 | 18.65 | 449.7 | 291.78 | 700 | 0.971 |
| 131,071 | measured | 35.62 | 19.00 | 18.98 | 454.7 | 288.58 | 678 | 1.000 |
| 199,999 | warmup | 36.11 | 18.86 | 18.83 | 450.7 | 444.17 | 682 | 0.989 |
| 199,999 | measured | 37.06 | 18.70 | 18.66 | 453.2 | 441.83 | 694 | 0.977 |
| 260,095 | warmup | 36.00 | 18.95 | 18.88 | 452.0 | 575.97 | 683 | 0.991 |
| 260,095 | measured | 36.05 | 18.89 | 18.81 | 452.4 | 575.65 | 682 | 0.991 |

- **Decode is flat across the window**: measured cells span **18.70–19.22 tok/s** (2.7% spread) from
  1k to 260k, versus D1's recorded 14.98 → 14.80 on the same ladder (identical shape, different level).
  MTP acceptance stays **0.95–1.00** at every length — speculative decoding does not degrade with context.
- **Prefill saturates at ~450 tok/s** from 16k upward (444.8 / 455.0 / 454.7 / 453.2 / 452.4), matching
  D1's 442–455 range. TTFT is therefore linear in prompt length above ~16k: 37 s at 16k, 144 s at 65k,
  289 s at 131k, 442 s at 200k, 576 s at 260k.
- **The window is genuinely usable end to end**: 260,095 input + 683 output = 260,778 of the 262,144
  limit, against KV 727,449 tokens (one request at a time, `--max-num-seqs 1`), with no OOM, no
  eviction, and the server still answering after the run.
- Receipt: `receipts-557f/out/mtp-ctx-sweep-full-20260917T160505Z.json` (436 KB, per-cell token-ID
  arrays and native counter deltas), log `out/ctx-sweep-full.log`, runner `mtp_ctx_sweep.py`
  (reuses the validated `mtp_serve_probe.py` client; cells written to disk as they finish).
- Scope: throughput/prefill/acceptance across context on **one** C1 stream with the counting stimulus.
  Not claimed: concurrency above 1, other workloads' acceptance rates, or long-context quality.

**What this does and does not say.** It does say MTP serves on this artifact lineage, with graphs, at
262,144 context, at the receipted D2 rate — the 9–10 tok/s figures quoted elsewhere in this program are
the *exl3-plain/SGLang* path, which drops the nextn draft. It does not say the mosaic is fast: the
mosaic is a **mul1** artifact and the MTP-capable images reject `codebook != "mcg"`, so the mosaic
cannot use this runtime (see `MTP-CODEBOOK-BLOCKER.md`). Closing that gap needs either a mul1
implementation in the vLLM exl3 overlay or a mcg re-quant of the mosaic — the latter still blocked by
the Spark exllamav3 build failure.

## 6. Fleet-integrity observation (not caused by this program's work)

Both sealed turboderp source quants disappeared from 557f `/home/valentine/models/` between
2026-09-16T22:00Z and 2026-09-17T12:00Z (~300 GB): the 2.05bpw and 3.05bpw directories that the mosaics
were byte-copied from. This run did not delete them (no `rm` was issued by any script here; the 10L/12L
builds only read them). Both mosaic artifacts remain intact and byte-identical to their pack receipts
(`96,105,137,024 B` / `94,293,197,696 B`, INDEX_BYTE_IDENTICAL). Reported for the owner's records; no
action taken.
