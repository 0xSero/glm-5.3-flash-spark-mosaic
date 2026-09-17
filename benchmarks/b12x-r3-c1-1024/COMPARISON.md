# Full K2 with FP8 MTP: legacy versus B12x

**Experimental reasoning-token measurements; quality promotion remains on hold.** All four requests reached the 2,048-token limit with empty final answers. They did not complete the requested 240-comment-line task. Transport, exact-token accounting, sustained-window recalculation and native MTP execution checks passed.

| Runtime | Repeat | Actual prompt | C | Native prefill tok/s | TTFT s | TOTAL decode tok/s | Per-request decode tok/s | Window s | End-to-end total tok/s | Accepted/drafted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy FlashInfer | 0 | 1024 | 1 | 109.18 | 9.440 | **13.753** | 13.753 | 148.84 | 12.939 | 893/1154 |
| Legacy FlashInfer | 1 | 1024 | 1 | 429.68 | 2.426 | **13.475** | 13.475 | 151.92 | 13.269 | 873/1174 |
| B12x/Jovian R3 | 0 | 1024 | 1 | 236.38 | 4.341 | **13.661** | 13.661 | 149.84 | 13.283 | 884/1164 |
| B12x/Jovian R3 | 1 | 1023 | 1 | 410.27 | 2.501 | **13.553** | 13.553 | 151.04 | 13.339 | 877/1170 |

Pooled TOTAL decode across the two matched windows: legacy **13.612 tok/s**, B12x **13.607 tok/s** (-0.04%). This small experiment does not establish a decode speed improvement. Prefill improved in the first repeat and regressed in the second; warm-state equivalence is not established.

Both use the same full 288-expert K2 target, native MTP depth 1 with FP8 routed draft experts, 262,144 requested context, 2,048 prefill chunk, and one active slot. Legacy has 460,208 aggregate KV tokens; B12x has 751,007. Source benchmark scripts have identical hashes. The second B12x prompt has 1,023 tokens; others have 1,024. The runtime versions, kernels, packed cache layout and physical Spark differ.

TOTAL decode counts exact token IDs in the common `(start,end]` window, excluding the first streamed bundle and preserving speculative bundles. It includes reasoning. End-to-end total includes prefill/waiting. Native prefill is request-level server timing, not an aggregate GPU-only rate. No multi-concurrency or long-context speed result is implied. Cached-token usage fields are absent; prefix caching is disabled in the sealed runtime configuration, not inferred as an exported zero.

## Failed cases

| Runtime | Repeats | Final content | Finish | Result |
|---|---|---|---|---|
| Legacy FlashInfer | 0,1 | Empty | length | Requested 240-line output not completed |
| B12x/Jovian R3 | 0,1 | Empty | length | Requested 240-line output not completed |

No transport/OOM/stream-accounting failure occurred in these four cells. Semantic failure/length truncation remains distinct from measured reasoning throughput. Separate serving-quality evidence remains parent-owned.
