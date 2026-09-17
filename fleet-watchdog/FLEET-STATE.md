# Fleet state — 2026-09-16T09:55Z

| host | state | last seen | notes |
|---|---|---|---|
| spark-2384 | OK | 09:55Z | glm53-a0-bench release serving (up 3d18h) |
| spark-2822 | OK | 09:55Z | idle; drop_caches loop STOPPED w/ receipt 05:32:13Z; F216 artifact + base staged |
| spark-557f | OK | 09:55Z | idle since 09:43Z teardown (glm53-f216 stopped/removed, drop loop killed — receipt f216/receipts/f216-serve-teardown-20260916.txt); only nv-persisted-restore shim remains (owner sudo item) |
| spark-de5c | DOWN (expected) | — | powered off by owner 2026-09-15; incident 092533Z open |
| pop-os | OK | 09:55Z | observe-only, production untouched; same boot since ~00:58Z 09-15 |

F216 CHAIN COMPLETE (stages 1-4, drivers 8-11): seal → plan → build → eval. VERDICT: F216 (REAP-native on the FULL seal) is worse than E216 (partial-seal REAP) on every G4 metric and loses GPQA MC; only quick MMLU improved (+1.40pp vs E216, still −4.7pp vs a0). Completing the observation dataset did not fix the REAP-native family. Benchmark-backed release remains a0. Next runs: watchdog-only. Parked user decisions: a0-fallback vs healing fork; owner sudo items on 2822/557f; de5c power; pop-os boot cause. glm53-f216 teardown DONE (receipted 09:43Z).

Automation replaced 15:16Z: scheduled driver prompt is now F216-aware. F216 chain: COMPLETE (seal → plan 342b840d… → build verified → eval verdict recorded; capture 29/29 shards + original 332 = FULL SEAL 23,088/37,328,459).
