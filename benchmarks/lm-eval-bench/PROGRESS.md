# Benchmark campaign progress — a0 vs r216 (3 tracks)

Updated: 2026-09-17T17:13:45Z. Refreshed every 10 min by `progress-refresh.sh` (pid 70408); rates/ETAs from deltas between refreshes. Receipts under `benchmarks/lm-eval-bench/`.

## Results so far (a0 = unpruned 2.05bpw reference, r216 = pruned 3.05bpw keep-216)

| Benchmark | Point | Score |
|---|---|---|
| MMLU (full) | a0 | not started |
| MMLU (full) | r216 | in flight 19920/56168 |
| GPQA Diamond MC | a0 | 43.43% ±3.53 |
| GPQA Diamond CoT (strict/flexible) | a0 | 3.54% ±1.32 / 31.82% ±3.32 |
| GPQA Diamond MC | r216 | 40.91% ±3.50 |
| GPQA Diamond CoT (strict/flexible) | r216 | 1.52% ±0.87 / 17.68% ±2.72 |

CoT note: strict = \boxed{} extraction (near-zero in raw-completions mode, no chat template); flexible = last-number extraction; MC loglikelihood is the clean signal. tbench rows appear when legs land.

## Track 1 — MMLU (a0 on spark-2384, r216 on spark-557f; suite-exclusive)

| Leg | Requests | % | Rate (last 10m) | Since start | ETA |
|---|---|---|---|---|---|
| a0 @2384 | 0/56168 | 0% | — | — | — |
| r216 @557f | 19920/56168 | 35% | — | — | — |

## Track 2 — GPQA Diamond

| Leg | Requests | % | Rate (last 10m) | Since start | ETA |
|---|---|---|---|---|---|
| MC a0 @2384 | 792/792 | 100% | — | — | — |
| MC r216 @557f | 792/792 | 100% | — | — | — |
| CoT a0 @2384 | 198/198 | 100% | — | — | — |
| CoT r216 @557f | 198/198 | 100% | — | — | — |

## Track 3 — Terminal-Bench 2.1 (omarchy x86_64 harness; legs launch on suite-exclusive endpoints)

```
full/r216-20260914T201046Z done=1 err=1 running=0 finished=False mean_reward: 0.0 rewards: {}
smoke/glm53-a0-20260914T160937Z done=1 err=1 running=0 finished=True mean_reward: 0.0 rewards: {"reward": 1}
smoke/glm53-r216-2822-20260914T182552Z done=1 err=1 running=0 finished=True mean_reward: 0.0 rewards: {"reward": 1}
smoke/oracle-20260914T160708Z done=1 err=0 running=0 finished=True mean_reward: 1.0 rewards: {"reward": 1}
```

- Smoke (a0, 1 task): plumbing verified (container+verifier+model wiring, reward computed); trial AgentTimeoutError under suite contention at multiplier 1 — preserved in harbor result.json. Not score-bearing.
- Full 89-task legs: timeout multiplier 100; a0 leg auto-launches on 2384 after a0-MMLU exits (tbench-auto-a0.sh); r216 leg targeted at spark-2822 dedicated engine after the 2822 purge + staging.

## Live traces (rolling, last errors + tails)

```
[results/a0.log]
[results/r216.log]
Requesting API:  34%|███▍      | 19186/56168 [4:55:34<5:40:03,  1.81it/s]2026-09-14:16:07:13 ERROR    [models.api_models:557] Exception:ServerDisconnectedError('Server disconnected
2026-09-14:16:07:13 ERROR    [models.api_models:557] Exception:ServerDisconnectedError('Server disconnected'), (no outputs), retrying.
2026-09-14:16:07:13 ERROR    [models.api_models:557] Exception:ClientOSError(54, 'Connection reset by peer'), (no outputs), retrying.
[results/a0-gpqa.log]
[results/r216-gpqa.log]
[tbench omarchy console]
--- /home/sero/terminal-bench/runs/full/r216-20260914T201046Z.console.log
  agent=terminus-2  model=openai//model  -n 1  smoke=0
22:10:51 - LiteLLM:WARNING: utils.py:3077 - register_model: model=openai//model has custom pricing but not in built-in cost map and no prefix/region variant matched; cache_creation_input_token_cost and cache_read_input_token_cost will default to 0 for this model (input/output cost tracking is unaffected). To track cache cost, add them to model_info
Terminated                 env OPENAI_API_KEY="$OPENAI_API_KEY" OPENAI_API_BASE="$API_BASE_URL" MSWEA_API_KEY="$MSWEA_API_KEY" "${cmd[@]}" 2>&1 | tee "$console_log"
--- /home/sero/terminal-bench/runs/full/r216-20260914T201046Z/harbor-console.log
22:10:51 - LiteLLM:WARNING: utils.py:3077 - register_model: model=openai//model has custom pricing but not in built-in cost map and no prefix/region variant matched; cache_creation_input_token_cost and cache_read_input_token_cost will default to 0 for this model (input/output cost tracking is unaffected). To track cache cost, add them to model_info
[r216-tbench server 2822]
[2026-09-14 20:13:56] Decode batch, #running-req: 1, #full token: 2496, full token usage: 0.00, mamba num: 4, mamba usage: 0.13, cuda graph: False, gen throughput (token/s): 10.43, #queue-req: 0
[2026-09-14 20:17:38] SIGTERM received. signum=None frame=None. Draining requests and shutting down...
[2026-09-14 20:17:42] Gracefully exiting... Remaining number of requests 0. Remaining requests remaining_rids=[].
[rank0]:[W914 20:17:46.902918599 ProcessGroupNCCL.cpp:1624] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
  return self.create_error_response(str(e))
2822-health 000
```

## Endpoints

| Point | Health (/health) |
|---|---|
| a0   glm53-a0-bench    spark-2384.internal:8000  (spark-2384) | 200 |
| r216  glm53-r216-bench spark-557f.internal:8000 (spark-557f) | 000 |

## Status / receipts

```
2026-09-16T03:52:01Z start f216-quick
2026-09-16T04:34:00Z exit=0 f216-quick
2026-09-16T04:40:58Z start f216-gpqa-mc
2026-09-16T05:27:12Z exit=0 f216-gpqa-mc
2026-09-16T21:44:55Z start mosaic12l-quick
2026-09-17T00:09:37Z exit=0 mosaic12l-quick
2026-09-17T00:13:08Z start mosaic12l-gpqa-mc
2026-09-17T00:24:02Z exit=0 mosaic12l-gpqa-mc
```

- Prereg: PREREG.json + PREREG-ADDENDUM.json (3 corrections); runners: run-point.sh, run-gpqa.sh
- Logs: results/{a0,r216}.log, results/{a0,r216}-gpqa.log, validate-*.log; history: PROGRESS-HISTORY.jsonl
- tbench prereg: /Users/sero/terminal-bench/TBENCH-PREREG.json; harness: /Users/sero/terminal-bench/framework/
- 2822 purge receipt: work/glm53-single-spark-release-20260911/receipts/DELETION-20260914-2822.log (on 2822)

## What remains

1. Track 2 completes (CoT ~30 min). Track 1 MMLU completes (~6-7h at isolated rates).
2. tbench r216 leg on dedicated spark-2822 engine (staged after purge); a0 leg auto-launches on 2384 post-MMLU.
3. Assemble a0-vs-r216 comparison with receipts; stop campaign lock refresher; resume ranking queue (E216 rebuild, healing, G5 loop); restore panel-tune serving.

Constraints: one GPU job per node; never kill processes; preserve failures; UTC; every number traceable to a receipt.
