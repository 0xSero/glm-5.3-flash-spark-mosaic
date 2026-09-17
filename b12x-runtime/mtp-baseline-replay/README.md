# Exact baseline replay for MTP depth comparisons

This C1 wrapper reuses the four frozen successful structured-generation helpers. It replays the exact accepted baseline request payloads, including JSON response format, low reasoning, output budget and prompt text. Two warmups precede two measured repeats at each selected size. The admitted runtime and per-position native MTP counters are checked around every request. It does not alter any server or launch a workload by itself.

Before execution, finish the current baseline, freeze its complete receipts, launch a fresh depth variant with the depth-sweep tools, and pass runtime/capture/resource admission. The initial functional smoke must have populated native MTP counters. This wrapper fails if counters are missing rather than silently accepting them. Actual GPU execution is pending; only Python syntax has been checked.

```sh
python3 b12x-runtime/mtp-baseline-replay/replay.py \
  --run /absolute/admitted-depth-run \
  --fixture /absolute/structured-sustained-20260912 \
  --output /absolute/new-depth-replay \
  --targets 1024,16384
python3 benchmarks/verify_structured.py /absolute/new-depth-replay --last-integer 300
```

Keep the same target precision, source weights, KV, chunk and sampling settings when attributing a difference to MTP depth. A later NVFP4 draft or different target allocation is a separately labeled comparison. TOTAL and per-request decode coincide at C1. New concurrency measurements use the separately admission-aware harness; queueing multiple clients behind one slot is not concurrency acceptance.
