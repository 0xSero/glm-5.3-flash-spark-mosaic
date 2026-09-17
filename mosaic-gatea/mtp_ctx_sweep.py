#!/usr/bin/env python3
"""Full-context-window validation of the D2/B12x MTP server.

Reuses the validated client from mtp_serve_probe.py (same sampled payload, same exact-token-ID
accounting, same native vLLM counters, same matched-window math) and walks the program's own
length ladder — the one the D1 structured-sustained TABLE used: 1024, 4096, 16384, 65536, 131072,
200000, 260096 input tokens — with one warmup and one measured cell per length, cold prefix cache.

Writes each cell to disk as soon as it finishes, so a crash preserves everything measured so far.

Usage: python3 mtp_ctx_sweep.py [base_url] [max_tokens] [label]
"""
import json
import os
import sys
import time

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080"
MAXTOK = sys.argv[2] if len(sys.argv) > 2 else "720"
LABEL = sys.argv[3] if len(sys.argv) > 3 else "ctxsweep"

# mtp_serve_probe reads its URL/MAXTOK from argv at import time; hand it ours.
sys.argv = ["mtp_serve_probe.py", URL, "1024", MAXTOK]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_serve_probe as P  # noqa: E402

import native_metrics  # noqa: E402
from timing import matched_window, summarize  # noqa: E402

LENGTHS = (1024, 4096, 16384, 65536, 131072, 200000, 260096)
REPEATS = ("warmup", "measured1")


def cell(target, repeat, prompt, actual):
    before = native_metrics.snapshot(P.BASE + "/metrics", P.MODEL)
    c0 = P.spec_counters()
    row = P.request(prompt)
    c1 = P.spec_counters()
    after = native_metrics.snapshot(P.BASE + "/metrics", P.MODEL)
    native = native_metrics.compare(before, after, [row]) if row["success"] else None
    spec = {}
    if "error" not in c0 and "error" not in c1:
        d = {k: c1.get(k, 0.0) - c0.get(k, 0.0) for k in P.SPEC}
        drafted, accepted = d.get(P.SPEC[0], 0.0), d.get(P.SPEC[1], 0.0)
        spec = {"drafted": drafted, "accepted": accepted,
                "acceptance_rate": (accepted / drafted) if drafted else None}
    entry = {"repeat": repeat, "target": target, "actual_prompt_tokens": actual,
             "usage": row.get("usage"), "finish_reason": row.get("finish_reason"),
             "success": row["success"], "token_id_accounting": row["token_id_accounting"],
             "sampling": {"temperature": 1.0, "top_p": 0.95},
             "matched_window": matched_window([row]),
             "summary": summarize([row], row["ended_monotonic"] - row["started_monotonic"]),
             "spec_decode": spec, "client_wall_seconds": row["ended_monotonic"] - row["started_monotonic"],
             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if native:
        entry["native_metrics"] = {
            "total_decode_tok_s": row["usage"]["completion_tokens"] / native["sum_request_decode_seconds"],
            "prompt_tokens_per_prefill_s": native["prompt_tokens_per_sum_request_prefill_second"],
            "sum_request_decode_seconds": native["sum_request_decode_seconds"],
            "sum_request_prefill_seconds": native["sum_request_prefill_seconds"]}
    else:
        entry["native_metrics"] = None
    return entry


def main():
    os.makedirs(P.OUT, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(P.OUT, f"mtp-ctx-sweep-{LABEL}-{stamp}.json")
    record = {"schema": "mosaic-gatea-mtp-ctx-sweep-v1", "base_url": P.BASE, "model": P.MODEL,
              "max_tokens": P.MAXTOK, "lengths": list(LENGTHS), "repeats": list(REPEATS),
              "sampling": {"temperature": 1.0, "top_p": 0.95},
              "recipe_receipt": "b12x-runtime/mtp-baseline-replay/d2-c1-attempt2",
              "note": ("Sampled (non-greedy) cells on the D2/B12x MTP server; D1/D2's recorded rows were "
                       "temperature 0 and D2's own replay covered only 1024/16384."),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cells": []}
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    def flush():
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
    flush()

    for target in LENGTHS:
        prompt, actual = P.prepare(target)
        print(f"=== target {target}: actual_prompt_tokens={actual}", flush=True)
        for repeat in REPEATS:
            t0 = time.time()
            entry = cell(target, repeat, prompt, actual)
            record["cells"].append(entry)
            record["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            flush()
            w = entry["matched_window"]
            n = entry["native_metrics"] or {}
            s = entry["spec_decode"]
            print(f"  {repeat:>9} {int(time.time()-t0):>4}s wall | {w.get('status')} "
                  f"{w.get('window_seconds', 0):6.2f}s | TOTAL {w.get('total_aggregate_decode_tok_s', float('nan')):6.2f} tok/s "
                  f"| native {n.get('total_decode_tok_s', float('nan')):6.2f} tok/s "
                  f"| prefill {n.get('prompt_tokens_per_prefill_s', float('nan')):6.1f} tok/s "
                  f"| ttft {entry['summary'].get('ttft_p50_seconds', float('nan')):7.2f}s "
                  f"| out {entry['usage']['completion_tokens'] if entry['usage'] else '?'} "
                  f"| acc {s.get('acceptance_rate', float('nan')):.3f} | ok {entry['success']}", flush=True)
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["status"] = "complete"
    flush()
    print("receipt:", path, flush=True)


if __name__ == "__main__":
    main()