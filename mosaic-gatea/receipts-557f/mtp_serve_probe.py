#!/usr/bin/env python3
"""MTP-serving proof + throughput on the D2/B12x recipe, with the program's own accounting.

Method (copied from the program's harness, not invented):
  - stimulus: exact 1024-token prompt built by the same tokenize-bisection as
    b12x-runtime/mtp-baseline-replay/d2-c1-attempt2/exact-baseline-replay/probe.py
  - accounting: exact per-chunk token IDs (return_token_ids), matched-window math from
    timing.py, and native vLLM request-decode seconds from native_metrics.py
  - counters: vllm:spec_decode_num_draft_tokens_total / _accepted_tokens_total deltas
Sampling is ON (temperature 1.0 / top_p 0.95) per standing instruction: no greedy runs.
Consequence, labelled everywhere: D2's temperature-0 row (19.11/19.03 tok/s) is an upper
bound for this row, because speculative acceptance falls off under sampling.
"""
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "harness"))

import native_metrics  # noqa: E402  (verbatim copy of the program harness file)
from timing import matched_window, summarize  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080"
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 720
MODEL = "glm-5.3-flash"
TEMPLATE = {"enable_thinking": True, "reasoning_effort": "low"}
TASK = ("\nReturn one JSON object with key numbers containing every integer from 1 through 300, "
        "in ascending order. Finish immediately after 300.\n")
FILLER = "The archive contains ordinary warehouse inventory records.\n"
SPEC = ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total")
OUT = os.path.join(HERE, "out")


def post(path, payload, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def count(prompt):
    return post("/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                              "add_generation_prompt": True, "chat_template_kwargs": TEMPLATE})["count"]


def prepare(target):
    low, high = 0, target
    while low < high:
        mid = (low + high + 1) // 2
        if count(FILLER * mid + TASK) <= target:
            low = mid
        else:
            high = mid - 1
    base = FILLER * low
    low, high = 0, target - count(base + TASK) + 8
    while low < high:
        mid = (low + high + 1) // 2
        if count(base + " x" * mid + TASK) <= target:
            low = mid
        else:
            high = mid - 1
    prompt = base + " x" * low + TASK
    actual = count(prompt)
    assert target - 4 <= actual <= target, f"prompt outside four-token tolerance: {actual}"
    return prompt, actual


def spec_counters():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=15) as r:
            txt = r.read().decode()
    except Exception as e:
        return {"error": str(e)[:100]}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for c in SPEC:
            if line.startswith(c):
                try:
                    out[c] = float(line.split()[-1])
                except ValueError:
                    pass
    return out


def request(prompt):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "temperature": 1.0, "top_p": 0.95, "max_tokens": MAXTOK, "stream": True,
               "stream_options": {"include_usage": True}, "return_token_ids": True,
               "chat_template_kwargs": TEMPLATE, "response_format": {"type": "json_object"}}
    started = time.monotonic()
    row = {"success": False, "started_monotonic": started, "events": [], "payload": payload}
    with urllib.request.urlopen(urllib.request.Request(
            BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}), timeout=1800) as response:
        for raw in response:
            now = time.monotonic()
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            if chunk.get("usage"):
                row["usage"] = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
                ids = choice.get("token_ids")
                if text or reasoning or ids:
                    row["events"].append({"monotonic": now, "token_ids": ids})
                if choice.get("finish_reason"):
                    row["finish_reason"] = choice["finish_reason"]
    row["ended_monotonic"] = time.monotonic()
    accounted = sum(len(e["token_ids"]) for e in row["events"] if e.get("token_ids") is not None)
    row["success"] = bool(row.get("finish_reason") and row.get("usage")
                          and accounted == row["usage"]["completion_tokens"])
    row["token_id_accounting"] = {"chunks_with_ids": sum(1 for e in row["events"] if e.get("token_ids")),
                                  "ids_total": accounted,
                                  "completion_tokens": row.get("usage", {}).get("completion_tokens")}
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    prompt, actual = prepare(TARGET)
    print(f"prompt_tokens={actual} task=1024-token structured-count stimulus", flush=True)
    cells = []
    for repeat in ("warmup", "measured1", "measured2"):
        before = native_metrics.snapshot(BASE + "/metrics", MODEL)
        c0 = spec_counters()
        row = request(prompt)
        c1 = spec_counters()
        after = native_metrics.snapshot(BASE + "/metrics", MODEL)
        native = native_metrics.compare(before, after, [row]) if row["success"] else None
        window = matched_window([row])
        summary = summarize([row], row["ended_monotonic"] - row["started_monotonic"])
        spec = {}
        if "error" not in c0 and "error" not in c1:
            d = {k: c1.get(k, 0.0) - c0.get(k, 0.0) for k in SPEC}
            drafted = d.get(SPEC[0], 0.0)
            accepted = d.get(SPEC[1], 0.0)
            spec = {"drafted": drafted, "accepted": accepted,
                    "acceptance_rate": (accepted / drafted) if drafted else None}
        native_row = None
        if native:
            generated = row["usage"]["completion_tokens"]
            native_row = {"total_decode_tok_s": generated / native["sum_request_decode_seconds"],
                          "prompt_tokens_per_prefill_s":
                              native["prompt_tokens_per_sum_request_prefill_second"],
                          "sum_request_decode_seconds": native["sum_request_decode_seconds"]}
        cell = {"repeat": repeat, "actual_prompt_tokens": actual, "usage": row.get("usage"),
                "finish_reason": row.get("finish_reason"), "success": row["success"],
                "token_id_accounting": row["token_id_accounting"],
                "sampling": {"temperature": 1.0, "top_p": 0.95},
                "matched_window": window, "summary": summary,
                "native_metrics": native_row, "spec_decode": spec,
                "client_wall_seconds": summary["cell_seconds"]}
        cells.append(cell)
        print(json.dumps({k: v for k, v in cell.items() if k != "matched_window"}, indent=2), flush=True)
    record = {"schema": "mosaic-gatea-mtp-serve-proof-v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "base_url": BASE, "model": MODEL, "image": "glm53-b12x-exl3:jovian-3aada677-r3",
              "recipe_receipt": "b12x-runtime/mtp-baseline-replay/d2-c1-attempt2/{plan.json,admission.json,launch.sh}",
              "note": ("Non-greedy sampling (temperature 1.0/top_p 0.95). D2's temperature-0 rows "
                       "(19.11/19.03 TOTAL decode tok/s) are an upper bound, not a same-conditions comparator."),
              "cells": cells}
    path = os.path.join(OUT, f"mtp-serve-proof-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print("receipt:", path, flush=True)


if __name__ == "__main__":
    main()