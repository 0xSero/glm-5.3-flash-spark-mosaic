#!/usr/bin/env python3
"""MTP-serving proof + throughput probe (non-greedy sampling).

Reports, per request: TTFT, total time, completion tokens, decode tok/s (excludes prefill), and the
server's speculative-decoding counters delta (drafted/accepted) read from /metrics before and after.
"""
import json
import re
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
PROMPT = ("Write a detailed technical explanation of how a mixture-of-experts transformer layer "
          "routes tokens: the router, top-k selection, capacity factors, and load balancing. "
          "Be thorough and include concrete numbers.")

COUNTERS = ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:num_generation_tokens_total")


def counters():
    out = {}
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=15) as r:
            txt = r.read().decode()
    except Exception as e:  # metrics may be disabled
        return {"error": str(e)[:80]}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for c in COUNTERS:
            if line.startswith(c):
                try:
                    out[c] = float(line.split()[-1])
                except Exception:
                    pass
    return out


def one(i: int):
    body = json.dumps({
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 1.0, "top_p": 0.95, "max_tokens": N, "stream": True,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            ch = d.get("choices") or []
            if ch:
                delta = ch[0].get("delta") or {}
                piece = (delta.get("content") or delta.get("reasoning_content")
                         or delta.get("reasoning") or "")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    ntok += 1
    dt = time.time() - t0
    dec = (ntok - 1) / (dt - ttft) if (ttft and dt > ttft and ntok > 1) else float("nan")
    print(f"run {i}: ttft={ttft:.2f}s total={dt:.2f}s tokens={ntok} "
          f"decode={dec:.2f} tok/s total_rate={ntok/dt:.2f} tok/s", flush=True)


c0 = counters()
print("counters_before:", json.dumps(c0), flush=True)
for i in range(1, REPEATS + 1):
    one(i)
c1 = counters()
print("counters_after:", json.dumps(c1), flush=True)
if "error" not in c0 and "error" not in c1:
    for k in COUNTERS:
        if k in c0 or k in c1:
            print(f"delta {k}: {c1.get(k,0) - c0.get(k,0):.0f}", flush=True)