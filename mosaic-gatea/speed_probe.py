#!/usr/bin/env python3
"""Single-stream greedy speed probe (matches the shape of the recorded a0/mosaic numbers:
128-token greedy generation, one request at a time, temperature 0, ignore_eos)."""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://spark-557f.internal:8000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
PROMPT = ("Write a detailed technical explanation of how a transformer mixture-of-experts "
          "layer routes tokens, covering the router, the top-k selection, capacity factors, "
          "and load balancing.")

for i in range(1, REPEATS + 1):
    body = json.dumps({"model": "default", "prompt": PROMPT, "max_tokens": N,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(URL + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    tok = d.get("usage", {}).get("completion_tokens", 0)
    print(f"{dt:.2f}s  {tok} tokens  {tok/dt:.2f} tok/s  (run {i}, incl. prefill)", flush=True)