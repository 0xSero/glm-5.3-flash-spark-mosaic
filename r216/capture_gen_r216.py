#!/usr/bin/env python3
"""R216 denser route capture — generation-only continuation of the sealed S216 capture.

The sealed S216 routing-counts.json (sha 1f0e98c8…) came from 7 prefill batches (222,285 tokens)
plus ONE partial generation batch (GEN_PROMPTS[0] only, harvested mid-run; see s216/BUILD-S216.md
deviation note). T216 (driver run 2) proved the route factor carries real ranking signal, so this
capture densifies exactly the thin part: the remaining 15 GEN_PROMPTS (indices 1..15, verbatim from
/home/sero/w2port/routing/capture_routing.py, prompt 0 excluded because its tokens are already in
the sealed counts), each with a hard max_tokens=2048 cap (the first uncapped prompt ran past 8k
reasoning tokens at ~10 tok/s). Recorder mechanics mirror capture_routing.py exactly:
/start_expert_distribution_record → requests → /stop → /dump; dumps land in the server's
SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR and this script records batch→dump mapping.

After this, r216/combine_reduce.py builds routing-counts-r216.json from the OLD dumps (prefill-000..006
+ generate-000-partial, full recomputation, provenance-preserving) + these NEW dumps combined.
"""
import argparse, json, time, urllib.request, os, glob

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--dump-dir", default="/home/sero/w2port/routing")
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--receipt", default="/home/sero/w2port/routing/r216-gen-capture.json")
a = ap.parse_args()

GEN_PROMPTS = [
 "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H5M' into a datetime.timedelta, with tests.",
 "Implement an LRU cache in Rust with O(1) get and put. Explain the ownership choices briefly.",
 "Write a SQL query that finds, for each customer, their second most recent order date. Assume tables customers(id) and orders(id, customer_id, placed_at).",
 "Prove that the square root of 2 is irrational.",
 "A train leaves at 9:15 and travels 210 km at 84 km/h. A second train leaves the same station at 9:45 at 105 km/h on the same route. When and where does it catch up?",
 "Explain, for a curious high-school student, why the sky is blue and sunsets are red.",
 "Balance the chemical equation for the combustion of propane and compute the mass of CO2 produced from 10 g of propane.",
 "Write a short story (about 300 words) about a lighthouse keeper who discovers the light has been guiding ships from the future.",
 "Translate into French, Spanish and Mandarin Chinese: 'The meeting has been moved to Thursday afternoon; please bring the revised budget.'",
 "You have tools: get_weather(city: str), book_flight(from_city: str, to_city: str, date: str). The user says: 'I want to fly from Boston to Denver next Friday if it isn't snowing there.' Respond with the tool calls you would make as JSON, then explain.",
 "Give me a bash one-liner that finds the 20 largest files under /var/log modified in the last 7 days, then explain each flag.",
 "Summarize the causes of the fall of the Western Roman Empire in five bullet points, then name the two most debated among historians.",
 "Draft a polite but firm email to a vendor whose delivery is three weeks late, requesting a revised timeline and a 10% credit.",
 "What are the differences between mitosis and meiosis? Present as a comparison table.",
 "Debug this: `def mean(xs): return sum(xs) / len(xs)` crashes on empty input inside a data pipeline. Propose a fix and discuss whether returning NaN or raising is better here.",
 "Write a haiku sequence (three haiku) about a data center at night.",
]
PROMPTS = GEN_PROMPTS[1:]  # index 0 consumed by the sealed partial capture

def post(path, body=None, timeout=3600):
    req = urllib.request.Request(a.url + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try: return json.loads(raw)
    except Exception: return {"raw": raw.decode(errors="replace")}

def dumps_now(): return set(glob.glob(os.path.join(a.dump_dir, "expert_distribution_recorder_*.pt")))

def record(batch_name, fn):
    before = dumps_now()
    post("/start_expert_distribution_record")
    t0 = time.time(); info = fn()
    post("/stop_expert_distribution_record"); post("/dump_expert_distribution_record")
    time.sleep(1.0)
    new = sorted(dumps_now() - before)
    assert len(new) == 1, (batch_name, new)
    r = {"batch": batch_name, "dump": os.path.basename(new[0]), "seconds": round(time.time() - t0, 1), **info}
    print(json.dumps(r), flush=True); return r

def run_gen(prompts):
    toks = 0; ptoks = 0; outs = []
    for p in prompts:
        out = post("/v1/chat/completions", {"model": "/model", "messages": [{"role": "user", "content": p}],
                                            "temperature": 0.7, "top_p": 0.95, "max_tokens": a.max_tokens})
        u = out.get("usage", {}); toks += u.get("completion_tokens", 0); ptoks += u.get("prompt_tokens", 0)
        msg = out["choices"][0]["message"]
        outs.append({"prompt_idx": GEN_PROMPTS.index(p), "completion_tokens": u.get("completion_tokens"),
                     "finish": out["choices"][0].get("finish_reason"),
                     "reasoning_chars": len(msg.get("reasoning_content") or ""), "content_chars": len(msg.get("content") or "")})
        print(json.dumps(outs[-1]), flush=True)
    return {"kind": "generate", "completion_tokens": toks, "prompt_tokens": ptoks, "outputs": outs}

receipt = {"url": a.url, "purpose": "R216 denser route capture (generation-only continuation)",
           "source_prompts": "verbatim GEN_PROMPTS[1:16] from capture_routing.py (prompt 0 in sealed partial capture)",
           "max_tokens_cap": a.max_tokens, "sealed_prefill_source": "capture-receipt.json (prefill-000..006, 222,285 tokens)",
           "batches": []}
for i in range(0, len(PROMPTS), a.batch):
    receipt["batches"].append(record(f"gen2-{i//a.batch:03d}", lambda ps=PROMPTS[i:i+a.batch]: run_gen(ps)))
receipt["total_prompt_tokens"] = sum(b.get("prompt_tokens", 0) for b in receipt["batches"])
receipt["total_completion_tokens"] = sum(b.get("completion_tokens", 0) for b in receipt["batches"])
receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(receipt, open(a.receipt, "w"), indent=1)
print("CAPTURE_DONE", json.dumps({k: receipt[k] for k in ("total_prompt_tokens", "total_completion_tokens")}), flush=True)
