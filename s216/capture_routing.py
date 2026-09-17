#!/usr/bin/env python3
"""Capture per-expert routing counts from a running SGLang server (expert_distribution_recorder_mode=stat).

Feeds (1) a domain-stratified sample of 0xSero/reap-calibration-data-v1 as raw prefill (/generate, max_new_tokens=1)
and (2) a set of varied chat prompts generated to natural completion (/v1/chat/completions, no token cap),
starting/stopping/dumping the recorder around each batch.  Dumps land in the server's
SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR; this script records which dump files belong to which batch.
"""
import argparse, json, random, time, urllib.request, os, glob, collections
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--corpus", default="/home/sero/w2port/calib/calibration-v1.jsonl")
ap.add_argument("--dump-dir", default="/home/sero/w2port/routing")
ap.add_argument("--per-domain", type=int, default=16)
ap.add_argument("--max-chars", type=int, default=12000)
ap.add_argument("--max-chars-long", type=int, default=20000)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--seed", type=int, default=20260913)
ap.add_argument("--receipt", default="/home/sero/w2port/routing/capture-receipt.json")
ap.add_argument("--skip-gen", action="store_true")
a = ap.parse_args()

def post(path, body=None, timeout=7200):
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

random.seed(a.seed)
bydom = collections.defaultdict(list)
with open(a.corpus) as f:
    for line in f:
        r = json.loads(line); bydom[r["domain"]].append(r)
sel = []
for d in sorted(bydom):
    rows = bydom[d]; random.shuffle(rows); sel += rows[:a.per_domain]
random.shuffle(sel)
print(f"selected {len(sel)} records from {len(bydom)} domains", flush=True)
receipt = {"url": a.url, "corpus": a.corpus, "seed": a.seed, "per_domain": a.per_domain, "batches": [], "records": []}

def run_prefill(rows):
    toks = 0; ids = []
    for r in rows:
        lim = a.max_chars_long if r["domain"] == "long_context" else a.max_chars
        text = r["text"][:lim]
        out = post("/generate", {"text": text, "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
        pt = out["meta_info"]["prompt_tokens"]; toks += pt; ids.append(r["id"])
        receipt["records"].append({"id": r["id"], "domain": r["domain"], "prompt_tokens": pt, "chars": len(text)})
    return {"kind": "prefill", "records": ids, "prompt_tokens": toks}

for i in range(0, len(sel), a.batch):
    receipt["batches"].append(record(f"prefill-{i//a.batch:03d}", lambda rows=sel[i:i+a.batch]: run_prefill(rows)))

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

def run_gen(prompts):
    toks = 0; ptoks = 0; outs = []
    for p in prompts:
        out = post("/v1/chat/completions", {"model": "/model", "messages": [{"role": "user", "content": p}], "temperature": 0.7, "top_p": 0.95})
        u = out.get("usage", {}); toks += u.get("completion_tokens", 0); ptoks += u.get("prompt_tokens", 0)
        msg = out["choices"][0]["message"]
        outs.append({"prompt": p[:60], "completion_tokens": u.get("completion_tokens"), "finish": out["choices"][0].get("finish_reason"),
                     "reasoning_chars": len(msg.get("reasoning_content") or ""), "content_chars": len(msg.get("content") or "")})
        print(json.dumps(outs[-1]), flush=True)
    return {"kind": "generate", "completion_tokens": toks, "prompt_tokens": ptoks, "outputs": outs}

if not a.skip_gen:
    for i in range(0, len(GEN_PROMPTS), 2):
        receipt["batches"].append(record(f"generate-{i//2:03d}", lambda ps=GEN_PROMPTS[i:i+2]: run_gen(ps)))
receipt["total_prompt_tokens"] = sum(b.get("prompt_tokens", 0) for b in receipt["batches"])
receipt["total_completion_tokens"] = sum(b.get("completion_tokens", 0) for b in receipt["batches"])
receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(receipt, open(a.receipt, "w"), indent=1)
print("CAPTURE_DONE", json.dumps({k: receipt[k] for k in ("total_prompt_tokens", "total_completion_tokens")}), flush=True)
