#!/usr/bin/env python3
"""Render the a0-vs-r216 results table (markdown rows) from lm-eval result JSONs.
Called by progress-refresh.sh each refresh; also runnable standalone.
Receipt paths are printed as trailing comments so every number stays traceable."""
import json, glob, os, sys

B = "/Users/sero/sessions/glm53-single-spark-release-20260911/benchmarks/lm-eval-bench"

def load(point, suite):
    hits = sorted(glob.glob(f"{B}/results/{suite}/bench-{point}/results_*.json"))
    if not hits:
        return None
    try:
        return json.load(open(hits[-1])), hits[-1]
    except Exception:
        return None

def fmt(x, pct=True):
    if x is None:
        return "—"
    return f"{x*100:.2f}%" if pct else str(x)

def cell(point, suite, task, keys):
    got = load(point, suite)
    if not got:
        return "—"
    d, path = got
    res = d.get("results", {}).get(task, {})
    vals = []
    for k in keys:
        v = res.get(k)
        se = res.get(k.replace(",", "_stderr,"))
        if isinstance(v, (int, float)):
            vals.append(fmt(v) + (f" ±{se*100:.2f}" if isinstance(se, (int, float)) else ""))
    return vals[0] if vals else "—"

rows = []
# MMLU: show result JSON score when the suite finished, else live progress from log
for tag, log, key in [("a0", "a0.log", "mmlu_a0"), ("r216", "r216.log", "mmlu_r216")]:
    score = cell(tag, tag, "mmlu", ["acc,none", "acc_norm,none"])
    if score == "—":
        n = 0
        try:
            for line in open(f"{B}/results/{log}"):
                if "| /56168" in line or "]| " in line:
                    pass
        except Exception:
            pass
        import re
        m = None
        try:
            txt = open(f"{B}/results/{log}", errors="replace").read()
            ms = re.findall(r"\| (\d+)/56168", txt)
            m = ms[-1] if ms else None
        except Exception:
            pass
        score = f"in flight {m}/56168" if m else "not started"
    rows.append(f"| MMLU (full) | {tag} | {score} |")
    if tag == "a0":
        pass
for tag, path in [("a0", "a0-gpqa"), ("r216", "r216-gpqa")]:
    mc = cell(tag, path, "gpqa_diamond_zeroshot", ["acc,none"])
    st = cell(tag, path, "gpqa_diamond_cot_zeroshot", ["exact_match,strict-match"])
    fx = cell(tag, path, "gpqa_diamond_cot_zeroshot", ["exact_match,flexible-extract"])
    rows.append(f"| GPQA Diamond MC | {tag} | {mc} |")
    rows.append(f"| GPQA Diamond CoT (strict/flexible) | {tag} | {st} / {fx} |")
    got = load(tag, path)
    if got:
        rows.append(f"<!-- receipt: {got[1]} -->")
for r in rows:
    if r.startswith("<!--"):
        print(r)
print("| Benchmark | Point | Score |")
print("|---|---|---|")
for r in rows:
    if not r.startswith("<!--"):
        print(r)
