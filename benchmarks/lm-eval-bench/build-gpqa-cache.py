#!/usr/bin/env python
# Reconstruct Idavidrein/gpqa gpqa_diamond.csv from three public mirrors.
# Authority chain for the correct answer: (1) hendrydong/gpqa_diamond boxed
# answer TEXT aligned by containment, (2) hendrydong/gpqa_diamond_mc boxed
# letter over its own options, (3) fingertap keymap/letter. Options text:
# fingertap embedded options (lowercase 'a)' / uppercase 'A.' styles) or, for
# list-style questions, the trailing 'A. val' block itself.
import csv, json, os, re, sys
import pyarrow.parquet as pq
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results/gpqa-reconstruct")
os.makedirs(OUT, exist_ok=True)

def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()

TRAIL = re.compile(r"(?m)^([A-D])\.\s*(\S[^\n]*)$")
LO = re.compile(r"(?m)^([a-d])\)\s?")
UPDOT = re.compile(r"(?m)^([A-D])\.\s+")
MCUP = re.compile(r"(?m)^\(([A-D])\)\s?")

def parse_block(text, rx):
    ms = list(rx.finditer(text))
    if [m.group(1).upper() for m in ms] != ["A", "B", "C", "D"]:
        return None
    opts = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        opts[m.group(1).upper()] = text[m.end() : end].strip()
    return norm(text[: ms[0].start()]), opts

ft = pq.read_table("/tmp/ft_diamond.parquet").to_pylist()
hd = pq.read_table("/tmp/hd_diamond.parquet").to_pylist()
mc = load_dataset("hendrydong/gpqa_diamond_mc", split="test")

hd_rows = []
for r in hd:
    boxed = re.search(r"\\boxed\{(.+?)\}\s*$", (r["solution"] or "").strip())
    if boxed:
        hd_rows.append((norm(r["problem"]), norm(boxed.group(1)), r.get("domain", "")))

def find_hd(stem):
    st = norm(stem)
    hits = [h for h in hd_rows if len(st) >= 25 and st in h[0]]
    return hits[0] if len(hits) == 1 else None

mc_rows = []
for r in mc:
    text = re.sub(r"\s*Please write your final answer in the form of[\s\S]*$", "", r["problem"] or "")
    pb = parse_block(text, MCUP)
    m = re.search(r"\\boxed\{([A-D])\}", r["solution"] or "")
    if pb and m:
        mc_rows.append((pb[0], pb[1], m.group(1)))

def find_mc(stem):
    st = norm(stem)
    hits = [r for r in mc_rows if len(st) >= 25 and st in r[0]]
    return hits[0] if len(hits) == 1 else None

rows, bad, stats = [], [], {"hd": 0, "mc": 0, "ft": 0, "agree2": 0}
for i, r in enumerate(ft):
    text = (r["question"] or "").rstrip()
    letter = (r["answer"] or "").strip()[-1:].upper()
    # trailing block = last 4 consecutive TRAIL lines ending at EOF
    trail = []
    lines = text.split("\n")
    while lines and (not lines[-1].strip()):
        lines.pop()
    for ln in reversed(lines[-8:]):
        mt = TRAIL.fullmatch(ln.strip()) if ln.strip() else None
        if mt:
            trail.insert(0, mt)
        elif trail:
            break
        elif ln.strip() == "":
            continue
        else:
            break
    question, opts, correct, domain, prov = None, None, None, "", None

    if len(trail) == 4 and [m.group(1) for m in trail] == ["A", "B", "C", "D"] and trail[-1].end() >= len(text) - 2:
        values = [m.group(2).strip() for m in trail]
        body = text[: text.rfind(trail[0].group(0))].rstrip()
        pb = parse_block(body, LO)
        if pb and all(re.fullmatch(r"[a-d]", v) for v in values):
            # keymap shape: body has the 4 lowercase options; trail maps display→label
            question, opts = pb
            correct = opts.get(values[ord(letter) - 65], "")
            prov = "ft-keymap"
        else:
            # value shape: trail IS the four choices (e.g. list-count questions)
            question = norm(body)
            opts = {k: v for k, v in zip("ABCD", values)}
            correct = values[ord(letter) - 65]
            prov = "ft-values"

    if opts is None:
        pb = parse_block(text, LO) or parse_block(text, UPDOT) or parse_block(text, MCUP)
        if not pb:
            bad.append((i, "ft-parse")); continue
        question, opts = pb

    # authority 1: hd boxed text matched against options
    h = find_hd(question)
    if h and correct is None:
        hits = [k for k, v in opts.items() if norm(v) == h[1]]
        if len(hits) == 1:
            correct = opts[hits[0]]; domain = h[2]; prov = prov or "hd"; stats["hd"] += 1
    # authority 2: mc boxed letter (only when hd didn't already fix correctness)
    m = find_mc(question)
    if correct is None and m and m[2] in m[1]:
        correct = m[1][m[2]]; prov = "mc"; stats["mc"] += 1
    if correct is None and prov and prov.startswith("ft"):
        stats["ft"] += 1  # trusted fingertap-only resolution
    if correct is None:
        bad.append((i, "no-correct-source")); continue
    # cross-check hd when available
    if h:
        normcorr = norm(correct)
        hbx = h[1]
        if normcorr != hbx and hbx not in normcorr and normcorr not in hbx:
            bad.append((i, f"hd-disagree prov={prov}")); continue
        domain = domain or h[2]
    normcorr = norm(correct)
    incorrect = [v for k, v in sorted(opts.items()) if norm(v) != normcorr]
    if len(incorrect) != 3:
        bad.append((i, "non-4-options")); continue
    rows.append({
        "Record ID": f"diamond-{len(rows):03d}",
        "Question": question,
        "Correct Answer": correct,
        "Incorrect Answer 1": incorrect[0],
        "Incorrect Answer 2": incorrect[1],
        "Incorrect Answer 3": incorrect[2],
        "Subdomain": domain,
    })

print(f"rows={len(rows)} bad={len(bad)} stats={stats}")
for b in bad[:10]:
    print("  BAD:", b)
if len(rows) != 198 or bad:
    sys.exit("RECONSTRUCTION INCOMPLETE — not writing cache")

csv_path = os.path.join(OUT, "gpqa_diamond.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL)
    w.writeheader()
    w.writerows(rows)
print("wrote", csv_path, os.path.getsize(csv_path), "bytes")
json.dump({"rows": len(rows), "bad": len(bad), "stats": stats,
           "authority": "hd boxed text > mc boxed letter > fingertap keymap/letter; cross-checked",
           "mirrors": ["hendrydong/gpqa_diamond", "hendrydong/gpqa_diamond_mc", "fingertap/GPQA-Diamond"]},
          open(os.path.join(OUT, "reconstruct-receipt.json"), "w"), indent=1)
