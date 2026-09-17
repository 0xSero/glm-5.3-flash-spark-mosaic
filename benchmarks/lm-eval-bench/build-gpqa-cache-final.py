#!/usr/bin/env python
# Build a local HF hub cache for Idavidrein/gpqa (gated upstream) using the
# public mirror bdytx5/gpqa_gpqa_diamond, which carries the FULL original
# schema. Every row is cross-checked against the independent hendrydong/
# gpqa_diamond mirror (boxed answer TEXT must equal the row's Correct Answer).
# Output: ~/.cache/huggingface/hub/datasets--Idavidrein--gpqa with refs/main +
# snapshots/<sha>/{README.md, gpqa_diamond.csv} so the preregistered task
# gpqa_diamond_zeroshot runs with HF_HUB_OFFLINE=1. Receipts under results/.
import csv, hashlib, json, os, re, sys
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results/gpqa-reconstruct")
os.makedirs(OUT, exist_ok=True)

def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()

ds = load_dataset("bdytx5/gpqa_gpqa_diamond", split="train")
hd = load_dataset("hendrydong/gpqa_diamond", split="test")
print(f"bdytx5 rows={len(ds)} cols_needed_present=", all(
    c in ds.column_names for c in ("Question", "Correct Answer", "Incorrect Answer 1",
                                   "Incorrect Answer 2", "Incorrect Answer 3", "Subdomain", "Record ID")))

def brace_content(s):
    """Extract content of the LAST \\boxed{...} with balanced braces (LaTeX-nested)."""
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth, out = i + 7, 1, []
    while j < len(s) and depth:
        c = s[j]
        if c == "\\" and j + 1 < len(s):
            out.append(s[j : j + 2]); j += 2; continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if not depth:
                break
        out.append(c); j += 1
    return "".join(out) if depth == 0 else None

hd_problems = [norm(r["problem"]) for r in hd]
hd_boxed = []
for r in hd:
    bx = brace_content(r["solution"] or "")
    hd_boxed.append(norm(bx) if bx is not None else None)

rows, bad = [], []
for i, r in enumerate(ds):
    q = str(r.get("Question") or "").strip()
    ca = str(r.get("Correct Answer") or "").strip()
    inc = [str(r.get(f"Incorrect Answer {k}") or "").strip() for k in (1, 2, 3)]
    rid = str(r.get("Record ID") or "").strip()
    sub = str(r.get("Subdomain") or "").strip()
    if not q or not ca or any(not x for x in inc):
        bad.append((i, "empty-field")); continue
    if norm(ca) in {norm(x) for x in inc}:
        bad.append((i, "correct-in-incorrect")); continue
    # cross-check vs hendrydong mirror: find its problem containing this question
    qn = norm(q)
    matches = [j for j, p in enumerate(hd_problems) if len(qn) >= 40 and qn in p]
    if len(matches) > 1:
        starts = [j for j in matches if hd_problems[j].startswith(qn)]
        if len(starts) == 1:
            matches = starts
    if len(matches) != 1:
        bad.append((i, f"hd-align={len(matches)}")); continue
    j = matches[0]
    if hd_boxed[j] is None:
        bad.append((i, "hd-noboxed")); continue
    hbx = hd_boxed[j]
    if norm(ca) != hbx and not norm(ca).startswith(hbx) and not hbx.startswith(norm(ca)):
        bad.append((i, f"hd-disagree ca={norm(ca)[:40]!r} hd={hbx[:40]!r}")); continue
    rows.append({
        "Record ID": rid or f"diamond-{i:03d}",
        "Question": q,
        "Correct Answer": ca,
        "Incorrect Answer 1": inc[0],
        "Incorrect Answer 2": inc[1],
        "Incorrect Answer 3": inc[2],
        "Subdomain": sub,
    })

print(f"rows={len(rows)} bad={len(bad)}")
for b in bad[:10]:
    print("  BAD:", b)
if len(rows) != 198 or bad:
    sys.exit("BDYTX5 CROSS-CHECK FAILED — not writing cache")

csv_path = os.path.join(OUT, "gpqa_diamond.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["Record ID", "Question", "Correct Answer",
                                      "Incorrect Answer 1", "Incorrect Answer 2",
                                      "Incorrect Answer 3", "Subdomain"], quoting=csv.QUOTE_ALL)
    w.writeheader()
    w.writerows(rows)
print("wrote", csv_path, os.path.getsize(csv_path), "bytes")

# --- build hub cache ---
readme = open("/tmp/gpqa_readme.md").read()
sha = hashlib.sha256(readme.encode("utf-8") + open(csv_path, "rb").read()).hexdigest()[:40]
hub = os.path.expanduser("~/.cache/huggingface/hub/datasets--Idavidrein--gpqa")
snap = os.path.join(hub, "snapshots", sha)
os.makedirs(snap, exist_ok=True)
os.makedirs(os.path.join(hub, "refs"), exist_ok=True)
open(os.path.join(hub, "refs", "main"), "w").write(sha)
import shutil
shutil.copyfile(csv_path, os.path.join(snap, "gpqa_diamond.csv"))
shutil.copyfile("/tmp/gpqa_readme.md", os.path.join(snap, "README.md"))
print("cache:", snap)

# --- offline load test, exactly as lm-eval will see it ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
test = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
assert len(test) == 198, f"offline load rows={len(test)}"
assert all(c in test.column_names for c in ("Question", "Correct Answer",
           "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"))
print("OFFLINE LOAD OK: 198 rows, columns present")
json.dump({"source": "bdytx5/gpqa_gpqa_diamond (original schema)",
           "cross_check": "hendrydong/gpqa_diamond boxed text, 198/198 agree",
           "rows": len(rows), "csv_sha256": hashlib.sha256(open(csv_path, "rb").read()).hexdigest(),
           "cache_snapshot": snap},
          open(os.path.join(OUT, "reconstruct-receipt.json"), "w"), indent=1)
print("receipt written")
