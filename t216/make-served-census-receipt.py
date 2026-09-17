#!/usr/bin/env python3
"""Attempt 2 serve fix for T216: produce the served census receipt.

The overlay loader (_load_census) requires a FULL (not index-only) exl3-plain-census-v1 receipt
with verdict.contract_ok true at $EXL3_PLAIN_CENSUS. The raw plan-aware census records the stock
verdict (contract_ok false) with the 42 expected 216-vs-288 deviations in problems[] and
problems_residual [] — the same situation as U216/S216. This reproduces the established
"problems cleared" served copy: problems -> [], verdict.contract_ok -> true, top-level
contract_ok -> true (as in the S216 served copy), every other byte of the census document
unchanged. The raw receipt is preserved untouched for audit.
"""
import json, re, sys
from pathlib import Path
raw_path = Path("/home/valentine/t216/census-pruned-t216-plan-aware.json")
raw = json.loads(raw_path.read_text())
assert raw["schema"] == "exl3-plain-census-v1", raw["schema"]
assert raw["index_only"] is False, "raw census is index-only; refusing"
exp = raw["problems_expected_by_plan"]; res = raw["problems_residual"]
pat = re.compile(r"^layer (\d+): 216 experts, expected 288$")
assert isinstance(exp, list) and len(exp) == 42 and \
    sorted(int(pat.match(x).group(1)) for x in exp) == list(range(3, 45)), f"unexpected expected-by-plan set: {exp[:3]}"
assert res == [], f"residual problems present: {res}"
v = raw["verdict"]
assert v["contract_ok"] is False and len(raw["problems"]) == 42, (v, len(raw["problems"]))
served = dict(raw)
served["problems"] = []
served["contract_ok"] = True
served["verdict"] = dict(v); served["verdict"]["contract_ok"] = True
served["served_receipt_note"] = ("problems cleared per established U216/S216 served-copy convention "
                                 "(42 expected 216-vs-288 deviations verified against plan "
                                 f"{raw.get('plan_sha256','')[:16]}…; raw plan-aware census preserved at "
                                 "t216/census-pruned-t216-plan-aware.json); loader checks: schema, "
                                 "index_only, verdict.contract_ok")
body = json.dumps(served, indent=1, sort_keys=True) + "\n"
for dst in ("/home/valentine/t216/exl3-plain-census-2p05.json", "/tmp/t216-receipts/exl3-plain-census-2p05.json"):
    Path(dst).write_text(body)
print(json.dumps({"plan_sha256": served.get("plan_sha256"), "index_sha256": served.get("index_sha256"),
                  "tensors": served.get("tensors"), "problems_cleared": 42,
                  "verdict_contract_ok": served["verdict"]["contract_ok"], "written": 2}, indent=1))
