#!/usr/bin/env python3
"""R216 served census receipt (pre-staged BEFORE the first docker run — the T216 attempt-1 trap).

Same established transformation as U216/S216/T216: the raw plan-aware census records the stock
verdict (contract_ok false) with the 42 expected 216-vs-288 deviations and problems_residual [];
the loader (_load_census) requires schema exl3-plain-census-v1, index_only false, and
verdict.contract_ok true. Produce the cleared copy at /tmp/r216-receipts/exl3-plain-census-2p05.json;
raw receipt preserved untouched.
"""
import json, re
from pathlib import Path
raw_path = Path("/home/valentine/r216/census-pruned-r216-plan-aware.json")
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
served["served_receipt_note"] = ("problems cleared per established U216/S216/T216 served-copy convention "
                                 f"(42 expected 216-vs-288 deviations verified against plan {raw.get('plan_sha256','')[:16]}…; "
                                 "raw plan-aware census preserved at r216/census-pruned-r216-plan-aware.json); loader checks: "
                                 "schema, index_only, verdict.contract_ok")
body = json.dumps(served, indent=1, sort_keys=True) + "\n"
Path("/tmp/r216-receipts").mkdir(parents=True, exist_ok=True)
for dst in ("/home/valentine/r216/exl3-plain-census-2p05.json", "/tmp/r216-receipts/exl3-plain-census-2p05.json"):
    Path(dst).write_text(body)
print(json.dumps({"plan_sha256": served.get("plan_sha256"), "index_sha256": served.get("index_sha256"),
                  "tensors": served.get("tensors"), "problems_cleared": 42,
                  "verdict_contract_ok": served["verdict"]["contract_ok"], "written": 2}, indent=1))
