#!/usr/bin/env python3
"""Run the stock exl3_plain census on a pruned artifact and separate the expected per-layer expert
count deviations (from plan.json) from any residual contract problem.  Writes both.
Usage: census_plan_aware.py <model_dir> <out.json>
"""
import json, re, sys
from pathlib import Path
from exl3_plain_sglang_overlay import census as C
model = Path(sys.argv[1]); out = Path(sys.argv[2])
plan = json.loads((model / "plan.json").read_text()); keep = {int(l): v for l, v in plan["keep_by_layer"].items()}
r = C.census(model / "model.safetensors.index.json", index_only=False)
pat = re.compile(r"^layer (\d+): (\d+) experts, expected (\d+)$")
expected, residual = [], []
for p in r["problems"]:
    m = pat.match(p)
    if m and int(m.group(1)) in keep and int(m.group(2)) == keep[int(m.group(1))]: expected.append(p)
    else: residual.append(p)
# every planned layer must have produced exactly its expected deviation (or 288 == no deviation)
missing = [l for l, k in keep.items() if k != 288 and not any(f"layer {l}: {k} experts" in p for p in expected)]
r["plan_sha256"] = plan["plan_sha256"]; r["problems_expected_by_plan"] = expected; r["problems_residual"] = residual
r["plan_layers_missing_expected_count"] = missing
r["verdict"]["contract_ok_stock"] = r["verdict"]["contract_ok"]
r["verdict"]["contract_ok_plan_aware"] = not residual and not missing
out.write_text(json.dumps(r, indent=1))
v = r["verdict"]; e = r["experts"]
print(f"census(plan-aware): {r['tensors']} tensors, {len(r['shards'])} shards, codebook={r['codebook']}, expert layers={e['layers'][0]}..{e['layers'][-1]} ({len(e['layers'])}), expert bits={e['bits_histogram']}")
print(f"  stock contract_ok={v['contract_ok_stock']} expected-by-plan problems={len(expected)} residual={residual} missing={missing}")
print(f"  plan-aware contract_ok={v['contract_ok_plan_aware']} moe native={v['moe_sparkinfer_native']} {v['moe_reasons']} dense native={sum(v['dense_sparkinfer_native'].values())}/{len(v['dense_sparkinfer_native'])}")
sys.exit(0 if v["contract_ok_plan_aware"] else 2)
