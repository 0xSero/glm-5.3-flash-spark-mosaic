#!/usr/bin/env python3
"""Combine the sealed S216 capture (prefill-000..006 + generate-000-partial, dumps still on disk)
with the new R216 generation batches, and reduce ALL dumps into routing-counts-r216.json.
Mirrors reduce_routing.py exactly (stat-mode dumps: logical_count [layers, experts], one slab per
forward step → sum over the step axis). Full recomputation from dumps preserves provenance:
every batch's dump file is re-read; nothing is trusted from the old counts beyond the batch→dump
mapping in the two receipts.
"""
import json, os, sys, time, glob, hashlib, torch
dump_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/sero/w2port/routing"
old_receipt = json.load(open(os.path.join(dump_dir, "capture-receipt.json")))
new_receipt = json.load(open(os.path.join(dump_dir, "r216-gen-capture.json")))
assert new_receipt.get("finished_utc"), "new capture incomplete; refusing to combine"
old_batches = [b for b in old_receipt["batches"]]           # prefill-000..006 + generate-000-partial
new_batches = [b for b in new_receipt["batches"]]
names = [b["batch"] for b in old_batches + new_batches]
assert len(names) == len(set(names)), "duplicate batch names"
total = None; per_batch = {}
for b in old_batches + new_batches:
    p = os.path.join(dump_dir, b["dump"])
    assert os.path.exists(p), f"missing dump {p}"
    d = torch.load(p, map_location="cpu")
    lc = d["logical_count"].to(torch.int64)
    if lc.dim() == 3: lc = lc.sum(0)
    per_batch[b["batch"]] = int(lc.sum())
    total = lc.clone() if total is None else total + lc
nz = [i for i in range(total.shape[0]) if int(total[i].sum()) > 0]
counts = {str(i): total[i].tolist() for i in nz}
prov = {
    "sealed_s216_counts_sha256": hashlib.sha256(open(os.path.join(dump_dir, "routing-counts.json"), "rb").read()).hexdigest(),
    "sealed_capture_receipt_sha256": hashlib.sha256(open(os.path.join(dump_dir, "capture-receipt.json"), "rb").read()).hexdigest(),
    "new_capture_receipt_sha256": hashlib.sha256(open(os.path.join(dump_dir, "r216-gen-capture.json"), "rb").read()).hexdigest(),
    "old_batches": [b["batch"] for b in old_batches],
    "new_batches": [b["batch"] for b in new_batches],
    "new_completion_tokens": new_receipt["total_completion_tokens"],
    "note": "routing-counts-r216 = same prefill dumps as the sealed S216 counts + 15 further generation prompts (2048-token cap each); T216 proved the route factor carries signal, this densifies it",
}
out = {"shape": list(total.shape), "layers_with_routes": nz, "per_batch_routes": per_batch,
       "total_routes": int(total.sum()), "counts": counts,
       "receipt": {"old": "capture-receipt.json", "new": "r216-gen-capture.json"}, "provenance": prov}
dst = os.path.join(dump_dir, "routing-counts-r216.json")
json.dump(out, open(dst, "w"))
print(json.dumps({"file": dst, "sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest(),
                  "total_routes": out["total_routes"], "layers": nz[:3], "n_layers": len(nz),
                  "new_completion_tokens": prov["new_completion_tokens"],
                  "sealed_total_routes_for_reference": None}, indent=1))
