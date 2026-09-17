#!/usr/bin/env python3
"""Sum SGLang stat-mode recorder dumps (torch .pt with logical_count [layers, experts]) into a JSON routing table."""
import sys, json, glob, os, torch
dump_dir, receipt, out = sys.argv[1], sys.argv[2], sys.argv[3]
rec = json.load(open(receipt))
total = None; per_batch = {}
for b in rec["batches"]:
    d = torch.load(os.path.join(dump_dir, b["dump"]), map_location="cpu")
    lc = d["logical_count"].to(torch.int64)
    if lc.dim() == 3: lc = lc.sum(0)  # stat mode buffers one [layers, experts] slab per forward step
    per_batch[b["batch"]] = int(lc.sum())
    total = lc.clone() if total is None else total + lc
print("shape", tuple(total.shape), "total routes", int(total.sum()))
nz_layers = [i for i in range(total.shape[0]) if int(total[i].sum()) > 0]
print("layers with routes", nz_layers)
counts = {str(i): total[i].tolist() for i in nz_layers}
json.dump({"shape": list(total.shape), "layers_with_routes": nz_layers, "per_batch_routes": per_batch,
           "total_routes": int(total.sum()), "counts": counts, "receipt": receipt}, open(out, "w"))
print("wrote", out)
