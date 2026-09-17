#!/usr/bin/env python3
"""Add the tensors of extra safetensors files (mtp.safetensors, kpool_aux.safetensors) to a
model.safetensors.index.json so the loader/census see layers 3..45 and the kpool tensors, as the
2.05 branch's index does natively.  Metadata-only: no tensor bytes are touched.
Usage: augment_index.py <dir> <out_index_path> [extra files...]   (writes JSON; reports bytes added)
"""
import json, struct, sys
from pathlib import Path
DT = {"F16": 2, "BF16": 2, "F32": 4, "I16": 2, "I32": 4, "I8": 1, "U8": 1, "F64": 8, "I64": 8, "BOOL": 1}
def header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; return json.loads(f.read(n))
d = Path(sys.argv[1]); out = Path(sys.argv[2]); extras = sys.argv[3:] or ["mtp.safetensors", "kpool_aux.safetensors"]
idx = json.loads((d / "model.safetensors.index.json").read_text()); wm = idx["weight_map"]; added = {}
for fn in extras:
    h = header(d / fn); n = 0
    for k, m in h.items():
        if k == "__metadata__": continue
        assert k not in wm, f"{k} already in index"
        wm[k] = fn; b = DT[m["dtype"]]
        for s in m["shape"]: b *= s
        n += b
    added[fn] = {"tensors": len(h) - ("__metadata__" in h), "tensor_bytes": n}
    idx["metadata"]["total_size"] += n
idx["weight_map"] = dict(sorted(wm.items()))
out.write_text(json.dumps(idx, indent=2) + "\n")
print(json.dumps({"index": str(out), "total_size": idx["metadata"]["total_size"], "added": added}))
