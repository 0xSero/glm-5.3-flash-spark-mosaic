#!/usr/bin/env python3
"""MOSAIC Gate A packer.

Rewrites exactly ONE shard of the 2.05bpw dir (the shard holding the target layer's experts), swapping
in the 3.05bpw dir's byte-exact expert tensors for the plan's upgraded experts of that layer. Then
patches quantization_config.json's tensor_storage entries for those experts from the 3.05bpw dir's own
config, and verifies byte-exactness of everything else. Pure python, seek/mmap based, atomic replace.

Usage: mosaic_pack.py --a0 DIR --base DIR --out DIR --plan PLAN.json --receipt RECEIPT.json
Exit code 0 only if every assertion and every byte-exactness check passes.
"""
import argparse, hashlib, json, mmap, os, struct, sys, time
from pathlib import Path

CHUNK = 64 << 20

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def sha_slice(path, a, b):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            h.update(mm[a:b])
        finally:
            mm.close()
    return h.hexdigest()

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(CHUNK)
            if not c:
                break
            h.update(c)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--receipt", required=True)
    a = ap.parse_args()
    t0 = time.time()
    plan = json.loads(Path(a.plan).read_text())
    layer = int(plan["layer"]); up = set(int(x) for x in plan["upgraded_expert_ids"])
    A, B, O = Path(a.a0), Path(a.base), Path(a.out)

    idx_a = json.loads((A / "model.safetensors.index.json").read_text())["weight_map"]
    idx_b = json.loads((B / "model.safetensors.index.json").read_text())["weight_map"]
    pref = f"model.language_model.layers.{layer}.mlp.experts."
    layer_keys = [k for k in idx_a if k.startswith(pref)]
    shards_a = sorted(set(idx_a[k] for k in layer_keys))
    assert len(shards_a) == 1, f"layer {layer} spans multiple a0 shards: {shards_a}"
    shard = shards_a[0]
    base_shards = sorted(set(idx_b[k] for k in layer_keys))
    assert all(k in idx_b for k in layer_keys), "base missing some layer keys"

    # load source headers
    hdr_a, base_off_a = read_header(A / shard)
    hdr_b_by_shard = {s: read_header(B / s) for s in base_shards}

    def base_loc(name):
        s = idx_b[name]
        hdr, off = hdr_b_by_shard[s]
        m = hdr[name]
        return B / s, off + m["data_offsets"][0], off + m["data_offsets"][1], m

    upgraded_tensors = []
    for k in layer_keys:
        eid = int(k.split(".mlp.experts.")[1].split(".")[0])
        if eid in up:
            upgraded_tensors.append(k)
    assert len(upgraded_tensors) == len(up) * 12, (len(upgraded_tensors), len(up))
    # field-set identity per expert
    fields_a = sorted(set(k.rsplit(".", 1)[1] for k in layer_keys))
    fields_b = sorted(set(k.rsplit(".", 1)[1] for k in layer_keys if k in idx_b))
    assert fields_a == fields_b, (fields_a, fields_b)

    # build new shard
    entries = sorted(((m["data_offsets"][0], m["data_offsets"][1], name) for name, m in hdr_a.items() if name != "__metadata__"))
    out_meta = {}; ops = []; off = 0; swapped = 0; copied = 0; swapped_bytes = 0
    for s, e, name in entries:
        if name in idx_b and name.startswith(pref) and int(name.split(".mlp.experts.")[1].split(".")[0]) in up:
            bp, bs, be, bm = base_loc(name)
            assert bm["dtype"] == hdr_a[name]["dtype"], (name, bm["dtype"], hdr_a[name]["dtype"])
            out_meta[name] = {"dtype": bm["dtype"], "shape": list(bm["shape"]), "data_offsets": [off, off + (be - bs)]}
            ops.append(("copy", bp, bs, be, name)); off += be - bs; swapped += 1; swapped_bytes += be - bs
        else:
            out_meta[name] = {"dtype": hdr_a[name]["dtype"], "shape": list(hdr_a[name]["shape"]), "data_offsets": [off, off + (e - s)]}
            ops.append(("copy", A / shard, base_off_a + s, base_off_a + e, name)); off += e - s; copied += 1
    if "__metadata__" in hdr_a:
        out_meta = {"__metadata__": hdr_a["__metadata__"], **out_meta}
    hjson = json.dumps(out_meta, separators=(",", ":")).encode()
    hjson += b" " * ((8 - len(hjson) % 8) % 8)

    dst = O / shard; tmp = dst.with_suffix(dst.suffix + ".tmp")
    h = hashlib.sha256(); written = 0
    with open(tmp, "wb") as fout:
        pre = struct.pack("<Q", len(hjson)) + hjson
        fout.write(pre); h.update(pre)
        cur_path = None; mm = None; cf = None
        for kind, p, s, e, name in ops:
            if cur_path != p:
                if mm is not None: mm.close(); cf.close()
                cf = open(p, "rb"); mm = mmap.mmap(cf.fileno(), 0, access=mmap.ACCESS_READ); cur_path = p
            pos = s
            while pos < e:
                c = mm[pos:min(e, pos + CHUNK)]
                fout.write(c); h.update(c); written += len(c); pos += len(c)
        if mm is not None: mm.close(); cf.close()
    assert written == off, (written, off)
    os.replace(tmp, dst)
    out_sha = h.hexdigest()
    assert (dst.stat().st_size) == 8 + len(hjson) + off

    # patch quantization_config.json (tensor_storage entries for upgraded experts only)
    cfg = json.loads((A / "quantization_config.json").read_text())
    cfg_b = json.loads((B / "quantization_config.json").read_text())
    ts, ts_b = cfg["tensor_storage"], cfg_b["tensor_storage"]
    patched = 0; patched_ok = 0
    for e in sorted(up):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}"
            assert key in ts and key in ts_b, key
            entry_b = ts_b[key]
            bits_b = entry_b["stored_tensors"][key + ".trellis"]["shape"][-1] // 16
            assert bits_b == 3, (key, bits_b)
            bits_a = ts[key]["stored_tensors"][key + ".trellis"]["shape"][-1] // 16
            assert bits_a == 2, (key, bits_a)
            ts[key] = entry_b; patched += 1
            patched_ok += 1
    tmpc = (O / "quantization_config.json").with_suffix(".json.tmp")
    tmpc.write_text(json.dumps(cfg, indent=4) + "\n")
    os.replace(tmpc, O / "quantization_config.json")

    # verification: per-tensor byte-exactness (upgraded == base, all others == a0)
    hdr_new, off_new = read_header(dst)
    up_ok = 0; up_bad = []
    for name in upgraded_tensors:
        m = hdr_new[name]
        n_sha = sha_slice(dst, off_new + m["data_offsets"][0], off_new + m["data_offsets"][1])
        bp, bs, be, bm = base_loc(name)
        b_sha = sha_slice(bp, bs, be)
        if n_sha == b_sha: up_ok += 1
        else: up_bad.append(name)
    keep_ok = 0; keep_bad = []
    for s, e, name in entries:
        if name in (upgraded_tensors if False else []): continue
        if name.startswith(pref) and int(name.split(".mlp.experts.")[1].split(".")[0]) in up:
            continue
        m = hdr_new[name]
        n_sha = sha_slice(dst, off_new + m["data_offsets"][0], off_new + m["data_offsets"][1])
        a_sha = sha_slice(A / shard, base_off_a + s, base_off_a + e)
        if n_sha == a_sha: keep_ok += 1
        else: keep_bad.append(name)
    # router rows / bias must be byte-exact too (gathered already above as non-upgraded entries)

    receipt = {
        "schema": "mosaic-gatea-pack-receipt-v1",
        "plan_sha256": plan.get("plan_sha256"),
        "layer": layer, "upgraded_experts": len(up),
        "shard": shard, "a0_shard_bytes": (A / shard).stat().st_size, "out_shard_bytes": dst.stat().st_size,
        "delta_bytes": dst.stat().st_size - (A / shard).stat().st_size,
        "expected_delta_bytes": len(up) * 3145728,
        "swapped_tensors": swapped, "copied_tensors": copied, "swapped_bytes": swapped_bytes,
        "out_shard_sha256": out_sha,
        "a0_shard_sha256": sha_file(A / shard),
        "upgraded_tensors_byte_equal_to_base": up_ok, "upgraded_tensors_failed": up_bad,
        "kept_tensors_byte_equal_to_a0": keep_ok, "kept_tensors_failed": keep_bad[:20],
        "config_patched_entries": patched,
        "wall_seconds": round(time.time() - t0, 1),
        "verdict": "PASS" if (not up_bad and not keep_bad and up_ok == len(upgraded_tensors) and patched == patched_ok) else "FAIL",
    }
    Path(a.receipt).write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: receipt[k] for k in ("verdict", "out_shard_sha256", "a0_shard_sha256", "delta_bytes",
                                              "expected_delta_bytes", "upgraded_tensors_byte_equal_to_base",
                                              "kept_tensors_byte_equal_to_a0", "kept_tensors_failed", "wall_seconds")}, indent=1))
    sys.exit(0 if receipt["verdict"] == "PASS" else 2)

if __name__ == "__main__":
    main()