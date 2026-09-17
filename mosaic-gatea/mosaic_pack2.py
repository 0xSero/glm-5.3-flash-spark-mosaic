#!/usr/bin/env python3
"""MOSAIC multi-layer packer (v2).

Rewrites every shard of the 2.05bpw dir that contains tensors of the plan's upgraded layers, swapping in
the 3.05bpw dir's byte-exact expert tensors for ALL experts of those layers. Patches
quantization_config.json's tensor_storage entries for the swapped modules. Verifies byte-exactness of
everything else. Pure python, seek/mmap, atomic replace per shard.

Plan schema: upgraded_expert_ids_by_layer: {"<layer>": [ids...], ...}
Exit 0 only if all checks pass.
"""
import argparse, hashlib, json, mmap, os, struct, sys, time
from pathlib import Path

CHUNK = 64 << 20
EXPERT_RE = None  # built per layer

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

def expert_of(name, layer):
    pref = f"model.language_model.layers.{layer}.mlp.experts."
    if not name.startswith(pref):
        return None
    return int(name[len(pref):].split(".")[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--plan", required=True)
    ap.add_argument("--receipt", required=True)
    a = ap.parse_args()
    t0 = time.time()
    plan = json.loads(Path(a.plan).read_text())
    up_by_layer = {int(l): set(int(x) for x in ids) for l, ids in plan["upgraded_expert_ids_by_layer"].items()}
    A, B, O = Path(a.a0), Path(a.base), Path(a.out)

    idx_a = json.loads((A / "model.safetensors.index.json").read_text())["weight_map"]
    idx_b = json.loads((B / "model.safetensors.index.json").read_text())["weight_map"]

    # which a0 shards hold upgraded tensors
    affected = {}
    for layer, ids in up_by_layer.items():
        for k, sh in idx_a.items():
            e = expert_of(k, layer)
            if e is not None and e in ids:
                affected.setdefault(sh, set()).add(layer)
    assert affected, "no affected shards"
    print(f"affected shards: {sorted(affected)}")

    hdr_b_by_shard = {}
    def base_loc(name):
        s = idx_b[name]
        if s not in hdr_b_by_shard:
            hdr_b_by_shard[s] = read_header(B / s)
        hdr, off = hdr_b_by_shard[s]
        m = hdr[name]
        return B / s, off + m["data_offsets"][0], off + m["data_offsets"][1], m

    shard_records = []
    total_swapped = total_kept = 0
    bad = []
    for shard in sorted(affected):
        layers_here = affected[shard]
        hdr_a, base_off_a = read_header(A / shard)
        entries = sorted(((m["data_offsets"][0], m["data_offsets"][1], name) for name, m in hdr_a.items() if name != "__metadata__"))
        out_meta = {}; ops = []; off = 0; swapped = 0; kept = 0; swapped_bytes = 0
        for s, e, name in entries:
            src_base = False
            for layer in layers_here:
                eid = expert_of(name, layer)
                if eid is not None and eid in up_by_layer[layer]:
                    src_base = eid is not None
                    break
            if src_base:
                assert name in idx_b, f"{name} missing in base"
                bp, bs, be, bm = base_loc(name)
                assert bm["dtype"] == hdr_a[name]["dtype"], (name, bm["dtype"], hdr_a[name]["dtype"])
                out_meta[name] = {"dtype": bm["dtype"], "shape": list(bm["shape"]), "data_offsets": [off, off + (be - bs)]}
                ops.append((bp, bs, be, name)); off += be - bs; swapped += 1; swapped_bytes += be - bs
            else:
                out_meta[name] = {"dtype": hdr_a[name]["dtype"], "shape": list(hdr_a[name]["shape"]), "data_offsets": [off, off + (e - s)]}
                ops.append((A / shard, base_off_a + s, base_off_a + e, name)); off += e - s; kept += 1
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
            for p, s, e, name in ops:
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
        assert dst.stat().st_size == 8 + len(hjson) + off

        # verify this shard
        hdr_new, off_new = read_header(dst)
        s_up_ok = s_keep_ok = 0
        for s, e, name in entries:
            lay = None
            for layer in layers_here:
                eid = expert_of(name, layer)
                if eid is not None and eid in up_by_layer[layer]:
                    lay = layer; break
            m = hdr_new[name]
            n_sha = sha_slice(dst, off_new + m["data_offsets"][0], off_new + m["data_offsets"][1])
            if lay is not None:
                bp, bs, be, bm = base_loc(name)
                ok = n_sha == sha_slice(bp, bs, be)
                s_up_ok += 1 if ok else 0
                if not ok: bad.append(f"UP:{name}")
            else:
                ok = n_sha == sha_slice(A / shard, base_off_a + s, base_off_a + e)
                s_keep_ok += 1 if ok else 0
                if not ok: bad.append(f"KEEP:{name}")
        rec = {"shard": shard, "layers": sorted(layers_here), "out_sha256": out_sha,
               "a0_sha256": sha_file(A / shard), "out_bytes": dst.stat().st_size,
               "delta_bytes": dst.stat().st_size - (A / shard).stat().st_size,
               "swapped_tensors": swapped, "kept_tensors": kept, "swapped_bytes": swapped_bytes}
        shard_records.append(rec)
        total_swapped += swapped; total_kept += kept
        print(f"  {shard}: swapped={swapped} kept={kept} delta={rec['delta_bytes']:,}B up_ok={s_up_ok} keep_ok={s_keep_ok}")

    # patch quantization_config.json
    cfg = json.loads((A / "quantization_config.json").read_text())
    cfg_b = json.loads((B / "quantization_config.json").read_text())
    ts, ts_b = cfg["tensor_storage"], cfg_b["tensor_storage"]
    patched = 0
    for layer, ids in sorted(up_by_layer.items()):
        for e in sorted(ids):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}"
                assert key in ts and key in ts_b, key
                assert ts[key]["stored_tensors"][key + ".trellis"]["shape"][-1] // 16 == 2, key
                assert ts_b[key]["stored_tensors"][key + ".trellis"]["shape"][-1] // 16 == 3, key
                ts[key] = ts_b[key]; patched += 1
    tmpc = (O / "quantization_config.json").with_suffix(".json.tmp")
    tmpc.write_text(json.dumps(cfg, indent=4) + "\n")
    os.replace(tmpc, O / "quantization_config.json")

    expected_delta = plan.get("expected_byte_delta")
    got_delta = sum(r["delta_bytes"] for r in shard_records)
    receipt = {
        "schema": "mosaic-pack-receipt-v2",
        "plan_sha256": plan.get("plan_sha256"), "variant": plan.get("variant"),
        "upgraded_layers": sorted(up_by_layer), "experts_per_layer": {str(l): len(v) for l, v in sorted(up_by_layer.items())},
        "shards": shard_records,
        "delta_bytes_total": got_delta, "expected_delta_bytes": expected_delta,
        "swapped_tensors": total_swapped, "kept_tensors": total_kept,
        "config_patched_entries": patched,
        "failures": bad[:20], "n_failures": len(bad),
        "wall_seconds": round(time.time() - t0, 1),
        "verdict": "PASS" if (not bad and (expected_delta is None or got_delta == expected_delta)) else "FAIL",
    }
    Path(a.receipt).write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"verdict": receipt["verdict"], "swapped": total_swapped, "kept": total_kept,
                      "delta": got_delta, "expected": expected_delta, "patched": patched,
                      "n_failures": len(bad), "wall_s": receipt["wall_seconds"]}, indent=1))
    sys.exit(0 if receipt["verdict"] == "PASS" else 2)

if __name__ == "__main__":
    main()