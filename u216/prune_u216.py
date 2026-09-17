#!/usr/bin/env python3
"""Pack a pruned EXL3 artifact from a per-expert-tensor safetensors base (pure python, CPU, mmap).

For the 42 target layers: retained experts' 12 tensors are byte-copied and renamed to
contiguous ids; mlp.gate.weight rows and e_score_correction_bias entries are gathered at
the retained original ids; everything else is copied byte-exact. Shard partition follows
the source (one output shard per source shard). Index and config are regenerated.
"""
import argparse, hashlib, json, mmap, os, re, shutil, struct, sys, time
from multiprocessing import Pool
from pathlib import Path

PACKER_VERSION = "u216-packer-v1"
EXPERT = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mul1|mcg)$")
GATE_W = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
GATE_B = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$")
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "I16": 2, "I32": 4, "I8": 1, "U8": 1, "F64": 8, "I64": 8, "BOOL": 1, "F8_E4M3": 1}
CHUNK = 64 << 20
SKIP_FILES = {".pull-state", "pull.log", "pull3.sh", "pullloop.sh", "pullpar.sh", "files.tsv", "DOWNLOAD-SEAL.json", ".gitattributes"}

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def nbytes(meta):
    n = DTYPE_BYTES[meta["dtype"]]
    for d in meta["shape"]: n *= d
    return n

def pack_shard(args):
    src_dir, dst_dir, shard, plan = args
    keep = {int(l): set(v) for l, v in plan["retained_original_ids_by_layer"].items()}
    remap = {int(l): {int(k): v for k, v in m.items()} for l, m in plan["original_id_to_contiguous_id"].items()}
    hdr, base = read_header(Path(src_dir) / shard)
    entries = sorted(((m["data_offsets"][0], m["data_offsets"][1], name) for name, m in hdr.items() if name != "__metadata__"))
    out_meta = {}; ops = []; off = 0; removed_experts = 0; renamed = 0; sliced = 0; copied = 0
    for s, e, name in entries:
        m = hdr[name]; assert e - s == nbytes(m), (name, m, e - s)
        me = EXPERT.match(name)
        if me:
            layer, eid = int(me.group(1)), int(me.group(2))
            if layer in keep:
                if eid not in keep[layer]:
                    removed_experts += 1; continue
                new = f"model.language_model.layers.{layer}.mlp.experts.{remap[layer][eid]}.{me.group(3)}.{me.group(4)}"
                renamed += 1
            else:
                new = name; copied += 1
            out_meta[new] = {"dtype": m["dtype"], "shape": list(m["shape"]), "data_offsets": [off, off + (e - s)]}
            ops.append(("copy", base + s, base + e, new)); off += e - s; continue
        mw = GATE_W.match(name); mb = GATE_B.match(name)
        if (mw or mb) and int((mw or mb).group(1)) in keep:
            layer = int((mw or mb).group(1)); ids = sorted(keep[layer])
            assert m["shape"][0] == 288, (name, m["shape"])
            row = nbytes(m) // 288
            shape = [len(ids)] + list(m["shape"][1:])
            out_meta[name] = {"dtype": m["dtype"], "shape": shape, "data_offsets": [off, off + row * len(ids)]}
            ops.append(("gather", base + s, row, ids, name)); off += row * len(ids); sliced += 1; continue
        out_meta[name] = {"dtype": m["dtype"], "shape": list(m["shape"]), "data_offsets": [off, off + (e - s)]}
        ops.append(("copy", base + s, base + e, name)); off += e - s; copied += 1
    if "__metadata__" in hdr: out_meta = {"__metadata__": hdr["__metadata__"], **out_meta}
    hjson = json.dumps(out_meta, separators=(",", ":")).encode()
    hjson += b" " * ((8 - len(hjson) % 8) % 8)
    dst = Path(dst_dir) / shard; tmp = dst.with_suffix(".safetensors.tmp")
    h = hashlib.sha256(); written = 0
    with open(Path(src_dir) / shard, "rb") as fin, mmap.mmap(fin.fileno(), 0, access=mmap.ACCESS_READ) as mm, open(tmp, "wb") as fout:
        pre = struct.pack("<Q", len(hjson)) + hjson; fout.write(pre); h.update(pre)
        for op in ops:
            if op[0] == "copy":
                _, a, b, _ = op
                while a < b:
                    c = mm[a:min(b, a + CHUNK)]; fout.write(c); h.update(c); written += len(c); a += len(c)
            else:
                _, a, row, ids, _ = op
                buf = b"".join(mm[a + i * row: a + (i + 1) * row] for i in ids)
                fout.write(buf); h.update(buf); written += len(buf)
    assert written == off, (shard, written, off)
    os.replace(tmp, dst)
    size = dst.stat().st_size
    return {"shard": shard, "bytes": size, "tensor_bytes": off, "header_bytes": 8 + len(hjson), "sha256": h.hexdigest(),
            "tensors": len(out_meta) - ("__metadata__" in out_meta), "removed_expert_tensors": removed_experts,
            "renamed_expert_tensors": renamed, "sliced_router_tensors": sliced, "copied_tensors": copied,
            "weight_map": {k: shard for k in out_meta if k != "__metadata__"}}

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(CHUNK), b""): h.update(c)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); ap.add_argument("--plan", required=True)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time(); src = Path(a.src); dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    plan_bytes = Path(a.plan).read_bytes(); plan = json.loads(plan_bytes)
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    shards = sorted(set(idx["weight_map"].values()))
    print(f"packing {len(shards)} shards with {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        results = []
        for r in pool.imap_unordered(pack_shard, [(str(src), str(dst), s, plan) for s in shards]):
            print(f"  {r['shard']}: {r['bytes']:,} B sha256={r['sha256'][:16]} removed={r['removed_expert_tensors']} renamed={r['renamed_expert_tensors']} sliced={r['sliced_router_tensors']} [{time.time()-t0:.0f}s]", flush=True)
            results.append(r)
    results.sort(key=lambda r: r["shard"])
    weight_map = {}
    for r in results: weight_map.update(r.pop("weight_map"))
    total = sum(r["tensor_bytes"] for r in results)
    new_idx = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (dst / "model.safetensors.index.json").write_text(json.dumps(new_idx, indent=2) + "\n")
    # config patch
    cfg = json.loads((src / "config.json").read_text())
    tc = cfg["text_config"]
    tc["routed_experts_per_layer"] = plan["text_config_patch"]["routed_experts_per_layer"]
    tc["retained_expert_ids_by_layer"] = plan["text_config_patch"]["retained_expert_ids_by_layer"]
    tc["n_routed_experts"] = plan["text_config_patch"]["n_routed_experts"]
    tc["dynamic_container"] = {"point_id": plan["name"], "plan_sha256": plan["plan_sha256"], "status": "derived", "packer_version": PACKER_VERSION,
                               "uniform_keep": plan["point"].get("uniform_keep"), "mtp_layer": plan.get("mtp_layer")}
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    # verbatim copies
    copied = {}
    for p in sorted(src.iterdir()):
        if p.name in SKIP_FILES or p.name.startswith("model-") or p.name in ("model.safetensors.index.json", "config.json") or p.suffix == ".part" or p.is_dir():
            continue
        shutil.copyfile(p, dst / p.name); copied[p.name] = {"bytes": p.stat().st_size, "sha256": sha_file(dst / p.name)}
        print(f"  copied {p.name} {copied[p.name]['bytes']:,} B", flush=True)
    shutil.copyfile(a.plan, dst / "plan.json"); os.chmod(dst / "plan.json", 0o444)
    removed = plan["point"]["removed_experts"]
    manifest = {
        "schema": "dynamic-container-manifest-v1", "point_id": plan["name"], "packer_version": PACKER_VERSION,
        "packer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_sha256": plan["plan_sha256"], "plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "base": plan["base"] | {"source_dir": str(src), "index_total_size": idx["metadata"]["total_size"]},
        "output_dir": str(dst), "shards": results, "index_total_size": total,
        "removed_experts": removed, "removed_tensor_bytes": idx["metadata"]["total_size"] - total,
        "bytes_per_removed_expert_incl_router_row": (idx["metadata"]["total_size"] - total) / removed,
        "other_files": copied, "config_patch_keys": ["text_config.n_routed_experts", "text_config.routed_experts_per_layer", "text_config.retained_expert_ids_by_layer", "text_config.dynamic_container"],
        "total_output_bytes": sum(r["bytes"] for r in results) + sum(c["bytes"] for c in copied.values()) + (dst / "config.json").stat().st_size + (dst / "model.safetensors.index.json").stat().st_size,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1), "g2_verified": False,
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("shards", "other_files")}, indent=1))
if __name__ == "__main__": main()
