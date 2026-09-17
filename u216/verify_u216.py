#!/usr/bin/env python3
"""G2 pack-integrity check: every output tensor byte-compared against its source origin."""
import hashlib, json, mmap, re, struct, sys, time
from multiprocessing import Pool
from pathlib import Path
EXPERT = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mul1|mcg)$")
GATE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.(weight|e_score_correction_bias)$")
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "I16": 2, "I32": 4, "I8": 1, "U8": 1, "F64": 8, "I64": 8, "BOOL": 1, "F8_E4M3": 1}
def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; return json.loads(f.read(n)), 8 + n
def nbytes(m):
    n = DTYPE_BYTES[m["dtype"]]
    for d in m["shape"]: n *= d
    return n
def check_shard(args):
    src_dir, dst_dir, shard, plan = args
    keep = {int(l): sorted(v) for l, v in plan["retained_original_ids_by_layer"].items()}
    contig2orig = {l: {i: e for i, e in enumerate(v)} for l, v in keep.items()}
    sh, sb = read_header(Path(src_dir) / shard); dh, db = read_header(Path(dst_dir) / shard)
    problems = []; experts_seen = {}; n_ok = 0; bytes_ok = 0
    # contiguity
    ents = sorted((m["data_offsets"][0], m["data_offsets"][1], k) for k, m in dh.items() if k != "__metadata__")
    pos = 0
    for s, e, k in ents:
        if s != pos: problems.append(f"{shard}:{k} offset gap {pos}->{s}")
        pos = e
    size = (Path(dst_dir) / shard).stat().st_size
    if db + pos != size: problems.append(f"{shard}: data end {db+pos} != file size {size}")
    with open(Path(src_dir) / shard, "rb") as fs, mmap.mmap(fs.fileno(), 0, access=mmap.ACCESS_READ) as sm, \
         open(Path(dst_dir) / shard, "rb") as fd, mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ) as dm:
        for s, e, k in ents:
            m = dh[k]; out = dm[db + s: db + e]
            if e - s != nbytes(m): problems.append(f"{shard}:{k} size {e-s} != dtype*shape {nbytes(m)}")
            me = EXPERT.match(k); mg = GATE.match(k)
            if me and int(me.group(1)) in keep:
                layer, cid = int(me.group(1)), int(me.group(2))
                if cid not in contig2orig[layer]: problems.append(f"{shard}:{k} contiguous id {cid} out of range"); continue
                experts_seen.setdefault(layer, set()).add(cid)
                srcname = f"model.language_model.layers.{layer}.mlp.experts.{contig2orig[layer][cid]}.{me.group(3)}.{me.group(4)}"
                sm_ = sh.get(srcname)
                if sm_ is None: problems.append(f"{shard}:{k} source {srcname} not in shard"); continue
                if sm_["dtype"] != m["dtype"] or list(sm_["shape"]) != list(m["shape"]): problems.append(f"{shard}:{k} dtype/shape differ from source")
                a, b = sm_["data_offsets"]
                if sm[sb + a: sb + b] != out: problems.append(f"{shard}:{k} bytes differ from {srcname}"); continue
            elif mg and int(mg.group(1)) in keep:
                layer = int(mg.group(1)); ids = keep[layer]; sm_ = sh[k]; a, b = sm_["data_offsets"]; row = (b - a) // 288
                if list(m["shape"]) != [len(ids)] + list(sm_["shape"][1:]) or m["dtype"] != sm_["dtype"]: problems.append(f"{shard}:{k} router shape/dtype wrong {m['shape']}")
                exp = b"".join(sm[sb + a + i * row: sb + a + (i + 1) * row] for i in ids)
                if exp != out: problems.append(f"{shard}:{k} router rows differ from retained source rows"); continue
            else:
                sm_ = sh.get(k)
                if sm_ is None: problems.append(f"{shard}:{k} protected tensor missing from source"); continue
                a, b = sm_["data_offsets"]
                if sm_["dtype"] != m["dtype"] or list(sm_["shape"]) != list(m["shape"]) or sm[sb + a: sb + b] != out:
                    problems.append(f"{shard}:{k} protected tensor not byte-identical"); continue
            n_ok += 1; bytes_ok += e - s
        # every source tensor accounted for
        for k in sh:
            if k == "__metadata__": continue
            me = EXPERT.match(k)
            if me and int(me.group(1)) in keep:
                if int(me.group(2)) in keep[int(me.group(1))]:
                    cid = keep[int(me.group(1))].index(int(me.group(2)))
                    nk = f"model.language_model.layers.{me.group(1)}.mlp.experts.{cid}.{me.group(3)}.{me.group(4)}"
                    if nk not in dh: problems.append(f"{shard}: retained {k} missing in output as {nk}")
                # pruned original ids: their names may collide with renamed contiguous ids; the forward
                # check (every output expert tensor byte-equal to its mapped source) covers correctness.
            elif k not in dh: problems.append(f"{shard}: source tensor {k} missing in output")
    return {"shard": shard, "tensors_ok": n_ok, "bytes_ok": bytes_ok, "problems": problems,
            "experts_seen": {l: sorted(v) for l, v in experts_seen.items()},
            "sha256": hashlib.sha256(Path(dst_dir, shard).read_bytes()).hexdigest()}
def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2]); workers = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    plan = json.loads((dst / "plan.json").read_text()); man = json.loads((dst / "manifest.json").read_text())
    sidx = json.loads((src / "model.safetensors.index.json").read_text()); didx = json.loads((dst / "model.safetensors.index.json").read_text())
    t0 = time.time(); shards = sorted(set(sidx["weight_map"].values())); problems = []
    with Pool(workers) as pool: res = sorted(pool.map(check_shard, [(str(src), str(dst), s, plan) for s in shards]), key=lambda r: r["shard"])
    per_layer = {}
    for r in res:
        problems += r["problems"]
        for l, ids in r["experts_seen"].items(): per_layer.setdefault(int(l), set()).update(ids)
        m = next(x for x in man["shards"] if x["shard"] == r["shard"])
        if m["sha256"] != r["sha256"]: problems.append(f"{r['shard']}: manifest sha256 != on-disk sha256")
        print(f"  {r['shard']}: {r['tensors_ok']} tensors ok, {len(r['problems'])} problems [{time.time()-t0:.0f}s]", flush=True)
    keep = {int(l): v for l, v in plan["keep_by_layer"].items()}
    for l, k in keep.items():
        # count only full expert groups (12 fields)
        if per_layer.get(l) != set(range(k)): problems.append(f"layer {l}: contiguous ids {sorted(per_layer.get(l, []))[:5]}.. != range({k})")
    cnt = {}
    for n in didx["weight_map"]:
        me = EXPERT.match(n)
        if me and me.group(4) == "trellis" and me.group(3) == "gate_proj": cnt[int(me.group(1))] = cnt.get(int(me.group(1)), 0) + 1
    for l, k in keep.items():
        if cnt.get(l) != k: problems.append(f"layer {l}: {cnt.get(l)} gate_proj.trellis tensors != plan keep {k}")
    # byte model from headers: expert bytes and router row bytes measured on layer 3 of the source
    sh, _ = read_header(src / sidx["weight_map"]["model.language_model.layers.3.mlp.experts.0.gate_proj.trellis"])
    eb = sum(nbytes(sh[k]) for k in sh if k.startswith("model.language_model.layers.3.mlp.experts.0."))
    rb = (nbytes(sh["model.language_model.layers.3.mlp.gate.weight"]) + nbytes(sh["model.language_model.layers.3.mlp.gate.e_score_correction_bias"])) // 288
    extra_files = sorted(set(didx["weight_map"].values()) - set(sidx["weight_map"].values()))
    extra_bytes = 0; seal = {l.split("\t")[0]: l.split("\t")[2].strip() for l in (src / "files.tsv").read_text().splitlines() if l.count("\t") == 2 and l.split("\t")[2].strip()}
    for fn in extra_files:
        h, _ = read_header(dst / fn)
        for k, m in h.items():
            if k == "__metadata__": continue
            if didx["weight_map"].get(k) != fn: problems.append(f"{fn}: tensor {k} not mapped to it in index")
            extra_bytes += nbytes(m)
        sha = hashlib.sha256((dst / fn).read_bytes()).hexdigest()
        if seal.get(fn) != sha: problems.append(f"{fn}: sha256 {sha[:16]} != base seal {seal.get(fn, '?')[:16]}")
    expect = sidx["metadata"]["total_size"] + extra_bytes - plan["point"]["removed_experts"] * (eb + rb)
    if didx["metadata"]["total_size"] != expect: problems.append(f"index total_size {didx['metadata']['total_size']} != source - removed*(E+R) = {expect}")
    if sum(r["bytes_ok"] for r in res) + extra_bytes != didx["metadata"]["total_size"]: problems.append("sum of verified tensor bytes (+extra files) != index total_size")
    cfg = json.loads((dst / "config.json").read_text())["text_config"]
    if cfg.get("routed_experts_per_layer") != plan["text_config_patch"]["routed_experts_per_layer"]: problems.append("config routed_experts_per_layer != plan")
    if cfg.get("retained_expert_ids_by_layer") != plan["text_config_patch"]["retained_expert_ids_by_layer"]: problems.append("config retained_expert_ids_by_layer != plan")
    if cfg.get("n_routed_experts") != plan["text_config_patch"]["n_routed_experts"]: problems.append(f"n_routed_experts {cfg.get('n_routed_experts')} != plan {plan['text_config_patch']['n_routed_experts']}")
    receipt = {"schema": "dynamic-container-g2-v1", "point_id": plan["name"], "plan_sha256": plan["plan_sha256"], "pass": not problems,
               "expert_bytes_from_headers": eb, "router_row_bytes_from_headers": rb, "expected_index_total_size": expect, "extra_index_files": extra_files, "extra_index_tensor_bytes": extra_bytes,
               "index_total_size": didx["metadata"]["total_size"], "tensors_verified": sum(r["tensors_ok"] for r in res),
               "bytes_verified": sum(r["bytes_ok"] for r in res), "expert_count_by_layer": {str(l): cnt.get(l) for l in sorted(keep)},
               "problems": problems, "elapsed_s": round(time.time() - t0, 1), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (dst / "g2-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    man["g2_verified"] = receipt["pass"]; man["g2_receipt_sha256"] = hashlib.sha256((dst / "g2-receipt.json").read_bytes()).hexdigest()
    (dst / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "expert_count_by_layer"}, indent=1))
    sys.exit(0 if receipt["pass"] else 2)
if __name__ == "__main__": main()
