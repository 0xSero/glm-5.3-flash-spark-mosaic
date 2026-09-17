#!/bin/bash
set -x
W=/home/valentine/mosaic-gatea
DST=/home/valentine/models/mosaic-10l-gatea
IMG=glm53-flash-sglang-exl3-plain:serve4
mkdir -p $W/receipts
docker run --rm -v /home/valentine/models:/home/valentine/models:ro -v $W:$W --entrypoint python3 $IMG \
  -m exl3_plain_sglang_overlay.census $DST --out $W/census-mosaic-10l-stock.json
echo "census exit=$?"
python3 - <<'PY'
import json
r = json.load(open('/home/valentine/mosaic-gatea/census-mosaic-10l-stock.json'))
e = r.get('experts', {})
v = r.get('verdict', {})
print(json.dumps({"tensors": r.get("tensors"), "total_size": r.get("total_size"), "codebook": r.get("codebook"),
  "bits_histogram": e.get("bits_histogram"),
  "layer3_bits": e.get("layer_bits", {}).get("3"), "layer20_bits": e.get("layer_bits", {}).get("20"),
  "layer44_bits": e.get("layer_bits", {}).get("44"),
  "n_problems": len(r.get("problems", [])), "problems": r.get("problems", [])[:4],
  "contract_ok": v.get("contract_ok"), "moe_sparkinfer_native": v.get("moe_sparkinfer_native"), "moe_reasons": v.get("moe_reasons")}, indent=1))
PY
echo "### CENSUS_10L_DONE $(date -u +%FT%TZ)"
