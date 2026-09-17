#!/bin/bash
# MOSAIC Gate A census (spark-557f). Stock census via the same overlay module the serving image uses.
set -x
W=/home/valentine/mosaic-gatea
DST=/home/valentine/models/mosaic-l20-gatea
IMG=glm53-flash-sglang-exl3-plain:serve4
mkdir -p $W/receipts
docker run --rm -v /home/valentine/models:/home/valentine/models:ro -v $W:$W --entrypoint python3 $IMG \
  -m exl3_plain_sglang_overlay.census $DST --out $W/census-mosaic-gatea-stock.json
echo "census exit=$?"
python3 - <<'PY'
import json
r = json.load(open('/home/valentine/mosaic-gatea/census-mosaic-gatea-stock.json'))
e = r.get('experts', {})
print(json.dumps({
    "schema": r.get("schema"), "tensors": r.get("tensors"), "total_size": r.get("total_size"),
    "codebook": r.get("codebook"), "expert_layers": len(e.get("layers", [])),
    "bits_histogram": e.get("bits_histogram"),
    "layer20_bits": e.get("layer_bits", {}).get("20"),
    "problems": r.get("problems", [])[:6], "n_problems": len(r.get("problems", [])),
    "verdict": r.get("verdict", {}).get("contract_ok"),
    "moe_sparkinfer_native": r.get("verdict", {}).get("moe_sparkinfer_native"),
    "moe_reasons": r.get("verdict", {}).get("moe_reasons"),
}, indent=1))
PY
echo "### CENSUS_DONE $(date -u +%FT%TZ)"