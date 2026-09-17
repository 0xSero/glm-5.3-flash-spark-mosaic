#!/bin/bash
set -x
W=/home/valentine/mosaic-gatea
DST=/home/valentine/models/mosaic-10l-gatea
IMG=glm53-flash-sglang-exl3-plain:serve4
CEN=$W/receipts/exl3-plain-census-10l.json
mkdir -p $W/receipts
cp -f $W/census-mosaic-10l-stock.json $CEN 2>/dev/null
test -s $CEN || { echo CENSUS_RECEIPT_MISSING_ABORT; exit 1; }
python3 -c "import json; r=json.load(open('$CEN')); assert r['schema']=='exl3-plain-census-v1' and r['index_only'] is False and r['verdict']['contract_ok'] is True, 'census receipt check failed'; print('receipt_ok tensors=%s'%r['tensors'])" || { echo RECEIPT_CHECK_ABORT; exit 1; }
echo "STOP own container glm53-mosaic-12lh (12L server idle; mosaic GPQA leg exited=0 2026-09-17T00:24:02Z) $(date -u +%FT%TZ)"
docker rm -f glm53-mosaic-12lh 2>/dev/null
docker rm -f glm53-mosaic-10l 2>/dev/null
docker run -d --name glm53-mosaic-10l \
  -e EXL3_PLAIN_CENSUS=/receipts/exl3-plain-census-10l.json \
  -e EXL3_PLAIN_DECODER=exl3_plain_sglang_overlay.exl3_reference:decode \
  -e SGLANG_EXL3_MAX_BATCH_TOKENS=256 \
  -v $DST:/model:ro -v $W/receipts:/receipts:ro \
  -p 8000:8000 --gpus all --shm-size=16g \
  --entrypoint python3 $IMG \
  -m sglang.launch_server --model-path /model --host 0.0.0.0 --port 8000 \
  --quantization exl3 --tp-size 1 --ep-size 1 --context-length 262144 --kv-cache-dtype fp8_e4m3 \
  --attention-backend dsa --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla \
  --linear-attn-backend triton --disable-shared-experts-fusion --chunked-prefill-size 256 \
  --max-prefill-tokens 256 --max-running-requests 1 --mem-fraction-static 0.90 --enable-multimodal \
  --chat-template /opt/glm53/chat-template-mm.jinja --reasoning-parser glm45 --tool-call-parser glm47 \
  --disable-cuda-graph
echo "serve_launch_exit=$?"
echo "### SERVE_10L_LAUNCHED $(date -u +%FT%TZ)"
