#!/bin/bash
set -x
W=/home/valentine/e216
DST=/home/valentine/models/glm53-3p05-pruned-e216
echo "### pre-stage served census receipt BEFORE launch (T216 attempt-1 trap)"
python3 $W/make-served-census-receipt-e216.py; echo "receipt exit=$?" || exit 1
echo "### record r216 serving state $(date -u +%FT%TZ)"
docker logs glm53-r216 > $W/r216-serve-final.log 2>&1; echo "log exit=$?"
docker inspect glm53-r216 > $W/r216-container-inspect-pre-stop.json; echo "inspect exit=$?"
echo "### stop glm53-r216 (docker stop; container preserved, not removed)"
docker stop glm53-r216; echo "stop exit=$?"
sleep 5
docker ps --format '{{.Names}} {{.Status}}'
echo "### launch glm53-e216 $(date -u +%FT%TZ)"
docker run -d --name glm53-e216 \
  -v $DST:/model:ro -v /tmp/e216-receipts:/receipts:ro \
  -p 8000:8000 --gpus all --shm-size=16g \
  --entrypoint python3 glm53-flash-sglang-exl3-plain:serve4-pruned \
  -m sglang.launch_server --model-path /model --host 0.0.0.0 --port 8000 \
  --quantization exl3 --tp-size 1 --ep-size 1 --context-length 262144 \
  --kv-cache-dtype fp8_e4m3 --attention-backend dsa \
  --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla \
  --linear-attn-backend triton --disable-shared-experts-fusion \
  --chunked-prefill-size 256 --max-prefill-tokens 256 --max-running-requests 1 \
  --mem-fraction-static 0.90 --enable-multimodal \
  --chat-template /model/chat_template.jinja --reasoning-parser glm45 \
  --tool-call-parser glm47 --disable-cuda-graph
echo "launch exit=$?"
echo "SERVE_LAUNCHED $(date -u +%FT%TZ)"
