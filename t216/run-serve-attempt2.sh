#!/bin/bash
set -x
W=/home/valentine/t216
DST=/home/valentine/models/glm53-3p05-pruned-t216
mkdir -p /tmp/t216-receipts
echo "### preserve attempt1 evidence $(date -u +%FT%TZ)"
docker logs glm53-t216 > $W/serve-attempt1-crash.log 2>&1; echo "log exit=$?"
docker rm glm53-t216; echo "rm exit=$?"
echo "### produce served census receipt (attempt2)"
python3 $W/make-served-census-receipt.py; echo "receipt exit=$?"
echo "### relaunch glm53-t216"
docker run -d --name glm53-t216 \
  -v $DST:/model:ro -v /tmp/t216-receipts:/receipts:ro \
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
echo "SERVE2_LAUNCHED $(date -u +%FT%TZ)"
