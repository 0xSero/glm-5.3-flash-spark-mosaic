#!/bin/bash
set -x
DST=/home/valentine/models/glm53-3p05-pruned-r216
mkdir -p /home/valentine/bench
echo "### preserve panel-era serving state $(date -u +%FT%TZ)"
docker logs glm53-r216 > /home/valentine/bench/r216-serve-panel-era-final.log 2>&1
docker inspect glm53-r216 > /home/valentine/bench/r216-container-inspect-pre-bench.json
test -f /tmp/r216-receipts/exl3-plain-census-2p05.json || python3 /home/valentine/r216/make-served-census-receipt-r216.py
echo "### stop glm53-r216 (preserved, not removed)"
docker stop glm53-r216
sleep 5
echo "### launch glm53-r216-bench (same artifact; batching knobs raised for benchmark throughput; SGLANG_EXL3_MAX_BATCH_TOKENS 256->8192)"
docker run -d --name glm53-r216-bench \
  -v $DST:/model:ro -v /tmp/r216-receipts:/receipts:ro \
  -p 8000:8000 --gpus all --shm-size=16g \
  -e SGLANG_EXL3_MAX_BATCH_TOKENS=8192 \
  --entrypoint python3 glm53-flash-sglang-exl3-plain:serve4-pruned \
  -m sglang.launch_server --model-path /model --host 0.0.0.0 --port 8000 \
  --quantization exl3 --tp-size 1 --ep-size 1 --context-length 262144 \
  --kv-cache-dtype fp8_e4m3 --attention-backend dsa \
  --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla \
  --linear-attn-backend triton --disable-shared-experts-fusion \
  --chunked-prefill-size 4096 --max-prefill-tokens 8192 --max-running-requests 32 \
  --mem-fraction-static 0.90 --enable-multimodal \
  --chat-template /model/chat_template.jinja --reasoning-parser glm45 \
  --tool-call-parser glm47 --disable-cuda-graph
docker ps --format '{{.Names}} {{.Status}}'
echo "BENCH_LAUNCHED $(date -u +%FT%TZ)"
