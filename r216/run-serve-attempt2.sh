#!/bin/bash
set -x
W=/home/valentine/r216
DST=/home/valentine/models/glm53-3p05-pruned-r216
echo "### preserve attempt1 evidence $(date -u +%FT%TZ)"
docker logs glm53-r216 > $W/serve-attempt1-short-kv.log 2>&1; echo "log exit=$?"
echo "### attempt1 verdict: READY-path but max_total_num_tokens=253248 < 262144 required (known profiling variance; T216 relaunch recovered 302656); relaunching once"
docker stop glm53-r216; echo "stop exit=$?"
docker rm glm53-r216; echo "rm exit=$?"
echo "### relaunch glm53-r216 (attempt 2)"
docker run -d --name glm53-r216 \
  -v $DST:/model:ro -v /tmp/r216-receipts:/receipts:ro \
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
