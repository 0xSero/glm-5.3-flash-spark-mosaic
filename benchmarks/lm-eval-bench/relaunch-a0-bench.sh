#!/bin/bash
set -x
M=/home/sero/models-2p05-stage/2p05bpw
mkdir -p /home/sero/w2port/bench
echo "### preserve recorder serving state $(date -u +%FT%TZ)"
docker logs glm53-exl3-plain-serve-rec > /home/sero/w2port/bench/rec-serve-panel-era-final.log 2>&1
docker inspect glm53-exl3-plain-serve-rec > /home/sero/w2port/bench/rec-container-inspect-pre-bench.json
echo "### stop glm53-exl3-plain-serve-rec (preserved, not removed)"
docker stop glm53-exl3-plain-serve-rec
sleep 5
echo "### launch glm53-a0-bench (same 2.05bpw artifact; batching knobs raised; recorder mode dropped for bench - observability only)"
docker run -d --name glm53-a0-bench \
  -v $M:/model:ro -v /home/sero/w2port/out:/receipts:ro -v /home/sero/w2port/routing:/routing \
  -p 8000:8000 --gpus all --shm-size=16g \
  -e SGLANG_EXL3_MAX_BATCH_TOKENS=8192 \
  -e EXL3_PLAIN_CENSUS=/receipts/exl3-plain-census-2p05.json \
  -e EXL3_PLAIN_DECODER=exl3_plain_sglang_overlay.exl3_reference:decode \
  --entrypoint python3 glm53-flash-sglang-exl3-plain:serve4 \
  -m sglang.launch_server --model-path /model --host 0.0.0.0 --port 8000 \
  --quantization exl3 --tp-size 1 --ep-size 1 --context-length 262144 \
  --kv-cache-dtype fp8_e4m3 --attention-backend dsa \
  --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla \
  --linear-attn-backend triton --disable-shared-experts-fusion \
  --chunked-prefill-size 4096 --max-prefill-tokens 8192 --max-running-requests 32 \
  --mem-fraction-static 0.90 --enable-multimodal \
  --chat-template /opt/glm53/chat-template-mm.jinja --reasoning-parser glm45 \
  --tool-call-parser glm47 --disable-cuda-graph
docker ps --format '{{.Names}} {{.Status}}'
echo "BENCH_LAUNCHED $(date -u +%FT%TZ)"
