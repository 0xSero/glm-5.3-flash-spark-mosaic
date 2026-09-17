#!/bin/bash
set -x
M=/home/sero/models-2p05-stage/2p05bpw
echo "### preserve attempt4 crash evidence $(date -u +%FT%TZ)"
docker logs glm53-a0-bench > /home/sero/w2port/bench/a0-bench-attempt4-overlay-batch-crash.log 2>&1
docker stop glm53-a0-bench 2>/dev/null; docker rm glm53-a0-bench 2>/dev/null
sleep 5
echo "### attempt 5: overlay moe.py raises 'EXL3 batch exceeds preplanned N tokens' when the"
echo "###   forward batch exceeds SGLANG_EXL3_MAX_BATCH_TOKENS. attempt4 (cap 256, chunked 512)"
echo "###   crashed the scheduler on the first real prefill batch. Fix = panel-proven pairing:"
echo "###   MAX_BATCH_TOKENS=256 + chunked 256 + prefill 256 + mf 0.90, only req raised 1->16."
docker run -d --name glm53-a0-bench \
  -v $M:/model:ro -v /home/sero/w2port/out:/receipts:ro -v /home/sero/w2port/routing:/routing \
  -p 8000:8000 --gpus all --shm-size=16g \
  -e SGLANG_EXL3_MAX_BATCH_TOKENS=256 \
  -e EXL3_PLAIN_CENSUS=/receipts/exl3-plain-census-2p05.json \
  -e EXL3_PLAIN_DECODER=exl3_plain_sglang_overlay.exl3_reference:decode \
  --entrypoint python3 glm53-flash-sglang-exl3-plain:serve4 \
  -m sglang.launch_server --model-path /model --host 0.0.0.0 --port 8000 \
  --quantization exl3 --tp-size 1 --ep-size 1 --context-length 262144 \
  --kv-cache-dtype fp8_e4m3 --attention-backend dsa \
  --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla \
  --linear-attn-backend triton --disable-shared-experts-fusion \
  --chunked-prefill-size 256 --max-prefill-tokens 256 --max-running-requests 16 \
  --mem-fraction-static 0.90 --enable-multimodal \
  --chat-template /opt/glm53/chat-template-mm.jinja --reasoning-parser glm45 \
  --tool-call-parser glm47 --disable-cuda-graph
docker ps --format '{{.Names}} {{.Status}}'
echo "BENCH5_LAUNCHED $(date -u +%FT%TZ)"
