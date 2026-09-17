#!/bin/bash
# F216 serve (spark-557f) — attempt-2 recipe from e216/run-serve-e216-attempt2.sh with F216 paths.
# Known-good config: mem-fraction 0.90 recovered KV 344,256 >= 262,144 for E216.
# Logs bridge from t0 (E216 orchestrator KV-gate blind-spot fix built in).
set -x
W=/home/valentine/f216
DST=/home/valentine/models/glm53-3p05-pruned-f216
echo "### pre-flight $(date -u +%FT%TZ)"
test -s $DST/model.safetensors.index.json || { echo MISSING_INDEX_ABORT; exit 1; }
python3 -c "import json; r=json.load(open('/tmp/f216-receipts/exl3-plain-census-2p05.json')); assert r['verdict']['contract_ok'] is True and r['plan_sha256']=='342b840d1fd709f364d7f6ac4a2771b60afaaed6558ac0a4ca178d53dfb33590'; print('receipt_ok', r['tensors'])" || { echo RECEIPT_CHECK_ABORT; exit 1; }
echo "### launch glm53-f216"
docker run -d --name glm53-f216 \
  -v $DST:/model:ro -v /tmp/f216-receipts:/receipts:ro \
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
echo "SERVE_F216_LAUNCHED $(date -u +%FT%TZ)"
docker logs -f glm53-f216 > $W/serve-f216.out 2>&1
