#!/usr/bin/env bash
# GLM-5.3-Flash EXL3 serving on a single DGX Spark (GB10, arm64, 128 GB unified memory).
#
# Panel-proven configuration, replayed from the program's own receipts
# (benchmarks/lm-eval-bench/relaunch-a0-bench5.sh — the exact command behind the
# a0 baseline and every G4/MMLU/GPQA number in this repo):
#   image : ghcr.io/0xsero/glm53-flash-exl3-plain:2p05-sglang-mul1-r1
#           (digest-pinned id c65c840f1908…, the image that served the a0 baseline)
#   steps : 1) census the model tensors (content-derived receipt, catches any
#              truncated/mismatched download BEFORE the server starts)
#           2) launch SGLang with the exl3 overlay, 262,144 context, fp8 KV,
#              vision on, MAX_BATCH_TOKENS=256 pairing (the pairing that avoids
#              the "EXL3 batch exceeds preplanned N tokens" scheduler crash)
#
# Defaults are the a0 (2.05bpw) values. The M288-12L mosaic is ~11 GB larger, so
# it needs GLM53_MEM_FRACTION=0.95 (KV 595,200) and GLM53_MAX_RUNNING=1 — the
# values its confirming run was measured with; at 0.90 the 262,144-token KV gate
# does not fit.
#
# Usage:  ./run-spark.sh /path/to/GLM-5.3-Flash-exl3-2.05bpw
#         GLM53_MEM_FRACTION=0.95 GLM53_MAX_RUNNING=1 ./run-spark.sh /path/to/mosaic-12l
set -euo pipefail
MODEL="${1:?usage: run-spark.sh /path/to/GLM-5.3-Flash-exl3-2.05bpw}"
IMG="${GLM53_IMG:-ghcr.io/0xsero/glm53-flash-exl3-plain:2p05-sglang-mul1-r1}"
NAME="${GLM53_CONTAINER:-glm53-a0}"
WORK="$(cd "$(dirname "$0")" && pwd)"
RECEIPTS="${GLM53_RECEIPTS:-$WORK/receipts}"
PORT="${GLM53_PORT:-8000}"
MEM="${GLM53_MEM_FRACTION:-0.90}"
MAXRUN="${GLM53_MAX_RUNNING:-16}"
mkdir -p "$RECEIPTS"

CENSUS="$RECEIPTS/exl3-plain-census.json"
echo "### census $(date -u +%FT%TZ)"
docker run --rm --gpus all \
  -v "$MODEL":/model:ro -v "$RECEIPTS":/receipts \
  --entrypoint python3 "$IMG" \
  -m exl3_plain_sglang_overlay.census /model --out /receipts/exl3-plain-census.json
python3 - "$CENSUS" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
assert r.get("schema") == "exl3-plain-census-v1" and r.get("verdict", {}).get("contract_ok") is True, \
    "census contract check FAILED — model does not match the served EXL3 layout; aborting before serve"
print("census_ok tensors=%s codebook=%s" % (r.get("tensors", "n/a"), r.get("codebook")))
EOF

docker rm -f "$NAME" 2>/dev/null || true
echo "### serve $(date -u +%FT%TZ)"
docker run -d --name "$NAME" \
  -v "$MODEL":/model:ro -v "$RECEIPTS":/receipts:ro \
  -p "$PORT":8000 --gpus all --shm-size=16g \
  -e SGLANG_EXL3_MAX_BATCH_TOKENS=256 \
  -e EXL3_PLAIN_CENSUS=/receipts/exl3-plain-census.json \
  -e EXL3_PLAIN_DECODER=exl3_plain_sglang_overlay.exl3_reference:decode \
  --entrypoint python3 "$IMG" \
  -m sglang.launch_server --model-path /model --host 0.0.0.0 --port 8000 \
  --quantization exl3 --tp-size 1 --ep-size 1 --context-length 262144 \
  --kv-cache-dtype fp8_e4m3 --attention-backend dsa \
  --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla \
  --linear-attn-backend triton --disable-shared-experts-fusion \
  --chunked-prefill-size 256 --max-prefill-tokens 256 --max-running-requests "$MAXRUN" \
  --mem-fraction-static "$MEM" --enable-multimodal \
  --chat-template /opt/glm53/chat-template-mm.jinja --reasoning-parser glm45 \
  --tool-call-parser glm47 --disable-cuda-graph

echo "### waiting for /health (first load can take several minutes)"
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "HEALTH_OK $(date -u +%FT%TZ) — serving on http://127.0.0.1:$PORT/v1"
    exit 0
  fi
  if ! docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true; then
    echo "CONTAINER_DIED — logs:"; docker logs --tail 50 "$NAME"; exit 1
  fi
  sleep 10
done
echo "HEALTH_TIMEOUT — logs:"; docker logs --tail 50 "$NAME"; exit 1
