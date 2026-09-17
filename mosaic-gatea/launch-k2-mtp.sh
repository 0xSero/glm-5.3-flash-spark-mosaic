#!/usr/bin/env bash
# K2 staging artifact (mcg codebook) as TARGET + the program's native MTP draft (recipe = the proven R4 run).
# Recipe provenance: native-mtp/REPORT.md + k2full-fp8draft-r4-inspect-ready.json (load 97.11 GiB,
# KV 460,208, target+draft graphs captured, MTP counters advancing) + b12x-runtime/mtp-baseline-replay/d2-c1-attempt2.
# CUDA graphs are ON (cudagraph_mode FULL_DECODE_ONLY); MTP is ON (--speculative-config method=mtp); no greedy anywhere.
# Runs ON spark-557f. Target artifact is never modified (mounted read-only).
set -euo pipefail
IMG=local/glm53-reap-native-mtp:20260911-r4-expert-fp8
MODEL=/home/valentine/flash-experimental-staging-20260906/model
NATIVE=/home/valentine/flash-experimental-staging-20260906/model
MTP=/home/valentine/glm53-single-spark-release-20260911/native-mtp-k2-control
NAME=${NAME:-glm53-k2-mtp}
MF=${MF:-0.93}
SPEC=${SPEC:-'{"method":"mtp","model":"/mtp","num_speculative_tokens":1}'}
LOG=/home/valentine/mosaic-gatea/mtp-$NAME.log

test -d "$MODEL" && test -d "$NATIVE" && test -d "$MTP" || { echo "missing mount source" >&2; exit 1; }
python3 - "$MTP" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); c=json.loads((p/'config.json').read_text()); r=json.loads((p/'native-mtp-view.json').read_text())
assert c.get('quantization_config') is None
assert c['text_config']['n_routed_experts']==288
assert r['index_tensor_count']==891
print('mtp_view_ok native_bytes=%s experts=%s draft_quant=%s' % (r['native_mtp_bytes'], r['mtp_experts'], r['draft_quantization']))
PY
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU already has a compute process; stopping own SGLang mosaic-10lg (receipted)" | tee -a "$LOG"
  docker rm -f glm53-mosaic-10lg glm53-mosaic12l-mtp >>"$LOG" 2>&1 || true
  sleep 5
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU still busy after own-container stop; refusing launch" >&2; exit 1
fi
docker container inspect "$NAME" >/dev/null 2>&1 && { echo "container $NAME exists; preserve it and choose another name" >&2; exit 1; }

exec docker run --rm --gpus all --ipc host --network host \
  --name "$NAME" \
  -v "$MODEL":/model:ro -v "$NATIVE":/native-source:ro -v "$MTP":/mtp:ro \
  -e GLM53_MTP_EXPERT_FP8=1 \
  -e HF_HUB_OFFLINE=1 -e SAFETENSORS_DROP_PAGE_CACHE=1 \
  -e SAFETENSORS_LOAD_DEVICE=cuda:0 \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_EXL3_TRELLIS_MIN_M=1 -e VLLM_EXL3_TRELLIS_MAX_M=32 \
  -e VLLM_EXL3_PREFILL_TRELLIS=1 -e EXL3_FUSED_MOE=1 \
  "$IMG" /model --served-model-name glm-5.3-flash \
  --host 127.0.0.1 --port 18080 --tensor-parallel-size 1 \
  --decode-context-parallel-size 1 --no-enable-expert-parallel \
  --quantization exl3 --load-format safetensors --dtype bfloat16 \
  --kv-cache-dtype fp8 --block-size 64 --gpu-memory-utilization "$MF" \
  --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --attention-backend FLASHINFER_MLA_SPARSE_SM120 \
  --enable-chunked-prefill --no-enable-prefix-caching --mm-processor-cache-gb 0.1 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"],"cudagraph_capture_sizes":[2],"max_cudagraph_capture_size":2}' \
  --generation-config vllm --reasoning-parser glm47 --tool-call-parser glm47 \
  --enable-auto-tool-choice --trust-remote-code \
  --speculative-config "$SPEC"