#!/usr/bin/env bash
# MTP serving + speed: replay of the program's D2/C1 recipe on the B12x runtime.
#
# Recipe provenance (read, not invented):
#   b12x-runtime/mtp-baseline-replay/d2-c1-attempt2/launch.sh   (flags, mounts, env)
#   b12x-runtime/mtp-baseline-replay/d2-c1-attempt2/plan.json   (depth 2, capture [1,3], mf 0.93)
#   b12x-runtime/mtp-baseline-replay/d2-c1-attempt2/admission.json (ADMITTED_FOR_ISOLATED_BENCHMARK)
#       image_id sha256:afb74c791853...  kv_dtype fp8_ds_mla  kv_capacity 753664  depth 2  B12X
#   benchmarks/structured-sustained-20260912/TABLE.md  D1 = 14.98 tok/s (depth 1, no B12X)
#   EXPERIMENTS.md 2026-09-12 D1->D2 table: D2 = 19.11/19.03 TOTAL decode tok/s (+27.6/28.1%)
#
# Differences from the r4-image variant (glm53-k2-mtp):
#   kv-cache-dtype fp8 -> fp8_ds_mla ; block-size 64 -> 256 ;
#   attention FLASHINFER_MLA_SPARSE_SM120 -> B12X (+ kda_prefill_backend b12x) ;
#   spec depth 1 -> 2 ; capture sizes [2] -> [1,3] ; image r4-expert-fp8 -> b12x-exl3.
#
# CUDA graphs stay ON (FULL_DECODE_ONLY); MTP stays ON (method=mtp). No greediness is set anywhere here.
# One GPU job per node: this script refuses to launch if the GPU is busy; stop the previous
# own-container first with its own receipt.
set -euo pipefail
IMG=glm53-b12x-exl3:jovian-3aada677-r3
IMG_ID_EXPECT=sha256:afb74c791853d438810b27bf28994128f5baa2a3bf5c9b50b9ecdad7b9afffd5
MODEL=/home/valentine/flash-experimental-staging-20260906/model
MTP=/home/valentine/glm53-single-spark-release-20260911/native-mtp-k2-control
PLUGIN=/home/valentine/mosaic-gatea/b12x-d2-plugin
PROFILES=/home/valentine/mosaic-gatea/d2-profiles
NAME=${NAME:-glm53-k2-mtp-b12x-d2}
MF=${MF:-0.93}
LOG=/home/valentine/mosaic-gatea/mtp-$NAME.log

test -d "$MODEL" && test -d "$MTP" && test -d "$PLUGIN" || { echo "missing mount source" >&2; exit 1; }
mkdir -p "$PROFILES/capture-receipts"
actual=$(docker image inspect "$IMG" --format '{{.Id}}')
[ "$actual" = "$IMG_ID_EXPECT" ] || { echo "image id mismatch: $actual" >&2; exit 1; }
python3 - "$MTP" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); c=json.loads((p/'config.json').read_text()); r=json.loads((p/'native-mtp-view.json').read_text())
assert c.get('quantization_config') is None
assert c['text_config'].get('quantization_config') is None
assert c['text_config']['n_routed_experts']==288 and r['index_tensor_count']==891
print('mtp_view_ok native_bytes=%s experts=%s draft_quant=%s' % (r['native_mtp_bytes'], r['mtp_experts'], r['draft_quantization']))
PY
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "GPU already has a compute process; refusing launch (stop your own container with a receipt first)" >&2
  exit 1
fi
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "container $NAME exists; preserve it and choose another name" >&2; exit 1
fi

exec docker run -d --pull never --gpus all --ipc host --network host \
  --name "$NAME" \
  -v "$MODEL":/model:ro -v "$MODEL":/native-source:ro -v "$MTP":/mtp:ro \
  -v "$PROFILES":/profiles -v "$PLUGIN":/depth_sweep:ro \
  -e VLLM_MXFP8_LM_HEAD=0 -e VLLM_MTP_NVFP4_LM_HEAD=0 \
  -e GLM53_MTP_EXPERT_FP8=1 \
  -e HF_HUB_OFFLINE=1 -e SAFETENSORS_DROP_PAGE_CACHE=1 \
  -e SAFETENSORS_LOAD_DEVICE=cuda:0 \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_EXL3_TRELLIS_MIN_M=1 -e VLLM_EXL3_TRELLIS_MAX_M=32 \
  -e VLLM_EXL3_PREFILL_TRELLIS=1 -e EXL3_FUSED_MOE=1 \
  -e PYTHONPATH=/depth_sweep -e GLM53_CAPTURE_RECEIPTS=/profiles/capture-receipts \
  "$IMG" --model /model --served-model-name glm-5.3-flash \
  --host 127.0.0.1 --port 18080 --tensor-parallel-size 1 \
  --decode-context-parallel-size 1 --no-enable-expert-parallel \
  --quantization exl3 --load-format safetensors --dtype bfloat16 \
  --kv-cache-dtype fp8_ds_mla --block-size 256 --gpu-memory-utilization "$MF" \
  --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --attention-backend B12X --additional-config '{"kda_prefill_backend":"b12x"}' \
  --enable-chunked-prefill --no-enable-prefix-caching --mm-processor-cache-gb 0.1 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"],"cudagraph_capture_sizes":[1,3],"max_cudagraph_capture_size":3}' \
  --generation-config vllm --reasoning-parser glm47 --tool-call-parser glm47 \
  --enable-auto-tool-choice --trust-remote-code \
  --speculative-config '{"method":"mtp","model":"/mtp","num_speculative_tokens":2,"attention_backend":"B12X"}'