#!/usr/bin/env bash
# Experimental single-Spark comparison; preserves its container and logs.
set -euo pipefail
: "${GLM53_MODEL_ROOT:?verified complete original K2 model}"
: "${GLM53_MTP_ROOT:?CPU-verified native MTP view with /model symlinks}"
: "${GLM53_IMAGE:?verified locally transferred B12x image}"
: "${GLM53_PROFILE_ROOT:?new experiment profile directory}"
name="${GLM53_CONTAINER_NAME:-glm53-b12x-k2full-fp8mtp-attempt1}"
docker image inspect "$GLM53_IMAGE" >/dev/null
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo 'GPU already has a compute process; refusing launch' >&2
  exit 1
fi
if docker container inspect "$name" >/dev/null 2>&1; then
  echo 'Experiment container already exists; preserve it and choose a new attempt' >&2
  exit 1
fi
python3 - "$GLM53_MTP_ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); c=json.loads((p/'config.json').read_text())
r=json.loads((p/'native-mtp-view.json').read_text())
assert c.get('quantization_config') is None
assert c['text_config'].get('quantization_config') is None
assert c['text_config']['n_routed_experts']==288 and r['index_tensor_count']==891
PY
mkdir -p "$GLM53_PROFILE_ROOT"
exec docker run -d --pull never --gpus all --ipc host --network host \
  --name "$name" \
  -v "$GLM53_MODEL_ROOT":/model:ro -v "$GLM53_MTP_ROOT":/mtp:ro \
  -v "$GLM53_PROFILE_ROOT":/profiles \
  -e VLLM_MXFP8_LM_HEAD=0 -e VLLM_MTP_NVFP4_LM_HEAD=0 \
  -e GLM53_MTP_EXPERT_FP8=1 \
  -e HF_HUB_OFFLINE=1 -e SAFETENSORS_DROP_PAGE_CACHE=1 \
  -e SAFETENSORS_LOAD_DEVICE=cuda:0 \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_EXL3_TRELLIS_MIN_M=1 -e VLLM_EXL3_TRELLIS_MAX_M=32 \
  -e VLLM_EXL3_PREFILL_TRELLIS=1 -e EXL3_FUSED_MOE=1 \
  -v "$GLM53_DEPTH_PLUGIN_ROOT":/depth_sweep:ro \
  -e PYTHONPATH=/depth_sweep -e GLM53_CAPTURE_RECEIPTS=/profiles/capture-receipts \
  "$GLM53_IMAGE" --model /model --served-model-name glm-5.3-flash \
  --host 127.0.0.1 --port 18080 --tensor-parallel-size 1 \
  --decode-context-parallel-size 1 --no-enable-expert-parallel \
  --quantization exl3 --load-format safetensors --dtype bfloat16 \
  --kv-cache-dtype fp8_ds_mla --block-size 256 --gpu-memory-utilization 0.93 \
  --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --attention-backend B12X --additional-config '{"kda_prefill_backend":"b12x"}' \
  --enable-chunked-prefill --no-enable-prefix-caching --mm-processor-cache-gb 0.1 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"],"cudagraph_capture_sizes":[1,3],"max_cudagraph_capture_size":3}' \
  --generation-config vllm --reasoning-parser glm47 --tool-call-parser glm47 \
  --enable-auto-tool-choice --trust-remote-code \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/profiles","ignore_frontend":true,"torch_profiler_with_stack":false,"torch_profiler_record_shapes":false,"max_iterations":8}' \
  --speculative-config '{"method":"mtp","model":"/mtp","num_speculative_tokens":2,"attention_backend":"B12X"}'
