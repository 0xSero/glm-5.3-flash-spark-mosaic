#!/usr/bin/env bash
# Candidate only. Run AFTER independently sealing the REAP model and MTP view.
set -euo pipefail
: "${GLM53_MODEL_ROOT:?sealed REAP target directory}"
: "${GLM53_NATIVE_SOURCE:?source directory used for the native draft view}"
: "${GLM53_MTP_ROOT:?prepared native draft view}"
: "${GLM53_IMAGE:?locally built native MTP candidate image}"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo 'GPU already has a compute process; refusing launch' >&2; exit 1
fi
python3 - "$GLM53_MTP_ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);c=json.loads((p/'config.json').read_text());r=json.loads((p/'native-mtp-view.json').read_text());assert c.get('quantization_config') is None;assert c['text_config']['n_routed_experts']==288;assert r['index_tensor_count']==891
PY
exec docker run --rm --gpus all --ipc host --network host \
  --name glm53-reap-native-mtp-candidate \
  -v "$GLM53_MODEL_ROOT":/model:ro \
  -v "$GLM53_NATIVE_SOURCE":/native-source:ro \
  -v "$GLM53_MTP_ROOT":/mtp:ro \
  -e GLM53_MTP_EXPERT_FP8="${GLM53_MTP_EXPERT_FP8:-0}" \
  -e HF_HUB_OFFLINE=1 -e SAFETENSORS_DROP_PAGE_CACHE=1 \
  -e SAFETENSORS_LOAD_DEVICE=cuda:0 \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_EXL3_TRELLIS_MIN_M=1 -e VLLM_EXL3_TRELLIS_MAX_M=32 \
  -e VLLM_EXL3_PREFILL_TRELLIS=1 -e EXL3_FUSED_MOE=1 \
  "$GLM53_IMAGE" /model --served-model-name glm-5.3-flash \
  --host 127.0.0.1 --port 18080 --tensor-parallel-size 1 \
  --decode-context-parallel-size 1 --no-enable-expert-parallel \
  --quantization exl3 --load-format safetensors --dtype bfloat16 \
  --kv-cache-dtype fp8 --block-size 64 --gpu-memory-utilization "${GLM53_MEMORY_FRACTION:-0.93}" \
  --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --attention-backend FLASHINFER_MLA_SPARSE_SM120 \
  --enable-chunked-prefill --no-enable-prefix-caching --mm-processor-cache-gb 0.1 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"],"cudagraph_capture_sizes":[2],"max_cudagraph_capture_size":2}' \
  --generation-config vllm --reasoning-parser glm47 --tool-call-parser glm47 \
  --enable-auto-tool-choice --trust-remote-code \
  --speculative-config '{"method":"mtp","model":"/mtp","num_speculative_tokens":1}'
