#!/usr/bin/env bash
# Host needs only Bash, Docker, and the NVIDIA Container Toolkit.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 || ${3:-} != '' && ${3:-} != '--download' ]]; then
  echo "Usage: GLM53_IMAGE=<image@sha256:digest> $0 MODEL_DIRECTORY STATE_DIRECTORY [--download]" >&2
  exit 2
fi
: "${GLM53_IMAGE:?Supply the independently validated B12x image by immutable digest}"
if [[ ! $GLM53_IMAGE =~ ^([a-zA-Z0-9./:_-]+@)?sha256:[a-f0-9]{64}$ ]]; then
  echo 'GLM53_IMAGE must be a registry digest or local sha256 image ID, not a mutable tag' >&2
  exit 2
fi
package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p -- "$1" "$2"
model=$(cd -- "$1" && pwd)
state=$(cd -- "$2" && pwd)
if [[ $model == "$state" || $state == "$model/"* || $model == "$state/"* ]]; then
  echo 'Model and state directories must be separate and non-nested' >&2
  exit 2
fi
name=${CONTAINER_NAME:-glm53-spark-next}
if docker container inspect "$name" >/dev/null 2>&1; then
  echo "Container $name already exists. It was not stopped or replaced." >&2
  exit 1
fi
if ! docker image inspect "$GLM53_IMAGE" >/dev/null 2>&1; then
  docker pull "$GLM53_IMAGE"
fi
gpu_guard() {
  docker run --rm --pull never --gpus all --entrypoint python3 \
    -v "$package":/setup:ro "$GLM53_IMAGE" /setup/setup.py gpu-check
}
# Refuse occupied GPUs before the download or full-checkpoint hash work.
gpu_guard
mode=ro
download=()
if [[ ${3:-} == '--download' ]]; then
  pin='0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw@35b4b580debe2a1510a2eb042d9d6d4cd7fb3a6b'
  if [[ -z $(ls -A -- "$model") ]]; then
    printf '%s\n' "$pin" > "$model/.spark-model-pin"
  elif [[ ! -f $model/.spark-model-pin || $(cat -- "$model/.spark-model-pin") != "$pin" ]]; then
    echo 'Download needs an empty directory or a previous download for this exact pin' >&2
    exit 1
  fi
  mode=rw
  download=(--download)
fi
# Download and full hash/index validation run without GPU access.
docker run --rm --pull never --cpus 2 --memory 3g --memory-swap 3g --entrypoint python3 \
  -v "$package":/setup:ro -v "$model":/model:"$mode" -v "$state":/state \
  "$GLM53_IMAGE" /setup/setup.py prepare "${download[@]}"
# Read-only inventory; refuses rather than stopping unrelated GPU work.
gpu_guard
docker run -d --pull never --init --gpus all --ipc host --network host \
  --name "$name" --restart unless-stopped \
  --health-cmd 'python3 /setup/setup.py health' \
  --health-interval 30s --health-timeout 15s --health-start-period 30m --health-retries 3 \
  -v "$package":/setup:ro -v "$model":/model:ro -v "$state":/state \
  -e GLM53_MTP_EXPERT_FP8=1 -e HF_HUB_OFFLINE=1 \
  -e VLLM_MXFP8_LM_HEAD=0 -e VLLM_MTP_NVFP4_LM_HEAD=0 \
  -e SAFETENSORS_DROP_PAGE_CACHE=1 -e SAFETENSORS_LOAD_DEVICE=cuda:0 \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_EXL3_TRELLIS_MIN_M=1 -e VLLM_EXL3_TRELLIS_MAX_M=32 \
  -e VLLM_EXL3_PREFILL_TRELLIS=1 -e EXL3_FUSED_MOE=1 \
  -e MTP_DEPTH="${MTP_DEPTH:-1}" -e ACTIVE_SLOTS="${ACTIVE_SLOTS:-1}" \
  -e PREFILL_CHUNK="${PREFILL_CHUNK:-2048}" -e MEMORY_FRACTION="${MEMORY_FRACTION:-0.93}" \
  -e BIND_HOST="${BIND_HOST:-127.0.0.1}" -e READINESS_TIMEOUT="${READINESS_TIMEOUT:-1800}" \
  --entrypoint python3 "$GLM53_IMAGE" /setup/setup.py serve
echo "Started $name. Readiness requires a fresh answer and native MTP counters; follow: docker logs -f $name"
echo 'Endpoint after health becomes healthy: http://127.0.0.1:18080/v1 ; model: glm-5.3-flash'
