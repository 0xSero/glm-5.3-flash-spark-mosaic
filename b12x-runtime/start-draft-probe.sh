#!/usr/bin/env bash
set -euo pipefail
root=/home/sero/work/glm53-single-spark-release-20260911
image=${GLM53_IMAGE:?set exact qualified image tag}
expected=${GLM53_EXPECTED_IMAGE:?set expected image SHA}
name=${GLM53_PROBE_NAME:-glm53-b12x-mtp-real-probe1}
[[ $(docker image inspect "$image" --format '{{.Id}}') == "$expected" ]]
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
if docker container inspect "$name" >/dev/null 2>&1; then exit 1; fi
mkdir -p "$root/b12x-runtime/probe-results/$name"
cp "$root/b12x-runtime/probe_native_mtp_real.py" "$root/b12x-runtime/probe-results/$name/probe.py"
cp "$root/b12x-runtime/native-parameter-baseline.json" "$root/b12x-runtime/probe-results/$name/"
docker run -d --name "$name" --gpus all --ipc host --network host \
 -v "$root/source-k2:/model:ro" -v "$root/b12x-runtime/native-mtp-view:/mtp:ro" \
 -v "$root/b12x-runtime/probe-results/$name/probe.py:/probe.py:ro" \
 -v "$root/b12x-runtime/probe-results/$name:/audit" \
 -e VLLM_MXFP8_LM_HEAD=0 -e VLLM_MTP_NVFP4_LM_HEAD=0 \
  -e GLM53_MTP_EXPERT_FP8=1 -e HF_HUB_OFFLINE=1 \
 -e SAFETENSORS_DROP_PAGE_CACHE=1 -e SAFETENSORS_LOAD_DEVICE=cuda:0 \
 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_FLOAT32_MATMUL_PRECISION=highest \
 --entrypoint python3 "$image" /probe.py
