#!/usr/bin/env bash
set -euo pipefail
root=/home/valentine/glm53-single-spark-release-20260911/nonuniform-smoke
image=sha256:afb74c791853d438810b27bf28994128f5baa2a3bf5c9b50b9ecdad7b9afffd5
name=${SMOKE_NAME:-glm53-nonuniform-moe-smoke-attempt1}
mkdir -p "$root/$name"
cp "$root/source/gpu_smoke.py" "$root/$name/gpu_smoke.py"
cp "$root/source/model.patched.py" "$root/$name/model.patched.py"
cp "$root/source/layerwise_counts.py" "$root/$name/layerwise_counts.py"
sha256sum "$root/$name/gpu_smoke.py" > "$root/$name/probe.sha256"
docker run --name "$name" --gpus all --ipc host --network none --cpus 4 --memory 12g --memory-swap 12g \
 -v /home/valentine/flash-experimental-staging-20260906/model:/model:ro \
 -v "$root/$name/model.patched.py":/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py:ro \
 -v "$root/$name/layerwise_counts.py":/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/layerwise_counts.py:ro \
 -v "$root/$name":/out \
 -e HF_HUB_OFFLINE=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_MXFP8_LM_HEAD=0 -e VLLM_MTP_NVFP4_LM_HEAD=0 \
 -e EXL3_FUSED_MOE=1 -e VLLM_EXL3_TRELLIS_MIN_M=1 -e VLLM_EXL3_TRELLIS_MAX_M=32 -e VLLM_EXL3_PREFILL_TRELLIS=1 \
 --entrypoint python3 "$image" /out/gpu_smoke.py
