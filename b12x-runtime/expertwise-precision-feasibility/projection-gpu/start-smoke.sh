#!/usr/bin/env bash
set -euo pipefail
root=/home/valentine/glm53-single-spark-release-20260911/projection-precision-smoke
image=sha256:c0d5f76237a5fe1262291eb23b8086ef0b2637e6d8d45ccdbb9c5706df321399
name=${SMOKE_NAME:-glm53-projection-k223-smoke-attempt1}
mkdir -p "$root/$name"
cp "$root/source/gpu_smoke.py" "$root/source/exl3.projection.py" "$root/source/glm_exl3_projection_bits.py" "$root/$name/"
sha256sum "$root/$name/"*.py > "$root/$name/source.sha256"
docker run --name "$name" --gpus all --ipc host --network none --cpus 4 --memory 12g --memory-swap 12g \
 -v /home/valentine/glm53-single-spark-release-20260911/quality/mixed-layer-candidate/model:/model:ro \
 -v /home/valentine/glm53-exl3-staging/release-trellis-k2-2bpw/layers:/k2-source:ro \
 -v "$root/$name/exl3.projection.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py:ro \
 -v "$root/$name/glm_exl3_projection_bits.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/glm_exl3_projection_bits.py:ro \
 -v "$root/$name":/out \
 -e HF_HUB_OFFLINE=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_MXFP8_LM_HEAD=0 -e VLLM_MTP_NVFP4_LM_HEAD=0 \
 -e EXL3_FUSED_MOE=1 -e VLLM_EXL3_TRELLIS_MIN_M=1 -e VLLM_EXL3_TRELLIS_MAX_M=32 -e VLLM_EXL3_PREFILL_TRELLIS=1 \
 --entrypoint python3 "$image" /out/gpu_smoke.py
