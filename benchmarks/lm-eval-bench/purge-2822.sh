#!/bin/bash
# DELETION per user directive 2026-09-14: spark-2822 keeps ONLY GLM-5.3-Flash
# single-spark artifacts. Every target logged (KB + path) BEFORE rm.
# NEVER TOUCHED: NFS NAS mount (GLM-5.2-EXL3-TR3-3.0bpw -> spark-raila.internal),
# glm53 campaign dirs, glm53-rank-relay, sealed laguna-reap-saliency-v1,
# reap calibration datasets, qwen36-hybrid-evidence, local-studio (runtime logs),
# spark service code, all dsv4 work/protected/archive dirs (flagged, not models).
set -u
[ "$(hostname)" = "spark-2822" ] || { echo "ABORT wrong host $(hostname)"; exit 1; }
LOG=/home/sero/work/glm53-single-spark-release-20260911/receipts/DELETION-20260914-2822.log
mkdir -p "$(dirname "$LOG")"
if ! df /home/sero/models/GLM-5.2-EXL3-TR3-3.0bpw 2>/dev/null | grep -q "spark-raila.internal"; then
  echo "$(date -u +%FT%TZ) ABORT: NAS NFS mount not present - refusing (wrong-state guard)" >> "$LOG"
  exit 1
fi
log_rm(){ local p="$1"; if [ -e "$p" ]; then du -sk "$p" >> "$LOG" 2>/dev/null; echo "RM $p" >> "$LOG"; rm -rf "$p"; else echo "SKIP missing $p" >> "$LOG"; fi; }
echo "$(date -u +%FT%TZ) START purge uid=$(id -u) df: $(df -h / | tail -1)" >> "$LOG"

M=/home/sero/models
for p in "$M/deepseek-ai" "$M/DeepSeek-V4-Flash-180B" "$M/Gemma-4-31B-IT-NVFP4" \
  "$M/Inkling-Small-EXL3-2.5bpw" "$M/Inkling-Small-EXL3-2.5bpw-tp1" \
  "$M/Inkling-Small-EXL3-3.0bpw-promoted-tp1" "$M/Inkling-Small-EXL3-3.0bpw-tp1" \
  "$M/Inkling-Small-EXL3-K4-promoted" "$M/Inkling-Small-NVFP4" \
  "$M/Laguna-S-2.1-DFlash-NVFP4" \
  "$M/Laguna-S-2.1-Vision-NVFP4-20260726-a10" "$M/Laguna-S-2.1-Vision-NVFP4-20260726-a2" \
  "$M/Laguna-S-2.1-Vision-NVFP4-20260726-a3" "$M/Laguna-S-2.1-Vision-NVFP4-20260726-a4" \
  "$M/Laguna-S-2.1-Vision-NVFP4-20260726-a5" "$M/Laguna-S-2.1-Vision-NVFP4-20260726-a6" \
  "$M/Laguna-S-2.1-Vision-NVFP4-20260726-a7" "$M/Laguna-S-2.1-Vision-NVFP4-20260726-a8" \
  "$M/Laguna-S-2.1-Vision-NVFP4-20260726-a9" \
  "$M/Laguna-S-2.1-Vision-NVFP4-a3-live-candidate-20260726-a1.partial" \
  "$M/Laguna-S-2.1-Vision-NVFP4-a3-live-candidate-20260726-a2" \
  "$M/Laguna-S-2.1-Vision-NVFP4-a3-live-candidate-20260726-a5" \
  "$M/Laguna-S-2.1-Vision-NVFP4-LiveAnchor-20260727-a11" \
  "$M/Laguna-S-2.1-Vision-NVFP4-LiveAnchor-templatefix-r2-20260728T1245Z-5b10" \
  "$M/LiquidAI" "$M/nvidia" "$M/NVIDIA-Nemotron" "$M/Qwen" "$M/Qwen2.5-0.5B-Instruct-GGUF" \
  "$M/Qwen3.6-27B-Text-NVFP4-MTP" "$M/Qwen3.6-35B-A3B-NVFP4" \
  "$M/Qwen3.6-35B-Hyrbid-3.25bpw-release" "$M/qwen36-hybrid-cache" \
  "$M/qwen36-hybrid-nf3g32-candidate" "$M/qwen36-nvfp4" "$M/Qwen3.8-27B-NVFP4-BF16-LMHead" \
  "$M/RadixArk" "$M/sglang-inkling" "$M/dl_glm52_nvfp4.log"; do log_rm "$p"; done

S=/home/sero/spark/models
for p in "$S/hf-cache/hub/models--0xSero--Step-3.7-Flash-173B" \
  "$S/hf-cache/hub/models--0xSero--DeepSeek-V4-Flash-180B-codex-K160-REAP" \
  "$S/hf-cache/models--0xSero--DeepSeek-V4-Flash-213B-uniaware-REAP" \
  "$S/hf-cache/models--0xSero--DeepSeek-V4-Flash-200B-REAP" \
  "$S/hf-cache/xet" "$S/ds4-gguf" "$S/ds4-hybrid" "$S/vllm-cache" "$S/adapters" \
  "$S/GLM-5.2-NVFP4-src-nvidia-meta" "$S/GLM-5.2-NVFP4-src-nvidia-meta.tmp"; do log_rm "$p"; done

for p in /home/sero/spark/Qwen3.8-27B-SGLang-DGX-Spark /home/sero/spark/nex-n2-mini/models \
  /home/sero/spark/locate-anything/models /home/sero/spark/services/ComfyUI/models \
  /home/sero/spark/services/ComfyUI/.hf-cache; do log_rm "$p"; done

H=/home/sero/.cache/huggingface/hub
for p in "$H/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw" \
  "$H/models--0xSero--GLM-5.3-Flash-Abliterated-EXL3-3.0bpw" \
  "$H/models--incoai--GLM-5.3-Flash-DFlash2" \
  "$H/models--Systran--faster-whisper-large-v3"; do log_rm "$p"; done
for d in "$H"/models--bootstrap--*; do [ -e "$d" ] && log_rm "$d"; done

for p in /home/sero/exo/models /home/sero/ollama-fleet/models /home/sero/.ollama/models; do log_rm "$p"; done

{ echo "$(date -u +%FT%TZ) docker images BEFORE:"; docker images -a --format "{{.ID}} {{.Repository}}:{{.Tag}} {{.Size}}" 2>&1
  sudo -n docker image prune -a -f 2>&1 | tail -3
  echo "$(date -u +%FT%TZ) docker system df AFTER:"; sudo -n docker system df 2>&1; } >> "$LOG"

echo "$(date -u +%FT%TZ) DONE df: $(df -h / | tail -1)" >> "$LOG"
sync
