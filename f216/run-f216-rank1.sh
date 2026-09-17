#!/usr/bin/env bash
# F216 completion capture — rank 1 (spark-557f). Derived runner; original pattern from
# glm53-full-observations-20260907/run_glm53_record_observations.sh with F216 paths.
# NOTE: rank1 does NOT write receipts; it needs the token manifest + model identity + its
# staged stage1 model files. /task is populated from 2822 (small rsync).
set -euo pipefail
image=glm53-full-observer-fla:20260907
actual=$(docker image inspect "$image" --format '{{.Id}}')
echo "image=$actual"
mode_args=(--qualify-then-collect)
if [[ "${1:-}" == "smoke" ]]; then mode_args=(--smoke-records 1); fi
docker run --rm --name "glm53-f216-completion-rank1" \
 --entrypoint python3 \
 --network host --ipc host --gpus all --user 1000:1000 \
 -e PYTHONPATH=/workspace/src -e GLM53_OBSERVER_KDA=fla \
 -e GLM53_OBSERVER_EXPERTS=native -e GLOO_SOCKET_IFNAME=tailscale0 \
 -e NCCL_SOCKET_IFNAME=tailscale0 -e NCCL_DEBUG=WARN \
 -v /home/valentine/f216/reap-src:/workspace/src/reap:ro \
 -v /home/valentine/models/q4-stage1-obs:/model:ro \
 -v /home/valentine/f216/task:/task:ro \
 "$image" \
 -m torch.distributed.run --nnodes=2 --nproc-per-node=1 --node-rank=1 \
 --master-addr=spark-2822.internal --master-port=29641 \
 -m reap.glm53_completion_capture \
 --model-identity /task/q4-original-records.model.json \
 --token-manifest /task/ours-original-records-v1/manifest.json \
 --model-root /model --output-root "/tmp/f216-rank1-unused" \
 --completion-start-shard 332 \
 --prefill-chunk-size 512 "${mode_args[@]}"
