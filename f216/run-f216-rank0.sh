#!/usr/bin/env bash
# F216 completion capture — rank 0 (spark-2822). Derived runner; original pattern from
# glm53-full-observations-20260907/run_glm53_record_observations.sh with F216 paths.
set -euo pipefail
image=glm53-full-observer-fla:20260907
actual=$(docker image inspect "$image" --format '{{.Id}}')
echo "image=$actual"
mode_args=(--qualify-then-collect)
if [[ "${1:-}" == "smoke" ]]; then mode_args=(--smoke-records 1); fi
docker run --rm --name "glm53-f216-completion-rank0" \
 --entrypoint python3 \
 --network host --ipc host --gpus all --user 1000:1000 \
 -e PYTHONPATH=/workspace/src -e GLM53_OBSERVER_KDA=fla \
 -e GLM53_OBSERVER_EXPERTS=native -e GLOO_SOCKET_IFNAME=tailscale0 \
 -e NCCL_SOCKET_IFNAME=tailscale0 -e NCCL_DEBUG=WARN \
 -v /home/sero/work/glm53-single-spark-release-20260911/observer-src:/workspace/src:ro \
 -v /home/sero/models/q4-stage0-obs:/model:ro \
 -v /home/sero/glm53-full-observations-20260907:/task:ro \
 -v /home/sero/work/f216-obs:/work \
 "$image" \
 -m torch.distributed.run --nnodes=2 --nproc-per-node=1 --node-rank=0 \
 --master-addr=spark-2822.internal --master-port=29641 \
 -m reap.glm53_completion_capture \
 --model-identity /task/q4-original-records.model.json \
 --token-manifest /task/ours-original-records-v1/manifest.json \
 --model-root /model --output-root "/work/completion" \
 --pause-file /work/pause-f216 --completion-start-shard 332 \
 --prefill-chunk-size 512 "${mode_args[@]}"
