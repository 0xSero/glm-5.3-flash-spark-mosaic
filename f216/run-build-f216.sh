#!/usr/bin/env bash
# F216 build (stage 3, spark-2822) — pack → index augment → G2 verify.
# Template: 557f e216/run-build.sh (verbatim flow, F216 paths + plan).
set -euo pipefail
W=/home/sero/work/f216-build
SRC=/home/sero/models/turboderp-glm53-flash-exl3-3p05bpw
DST=/home/sero/models/glm53-3p05-pruned-f216
PLAN=$W/f216-plan.json
python3 $W/prune_u216.py --src $SRC --dst $DST --plan $PLAN --workers 5 | tee $W/pack.log
cp $DST/model.safetensors.index.json $DST/model.safetensors.index.shards-only.json
python3 $W/augment_index.py $DST $DST/model.safetensors.index.json mtp.safetensors kpool_aux.safetensors | tee $W/augment.log
python3 $W/verify_u216.py $SRC $DST 5 | tee $W/verify.log
du -sb $DST
echo BUILD_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)
