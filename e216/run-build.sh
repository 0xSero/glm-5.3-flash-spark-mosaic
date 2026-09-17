#!/bin/bash
set -x
W=/home/valentine/e216; SRC=/home/valentine/models/turboderp-glm53-flash-exl3-3p05bpw; DST=/home/valentine/models/glm53-3p05-pruned-e216
cd $W
echo "### pack $(date -u +%FT%TZ)"
python3 $W/prune_u216.py --src $SRC --dst $DST --plan $W/e216-plan.json --workers 5 2>&1 | tee $W/pack.log; echo "pack exit=${PIPESTATUS[0]}"
echo "### augment index (mtp + kpool_aux)"
cp $DST/model.safetensors.index.json $DST/model.safetensors.index.shards-only.json
python3 $W/augment_index.py $DST $DST/model.safetensors.index.json mtp.safetensors kpool_aux.safetensors | tee $W/augment.log; echo "augment exit=${PIPESTATUS[0]}"
echo "### G2 verify $(date -u +%FT%TZ)"
python3 $W/verify_u216.py $SRC $DST 5 2>&1 | tee $W/verify.log; echo "verify exit=${PIPESTATUS[0]}"
du -sb $DST; ls -la $DST
echo "BUILD_DONE $(date -u +%FT%TZ)"
