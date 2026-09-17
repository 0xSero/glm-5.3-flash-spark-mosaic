#!/bin/bash
# MOSAIC Gate A build driver (spark-557f). CPU only. Creates a NEW artifact dir; sources never modified.
set -x
W=/home/valentine/mosaic-gatea
A=/home/valentine/models/turboderp-glm53-flash-exl3-2p05bpw
B=/home/valentine/models/turboderp-glm53-flash-exl3-3p05bpw
DST=/home/valentine/models/mosaic-l20-gatea
echo "### preflight $(date -u +%FT%TZ)"
test -s $A/model.safetensors.index.json || { echo A0_MISSING_ABORT; exit 1; }
test -s $B/model.safetensors.index.json || { echo BASE_MISSING_ABORT; exit 1; }
test -s $W/mosaic-gatea-plan.json || { echo PLAN_MISSING_ABORT; exit 1; }
test -s $W/mosaic_pack.py || { echo PACKER_MISSING_ABORT; exit 1; }
test ! -e $DST || { echo DST_EXISTS_ABORT; exit 1; }
df -h / | tail -1
echo "### copy a0 -> $DST $(date -u +%FT%TZ)"
cp -a $A $DST || { echo COPY_FAILED; exit 1; }
du -sb $DST
echo "### pack (rewrite one shard + quantization_config) $(date -u +%FT%TZ)"
python3 $W/mosaic_pack.py --a0 $A --base $B --out $DST --plan $W/mosaic-gatea-plan.json --receipt $W/mosaic-gatea-pack-receipt.json
RC=$?; echo "pack_rc=$RC"
[ $RC -eq 0 ] || { echo PACK_FAILED_ABORT; exit 1; }
echo "### shard shas $(date -u +%FT%TZ)"
( cd $A && sha256sum model-*-of-00012.safetensors ) > $W/shas-a0-shards.txt
( cd $DST && sha256sum model-*-of-00012.safetensors ) > $W/shas-out-shards.txt
DIFFCOUNT=$(diff $W/shas-a0-shards.txt $W/shas-out-shards.txt | grep -c '^<')
echo "shards_differing_from_a0=$DIFFCOUNT"
if cmp -s $A/model.safetensors.index.json $DST/model.safetensors.index.json; then echo INDEX_BYTE_IDENTICAL; else echo INDEX_DIFFERS_ABORT; exit 1; fi
du -sb $DST > $W/du-out.txt; cat $W/du-out.txt
echo "### BUILD_DONE $(date -u +%FT%TZ)"