#!/bin/bash
# MOSAIC 12-layer build driver (spark-557f). CPU only. New artifact dir; sources never modified.
set -x
W=/home/valentine/mosaic-gatea
A=/home/valentine/models/turboderp-glm53-flash-exl3-2p05bpw
B=/home/valentine/models/turboderp-glm53-flash-exl3-3p05bpw
DST=/home/valentine/models/mosaic-12l-gatea
echo "### preflight $(date -u +%FT%TZ)"
test -s $W/mosaic-12l-plan.json || { echo PLAN_MISSING_ABORT; exit 1; }
test -s $W/mosaic_pack2.py || { echo PACKER_MISSING_ABORT; exit 1; }
test ! -e $DST || { echo DST_EXISTS_ABORT; exit 1; }
df -h / | tail -1
echo "### copy a0 -> $DST $(date -u +%FT%TZ)"
cp -a $A $DST || { echo COPY_FAILED; exit 1; }
echo "### pack $(date -u +%FT%TZ)"
python3 $W/mosaic_pack2.py --a0 $A --base $B --out $DST --plan $W/mosaic-12l-plan.json --receipt $W/mosaic-12l-pack-receipt.json
RC=$?; echo "pack_rc=$RC"
[ $RC -eq 0 ] || { echo PACK_FAILED_ABORT; exit 1; }
echo "### shard shas $(date -u +%FT%TZ)"
( cd $A && sha256sum model-*-of-00012.safetensors ) > $W/shas-a0-shards-12l.txt
( cd $DST && sha256sum model-*-of-00012.safetensors ) > $W/shas-out-shards-12l.txt
echo "shards_differing_from_a0=$(diff $W/shas-a0-shards-12l.txt $W/shas-out-shards-12l.txt | grep -c '^<')"
if cmp -s $A/model.safetensors.index.json $DST/model.safetensors.index.json; then echo INDEX_BYTE_IDENTICAL; else echo INDEX_DIFFERS_ABORT; exit 1; fi
du -sb $DST > $W/du-out-12l.txt; cat $W/du-out-12l.txt
echo "### BUILD_12L_DONE $(date -u +%FT%TZ)"
