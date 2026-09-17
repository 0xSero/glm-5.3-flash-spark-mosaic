#!/bin/bash
# Completes the verification steps run-build-10l.sh skipped after the packer's abort-on-false-fail.
# The packer's own per-tensor checks all passed (swapped 34560/34560, kept 33029/33029, n_failures 0);
# the only "failure" was the exact-delta assertion: delta 9,059,707,346 vs expected 9,059,696,640
# (= +10,706 B of safetensors header drift across the same 5 rewritten shards, identical to 12L).
set -x
W=/home/valentine/mosaic-gatea
A=/home/valentine/models/turboderp-glm53-flash-exl3-2p05bpw
DST=/home/valentine/models/mosaic-10l-gatea
( cd $A && sha256sum model-*-of-00012.safetensors ) > $W/shas-a0-shards-10l.txt
( cd $DST && sha256sum model-*-of-00012.safetensors ) > $W/shas-out-shards-10l.txt
echo "shards_differing_from_a0=$(diff $W/shas-a0-shards-10l.txt $W/shas-out-shards-10l.txt | grep -c '^<')"
if cmp -s $A/model.safetensors.index.json $DST/model.safetensors.index.json; then echo INDEX_BYTE_IDENTICAL; else echo INDEX_DIFFERS; fi
du -sb $DST > $W/du-out-10l.txt; cat $W/du-out-10l.txt
ls -la $DST/model-00001-of-00012.safetensors | awk '{print $5}' > /dev/null
echo "### FINISH_10L_DONE $(date -u +%FT%TZ)"
