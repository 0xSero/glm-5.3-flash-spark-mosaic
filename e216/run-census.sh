#!/bin/bash
set -x
IMG=ghcr.io/0xsero/glm53-flash-exl3-plain@sha256:85cb3fa86d31a781b94dcf10ee168adf096cfeaac14d2f1e6c560504e58e4eed
DST=/home/valentine/models/glm53-3p05-pruned-e216; W=/home/valentine/e216
R() { docker run --rm -v /home/valentine/models:/home/valentine/models:ro -v $W:$W --entrypoint python3 $IMG "$@"; }
echo "### pruned e216 stock census $(date -u +%FT%TZ)"; R -m exl3_plain_sglang_overlay.census $DST --out $W/census-pruned-e216-stock.json; echo "exit=$?"
echo "### pruned e216 plan-aware census"; R $W/census_plan_aware.py $DST $W/census-pruned-e216-plan-aware.json; echo "exit=$?"
echo CENSUS_DONE
