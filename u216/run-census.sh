#!/bin/bash
set -x
IMG=ghcr.io/0xsero/glm53-flash-exl3-plain@sha256:85cb3fa86d31a781b94dcf10ee168adf096cfeaac14d2f1e6c560504e58e4eed
DST=/home/valentine/models/glm53-3p05-pruned-u216; W=/home/valentine/u216
R() { docker run --rm -v /home/valentine/models:/home/valentine/models:ro -v $W:$W --entrypoint python3 $IMG "$@"; }
echo "### pruned u216 stock census"; R -m exl3_plain_sglang_overlay.census $DST --out $W/census-pruned-u216-stock.json; echo "exit=$?"
echo "### pruned u216 plan-aware census"; R $W/census_plan_aware.py $DST $W/census-pruned-u216-plan-aware.json; echo "exit=$?"
echo CENSUS_DONE
