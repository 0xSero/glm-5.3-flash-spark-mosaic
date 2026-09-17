#!/bin/bash
set -x
IMG=ghcr.io/0xsero/glm53-flash-exl3-plain@sha256:85cb3fa86d31a781b94dcf10ee168adf096cfeaac14d2f1e6c560504e58e4eed
DST=/home/valentine/models/glm53-3p05-pruned-s216; W=/home/valentine/s216
R() { docker run --rm -v /home/valentine/models:/home/valentine/models:ro -v $W:$W --entrypoint python3 $IMG "$@"; }
echo "### pruned s216 stock census"; R -m exl3_plain_sglang_overlay.census $DST --out $W/census-pruned-s216-stock.json; echo "exit=$?"
echo "### pruned s216 plan-aware census"; R $W/census_plan_aware.py $DST $W/census-pruned-s216-plan-aware.json; echo "exit=$?"
echo CENSUS_DONE
