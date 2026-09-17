#!/bin/bash
# M288-10L full G4 panel (32 rows / 65,504 positions) vs the recorded a0 row; panel only (no samples leg).
# Served on spark-557f as container glm53-mosaic-10l (mf 0.90). Runs ON spark-2384.

# Mirrors f216/run-f216-g4.sh (candidate -> teacher-row symlinks -> compare).
# Runs ON spark-2384. All outputs are NEW paths; nothing existing is overwritten.
set -u
IMG=glm53-flash-sglang-exl3-plain:serve4
IMG_SHA=c65c840f1908
OUT=/home/sero/w2port/out
LOG=/home/sero/w2port/out/panel-10l.log
utc() { date -u +%FT%TZ; }
exec >> "$LOG" 2>&1
echo "PANEL_10L_START $(utc)"

R() { docker run --rm --network host -v /home/sero/w2port:/w2port -v /home/sero/models-2p05-stage/2p05bpw:/model:ro -v /home/sero/work/glm53-single-spark-release-20260911/bf16-normalized:/teacher-hidden:ro --entrypoint python3 $IMG "$@"; }

MOSAIC10_POINT="M288-10L-GateA3 mixed-precision mosaic (2.05bpw dressing + layers 3,33,37,38,39,40,41,42,43,44 at 3bpw; all 288 experts) served on spark-557f (container glm53-mosaic-10l, mem-fraction 0.90, KV 209024); plan 2dd29c133f52318d8d1e49b92265aa49e4a5b1a539adc67941d9c36c575bb474"
A0_POINT="a0 reference: 2.05bpw unpruned, served on spark-2384 (container glm53-a0-bench), measured from spark-2384"

run_point() {  # $1=name  $2=url  $3=point
  local NAME=$1 URL=$2 POINT=$3
  mkdir -p /home/sero/w2port/g4/panel-$NAME
  echo "### $NAME candidate $(utc)"
  R /w2port/g4/g4_panel.py --phase candidate --url "$URL" --out /w2port/g4/panel-$NAME \
      --receipt /w2port/out/$NAME-panel.json --topk-cand 2048 --point "$POINT"
  echo "$NAME candidate exit=$?"
  docker run --rm -w /w2port/g4/panel-$NAME -v /home/sero/w2port:/w2port --entrypoint bash $IMG \
      -c "rm -f teacher-row-*.pt; ln -sf ../panel/teacher-row-*.pt .; ls | grep -c 'teacher-row-[0-9]'"
  echo "$NAME symlink exit=$?"
  echo "### $NAME compare $(utc)"
  R /w2port/g4/g4_panel.py --phase compare --url "$URL" --out /w2port/g4/panel-$NAME \
      --receipt /w2port/out/$NAME-panel.json --topk-cand 2048 --point "$POINT"
  echo "$NAME compare exit=$?"
  echo "### $NAME POINT_DONE $(utc)"
}

run_point mosaic10l-full http://spark-557f.internal:8000 "$MOSAIC10_POINT"
echo "PANEL_10L_DONE $(utc)"
