#!/bin/bash
# Full G4 panel (default --rows 0-31 = 32 rows x 2048 = 65,504 positions) for:
#   1) M288-12L mosaic  -> http://spark-557f.internal:8000 (spark-557f, container glm53-mosaic-12lh)
#   2) a0 baseline      -> http://127.0.0.1:8000     (spark-2384, container glm53-a0-bench)
# Mirrors f216/run-f216-g4.sh (candidate -> teacher-row symlinks -> compare -> samples -> summary).
# Runs ON spark-2384. All outputs are NEW paths; nothing existing is overwritten.
set -u
IMG=glm53-flash-sglang-exl3-plain:serve4
IMG_SHA=c65c840f1908
OUT=/home/sero/w2port/out
LOG=$OUT/full-panel-both.log
utc() { date -u +%FT%TZ; }
exec >> "$LOG" 2>&1
echo "FULL_PANEL_BOTH_START $(utc)"

R() { docker run --rm --network host -v /home/sero/w2port:/w2port -v /home/sero/models-2p05-stage/2p05bpw:/model:ro -v /home/sero/work/glm53-single-spark-release-20260911/bf16-normalized:/teacher-hidden:ro --entrypoint python3 $IMG "$@"; }

MOSAIC_POINT="M288-12L-GateA2 mixed-precision mosaic (2.05bpw dressing + layers 3,32,33,36-44 at 3bpw; all 288 experts) served on spark-557f (container glm53-mosaic-12lh, mem-fraction 0.95, KV 595200); plan 7938939c3a5b1cbaaa120ff1fdb51e7d8171f1f46c18630e59727812e969204e"
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
  echo "### $NAME samples $(utc)"
  R /w2port/g4/g4_samples.py --url "$URL" --out-dir /w2port/g4/samples-$NAME \
      --receipt /w2port/out/$NAME-samples.json --point "$POINT"
  echo "$NAME samples exit=$?"
  echo "### $NAME summary $(utc)"
  R /w2port/g4/g4_summary.py --panel /w2port/out/$NAME-panel.json --samples /w2port/out/$NAME-samples.json \
      --out /w2port/out/$NAME-summary.json --image glm53-flash-sglang-exl3-plain:serve4 --image-sha $IMG_SHA
  echo "$NAME summary exit=$?"
  echo "### $NAME POINT_DONE $(utc)"
}

run_point mosaic12l-full http://spark-557f.internal:8000 "$MOSAIC_POINT"
run_point a0-full http://127.0.0.1:8000 "$A0_POINT"
echo "FULL_PANEL_BOTH_DONE $(utc)"