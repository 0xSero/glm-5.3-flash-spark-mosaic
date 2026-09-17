#!/bin/bash
# Gate A mini-panel: mosaic-12l (557f) vs a0 (2384 local), frozen rows 0-3 = 8192 positions.
set -x
IMG=glm53-flash-sglang-exl3-plain:serve4
R() { docker run --rm --network host -v /home/sero/w2port:/w2port -v /home/sero/models-2p05-stage/2p05bpw:/model:ro -v /home/sero/work/glm53-single-spark-release-20260911/bf16-normalized:/teacher-hidden:ro --entrypoint python3 $IMG "$@"; }
B() { docker run --rm -w "$1" -v /home/sero/w2port:/w2port --entrypoint bash $IMG -c "$2"; }
MOSAIC_POINT="M288-12L-GateA2 mixed-precision mosaic (2.05bpw dressing + layers 3,32,33,36-44 at 3bpw; all 288 experts) served on spark-557f; plan 7938939c3a5b1cbaaa120ff1fdb51e7d8171f1f46c18630e59727812e969204e"
A0_POINT="a0 reference: 2.05bpw unpruned, served on spark-2384"

echo "### mosaic candidate $(date -u +%FT%TZ)"
R /w2port/g4/g4_panel.py --phase candidate --url http://spark-557f.internal:8000 --out /w2port/g4/panel-mosaic12l-mini --receipt /w2port/out/mosaic12l-mini-panel.json --topk-cand 2048 --rows 0-3 --point "$MOSAIC_POINT" 2>&1 | grep -v "UserWarning\|warnings.warn" | tail -2
echo "mosaic candidate exit=${PIPESTATUS[0]}"
B /w2port/g4/panel-mosaic12l-mini 'rm -f teacher-row-*.pt; ln -sf ../panel/teacher-row-00[0-3].pt .; ls teacher-row-* | wc -l'
echo "### mosaic compare $(date -u +%FT%TZ)"
R /w2port/g4/g4_panel.py --phase compare --url http://spark-557f.internal:8000 --out /w2port/g4/panel-mosaic12l-mini --receipt /w2port/out/mosaic12l-mini-panel.json --topk-cand 2048 --rows 0-3 --point "$MOSAIC_POINT" 2>&1 | grep -v "UserWarning\|warnings.warn" | tail -2
echo "mosaic compare exit=${PIPESTATUS[0]}"

echo "### a0 candidate $(date -u +%FT%TZ)"
R /w2port/g4/g4_panel.py --phase candidate --url http://127.0.0.1:8000 --out /w2port/g4/panel-a0-mini --receipt /w2port/out/a0-mini-panel.json --topk-cand 2048 --rows 0-3 --point "$A0_POINT" 2>&1 | grep -v "UserWarning\|warnings.warn" | tail -2
echo "a0 candidate exit=${PIPESTATUS[0]}"
B /w2port/g4/panel-a0-mini 'rm -f teacher-row-*.pt; ln -sf ../panel/teacher-row-00[0-3].pt .; ls teacher-row-* | wc -l'
echo "### a0 compare $(date -u +%FT%TZ)"
R /w2port/g4/g4_panel.py --phase compare --url http://127.0.0.1:8000 --out /w2port/g4/panel-a0-mini --receipt /w2port/out/a0-mini-panel.json --topk-cand 2048 --rows 0-3 --point "$A0_POINT" 2>&1 | grep -v "UserWarning\|warnings.warn" | tail -2
echo "a0 compare exit=${PIPESTATUS[0]}"
echo "### MINI_PANELS_DONE $(date -u +%FT%TZ)"
