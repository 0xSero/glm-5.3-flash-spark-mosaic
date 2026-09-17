#!/bin/bash
set -x
IMG=glm53-flash-sglang-exl3-plain:serve4
URL=http://spark-557f.internal:8000
POINT="T216: exl3_plain 3.05bpw pruned uniform-216 by K3-sensitivity only (glm53-3p05-pruned-t216, plan 4e4d8973661b414cb0f2bc8496e756d41b25f1dd6a8e2299d5bcb97c574f492a) served on spark-557f (container glm53-t216, image serve4-pruned b9cf5753), measured from spark-2384"
R() { docker run --rm --network host -v /home/sero/w2port:/w2port -v /home/sero/models-2p05-stage/2p05bpw:/model:ro -v /home/sero/work/glm53-single-spark-release-20260911/bf16-normalized:/teacher-hidden:ro --entrypoint python3 $IMG "$@"; }
B() { docker run --rm -v /home/sero/w2port:/w2port --entrypoint bash $IMG "$@"; }
echo "### candidate $(date -u +%FT%TZ)"
R /w2port/g4/g4_panel.py --phase candidate --url $URL --out /w2port/g4/panel-t216 --receipt /w2port/out/t216-g4-panel.json --topk-cand 2048 --point "$POINT" 2>&1 | grep -v "UserWarning\|warnings.warn"; echo "candidate exit=${PIPESTATUS[0]}"
echo "### teacher-row symlinks (must run inside a container; out dir is docker-root-owned)"
B -c "ln -sf ../panel/teacher-row-*.pt /w2port/g4/panel-t216/"; echo "symlink exit=$?"
echo "### compare $(date -u +%FT%TZ)"
R /w2port/g4/g4_panel.py --phase compare --url $URL --out /w2port/g4/panel-t216 --receipt /w2port/out/t216-g4-panel.json --topk-cand 2048 --point "$POINT" 2>&1 | grep -v "UserWarning\|warnings.warn"; echo "compare exit=${PIPESTATUS[0]}"
echo "### samples $(date -u +%FT%TZ)"
R /w2port/g4/g4_samples.py --url $URL --out-dir /w2port/g4/samples-t216 --receipt /w2port/out/t216-g4-samples.json --point "$POINT" 2>&1 | grep -v "UserWarning\|warnings.warn"; echo "samples exit=${PIPESTATUS[0]}"
echo "### summary $(date -u +%FT%TZ)"
R /w2port/g4/g4_summary.py --panel /w2port/out/t216-g4-panel.json --samples /w2port/out/t216-g4-samples.json --out /w2port/out/t216-g4-summary.json --image glm53-flash-sglang-exl3-plain:serve4-pruned --image-sha b9cf57530a41; echo "summary exit=${PIPESTATUS[0]}"
echo "T216_G4_DONE $(date -u +%FT%TZ)"
