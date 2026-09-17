#!/bin/bash
# a0 samples re-run (r2) after the 2026-09-16T21:41:41Z NCCL crash killed the a0 server mid-samples.
# The a0 PANEL row from the first run stays valid (it completed, teacher-exact); only the samples
# leg was invalidated (16/20 requests died on ConnectionResetError against the dead server).
# Runs ON spark-2384 against the restarted glm53-a0-bench (identical flags). New receipt paths only.
set -u
IMG=glm53-flash-sglang-exl3-plain:serve4
OUT=/home/sero/w2port/out
LOG=$OUT/a0-samples-r2.log
utc() { date -u +%FT%TZ; }
exec >> "$LOG" 2>&1
echo "A0_SAMPLES_R2_START $(utc)"
R() { docker run --rm --network host -v /home/sero/w2port:/w2port -v /home/sero/models-2p05-stage/2p05bpw:/model:ro -v /home/sero/work/glm53-single-spark-release-20260911/bf16-normalized:/teacher-hidden:ro --entrypoint python3 $IMG "$@"; }
POINT="a0 reference: 2.05bpw unpruned, served on spark-2384 (container glm53-a0-bench, mf 0.90, req 16), samples re-run r2 after the NCCL crash of 2026-09-16T21:41:41Z; measured from spark-2384"

READY=0
for i in $(seq 1 60); do
  C=$(curl -s -m 8 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health 2>/dev/null)
  echo "poll $i health=$C $(utc)"
  if [ "$C" = "200" ]; then READY=1; break; fi
  sleep 30
done
[ "$READY" = "1" ] || { echo "A0_SAMPLES_R2_ABORT not-ready"; exit 1; }
echo "### samples $(utc)"
R /w2port/g4/g4_samples.py --url http://127.0.0.1:8000 --out-dir /w2port/g4/samples-a0-r2 \
    --receipt /w2port/out/a0-full-samples-r2.json --point "$POINT"
echo "a0-full samples r2 exit=$?"
echo "### summary r2 $(utc)"
R /w2port/g4/g4_summary.py --panel /w2port/out/a0-full-panel.json \
    --samples /w2port/out/a0-full-samples-r2.json \
    --out /w2port/out/a0-full-summary-r2.json \
    --image glm53-flash-sglang-exl3-plain:serve4 --image-sha c65c840f1908
echo "a0-full summary r2 exit=$?"
echo "A0_SAMPLES_R2_DONE $(utc)"