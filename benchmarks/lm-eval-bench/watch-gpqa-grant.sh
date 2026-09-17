#!/bin/bash
# watch-gpqa-grant.sh — polls the gated GPQA file endpoint (pure HTTP, token from file, never
# printed). The account (0xSero) must accept the gate at
# https://huggingface.co/datasets/Idavidrein/gpqa — once the data file returns HTTP 200 this
# launches run-gpqa.sh for both points and exits. Poll every 5 min.
B=$HOME/sessions/glm53-single-spark-release-20260911/benchmarks/lm-eval-bench
LOG=$B/watch-gpqa-grant.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "WATCH_START"
while true; do
  C=$(curl -sS -m 15 -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
    https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_diamond.csv 2>/dev/null)
  log "poll http=$C"
  if [ "$C" = 200 ]; then break; fi
  sleep 300
done
log "GRANTED launching gpqa runners for both points"
nohup bash $B/run-gpqa.sh a0 http://spark-2384.internal:8000 2880 >> $B/run-gpqa-a0.nohup 2>&1 &
nohup bash $B/run-gpqa.sh r216 http://spark-557f.internal:8000 2880 >> $B/run-gpqa-r216.nohup 2>&1 &
log "launched; watch exits"
