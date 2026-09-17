#!/bin/bash
# tbench-auto.sh — Mac-side watcher: when BOTH points' MMLU and GPQA lm-eval
# suites have recorded their exits in results/status.txt, launch the two full
# Terminal-Bench 2.1 runs on omarchy (one per served endpoint, parallel).
# Score-bearing runs only fire after suites complete (endpoint contention guard
# in TBENCH-PREREG.json). Single-shot guard via marker file.
set -u
B=/Users/sero/sessions/glm53-single-spark-release-20260911/benchmarks/lm-eval-bench
LOG=$B/tbench-auto.log
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*" >> "$LOG"; }
MARKER=$B/.tbench-full-launched
log "WATCHER_START (full tbench auto-launch after suites)"

while true; do
  A=$(grep -c "^2026-.*exit=[0-9]* a0$" $B/results/status.txt 2>/dev/null || true)
  R=$(grep -c "^2026-.*exit=[0-9]* r216$" $B/results/status.txt 2>/dev/null || true)
  AG=$(grep -c "^2026-.*exit=[0-9]* a0-gpqa$" $B/results/status.txt 2>/dev/null || true)
  RG=$(grep -c "^2026-.*exit=[0-9]* r216-gpqa$" $B/results/status.txt 2>/dev/null || true)
  log "poll suites_done a0=$A r216=$R a0-gpqa=$AG r216-gpqa=$RG"
  if [ "${A:-0}" -ge 1 ] && [ "${R:-0}" -ge 1 ] && [ "${AG:-0}" -ge 1 ] && [ "${RG:-0}" -ge 1 ]; then
    break
  fi
  sleep 600
done

if [ -f "$MARKER" ]; then log "already-launched marker present, exiting"; exit 0; fi
HA=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' http://spark-2384.internal:8000/health 2>/dev/null)
HR=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' http://spark-557f.internal:8000/health 2>/dev/null)
log "endpoint gates a0=$HA r216=$HR"
if [ "$HA" != "200" ] || [ "$HR" != "200" ]; then log "FAIL endpoint-down — tbench NOT launched"; exit 3; fi
touch "$MARKER"
log "ALL_SUITES_DONE launching full tbench runs on omarchy"
echo "$(utc) start a0-tbench" >> $B/results/status.txt
echo "$(utc) start r216-tbench" >> $B/results/status.txt
perl -e 'alarm 60; exec @ARGV' ssh -o ConnectTimeout=15 -o BatchMode=yes omarchy \
  'cd ~/terminal-bench && mkdir -p runs/full && STAMP=$(date -u +%Y%m%dT%H%M%SZ) && \
   nohup env TBENCH_DATASET=terminal-bench/terminal-bench-2-1 TBENCH_TIMEOUT_MULTIPLIER=20 \
     TBENCH_AGENT_TIMEOUT_MULTIPLIER=100 MODEL_LABEL=glm53-a0 \
     API_BASE_URL=http://spark-2384.internal:8000/v1 \
     OUT_DIR=$HOME/terminal-bench/runs/full/a0-$STAMP \
     bash framework/run.sh > runs/full/a0-$STAMP.console.log 2>&1 & \
   echo "a0-tbench pid $!" >> /tmp/tbench-launch.log; \
   nohup env TBENCH_DATASET=terminal-bench/terminal-bench-2-1 TBENCH_TIMEOUT_MULTIPLIER=20 \
     TBENCH_AGENT_TIMEOUT_MULTIPLIER=100 MODEL_LABEL=glm53-r216 \
     API_BASE_URL=http://spark-557f.internal:8000/v1 \
     OUT_DIR=$HOME/terminal-bench/runs/full/r216-$STAMP \
     bash framework/run.sh > runs/full/r216-$STAMP.console.log 2>&1 & \
   echo "r216-tbench pid $!" >> /tmp/tbench-launch.log'
RC=$?
log "LAUNCH ssh rc=$RC"
exit $RC
