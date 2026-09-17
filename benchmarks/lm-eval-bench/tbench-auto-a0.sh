#!/bin/bash
# tbench-auto-a0.sh — Mac-side watcher (3-track restructure, 2026-09-14):
# when a0's MMLU and GPQA suites have recorded exits in results/status.txt,
# launch the a0 full Terminal-Bench 2.1 leg on omarchy against the then-suite-
# exclusive a0 endpoint (spark-2384). The r216 leg runs separately on the
# dedicated spark-2822 engine. Single-shot via marker file.
set -u
B=/Users/sero/sessions/glm53-single-spark-release-20260911/benchmarks/lm-eval-bench
LOG=$B/tbench-auto-a0.log
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*" >> "$LOG"; }
MARKER=$B/.tbench-full-a0-launched
log "WATCHER_START (a0 tbench leg after a0 suites)"

while true; do
  A=$(grep -c "^2026-.*exit=[0-9]* a0$" $B/results/status.txt 2>/dev/null || true)
  AG=$(grep -c "^2026-.*exit=[0-9]* a0-gpqa$" $B/results/status.txt 2>/dev/null || true)
  log "poll a0_done=$A a0_gpqa_done=$AG"
  if [ "${A:-0}" -ge 1 ] && [ "${AG:-0}" -ge 1 ]; then break; fi
  sleep 600
done

if [ -f "$MARKER" ]; then log "already-launched marker present, exiting"; exit 0; fi
HA=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' http://spark-2384.internal:8000/health 2>/dev/null)
log "endpoint gate a0=$HA"
if [ "$HA" != "200" ]; then log "FAIL endpoint-down — a0 tbench NOT launched"; exit 3; fi
touch "$MARKER"
log "A0_SUITES_DONE launching a0 full tbench on omarchy -> 2384"
echo "$(utc) start a0-tbench" >> $B/results/status.txt
perl -e 'alarm 60; exec @ARGV' ssh -o ConnectTimeout=15 -o BatchMode=yes omarchy \
  'cd ~/terminal-bench && mkdir -p runs/full && STAMP=$(date -u +%Y%m%dT%H%M%SZ) && \
   nohup env TBENCH_DATASET=terminal-bench/terminal-bench-2-1 TBENCH_TIMEOUT_MULTIPLIER=20 \
     TBENCH_AGENT_TIMEOUT_MULTIPLIER=100 MODEL_LABEL=glm53-a0 \
     API_BASE_URL=http://spark-2384.internal:8000/v1 \
     OUT_DIR=$HOME/terminal-bench/runs/full/a0-$STAMP \
     bash framework/run.sh > runs/full/a0-$STAMP.console.log 2>&1 & \
   echo "a0-tbench pid $!" >> /tmp/tbench-launch.log'
RC=$?
log "LAUNCH ssh rc=$RC"
exit $RC
