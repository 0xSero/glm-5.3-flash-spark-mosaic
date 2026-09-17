#!/bin/bash
# run-mosaic12l-suite.sh — Mac-side driver for the M288-12L mosaic confirming run.
#   0) wait for the full-G4-panel marker on spark-2384 (panel uses the same 557f endpoint;
#      the mosaic serves ONE request at a time at panel-tune knobs, so keep 557f legs serialized)
#   1) quick MMLU  (limit-20/subject, 1,140 docs)  -> like-for-like vs a0-quick 0.8342
#   2) GPQA MC     (gpqa_diamond_zeroshot, 198 docs) -> like-for-like vs a0 43.43
# Pure HTTP. Launched detached; poll run-mosaic12l-suite.log for the DONE line.
set -u
B=$HOME/sessions/glm53-single-spark-release-20260911/benchmarks/lm-eval-bench
LOG=$B/run-mosaic12l-suite.log
SSH="ssh -o ControlMaster=auto -o ControlPath=/tmp/ssh-2384.sock -o ControlPersist=900 -o ConnectTimeout=6 -o BatchMode=yes sero@spark-2384.internal"
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*" >> "$LOG"; }
log "SUITE_START"

# --- step 0: wait for the 2384 panel run to finish (bounded: 24 x 5 min = 2 h) ---
WAITED=0
for i in $(seq 1 24); do
  D=$(perl -e 'alarm 30; exec @ARGV' $SSH 'grep -c "mosaic12l-full POINT_DONE" /home/sero/w2port/out/full-panel-both.log 2>/dev/null' 2>/dev/null | tail -1)
  if [ "${D:-0}" = "1" ]; then log "MOSAIC_PANEL_DONE after $((i-1)) waits"; WAITED=1; break; fi
  log "wait $i panel marker=${D:-ssh-fail}"
  sleep 300
done
[ "$WAITED" = "1" ] || log "WARN panel marker not seen after 2 h — starting suite anyway"

log "leg1 quick MMLU start"
bash $B/run-mosaic12l-quick.sh
log "leg1 quick MMLU done"

log "leg2 gpqa MC start"
bash $B/run-mosaic12l-gpqa-mc.sh
log "leg2 gpqa MC done"

log "SUITE_DONE"