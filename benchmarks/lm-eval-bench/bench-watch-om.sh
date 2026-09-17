#!/bin/bash
# Same wait-validate-launch loop as bench-watch.sh but RUNNING ON OMARCHY (its ssh legs to the
# sparks are reliable; the Mac->tailscale ssh wedged in D-state twice today).
set -u
LOG=$HOME/bench-watch.log
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*" >> "$LOG"; }
log "OM_WATCHER_START"
READY=""
for i in $(seq 1 30); do
  sleep 45
  R=$(ssh -o ConnectTimeout=8 -o BatchMode=yes sero@spark-2384.internal 'docker logs glm53-a0-bench 2>&1 | grep -c "fired up and ready"' 2>/dev/null | tail -1)
  A=$(ssh -o ConnectTimeout=8 -o BatchMode=yes valentine@spark-557f.internal 'docker logs glm53-r216-bench 2>&1 | grep -c "fired up and ready"' 2>/dev/null | tail -1)
  log "poll $i ready_a0=${R:-0} ready_r216=${A:-0}"
  if [ "${R:-0}" -ge 1 ] 2>/dev/null && [ "${A:-0}" -ge 1 ] 2>/dev/null; then
    KVA=$(ssh -o ConnectTimeout=8 -o BatchMode=yes sero@spark-2384.internal 'docker logs glm53-a0-bench 2>&1 | grep -m1 -o "max_total_num_tokens=[0-9]*"' 2>/dev/null | tail -1)
    KVR=$(ssh -o ConnectTimeout=8 -o BatchMode=yes valentine@spark-557f.internal 'docker logs glm53-r216-bench 2>&1 | grep -m1 -o "max_total_num_tokens=[0-9]*"' 2>/dev/null | tail -1)
    log "BOTH_READY a0_kv=$KVA r216_kv=$KVR"
    READY=1; break
  fi
done
[ -n "$READY" ] || { log "FAIL servers-not-ready"; exit 1; }
export PATH=$HOME/lm-eval-venv/bin:$PATH
V=$(timeout 1200 lm_eval run -M local-completions \
  -a "base_url=http://spark-557f.internal:8000,tokenizer=/home/sero/glm53-tokenizer,api_key=local,num_concurrent=2,max_retries=2,timeout=300" \
  -t gpqa_diamond_zeroshot -L 2 --seed 1234 --output_path /tmp/lmeval-validate 2>&1 | tail -8)
log "VALIDATE_TAIL: $(echo "$V" | tr '\n' ' ' | head -c 500)"
if echo "$V" | grep -qiE "gated|denied|401|403"; then log "VALIDATE_BLOCKED gpqa-access"; exit 2; fi
if echo "$V" | grep -qE '"acc' ; then
  log "VALIDATE_OK launching full runs"
  nohup bash $HOME/bench-runner.sh > /dev/null 2>&1 &
  log "RUNNER_LAUNCHED"
else
  log "VALIDATE_UNCLEAR not launching"
  exit 3
fi
