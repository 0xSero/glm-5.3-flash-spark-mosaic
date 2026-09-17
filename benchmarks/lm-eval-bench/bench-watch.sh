#!/bin/bash
# Watch both bench servers until READY, validate lm-eval with a 2-doc GPQA run, then launch the full runs.
set -u
ROOT=/Users/sero/sessions/glm53-single-spark-release-20260911
LOG=$ROOT/benchmarks/lm-eval-bench/watcher.log
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*" >> "$LOG"; }
refresh_lock() { printf '{"started_utc": "%s", "run": "driver", "note": "benchmark campaign: lm-eval mmlu+gpqa in flight (auto-refresh)"}' "$(utc)" > "$ROOT/loop-driver/DRIVER-LOCK.json"; }
S557='ssh -o ProxyJump=none -o Hostname=spark-557f.internal -o ConnectTimeout=8 -o BatchMode=yes spark-557f-extension'
S238='ssh -o ConnectTimeout=8 -o BatchMode=yes spark-2384'
SOM='ssh -o ConnectTimeout=8 -o BatchMode=yes sero@omarchy.internal'
log "WATCHER_START (attempt-2 moderate bench-tune)"
READY=""
for i in $(seq 1 25); do
  sleep 60; refresh_lock
  R=$(perl -e 'alarm 20; exec @ARGV' $S557 'docker logs glm53-r216-bench 2>&1 | grep -c "fired up and ready"' 2>/dev/null | tail -1)
  A=$(perl -e 'alarm 20; exec @ARGV' $S238 'docker logs glm53-a0-bench 2>&1 | grep -c "fired up and ready"' 2>/dev/null | tail -1)
  log "poll $i ready_r216=${R:-0} ready_a0=${A:-0}"
  if [ "${R:-0}" -ge 1 ] 2>/dev/null && [ "${A:-0}" -ge 1 ] 2>/dev/null; then
    KV_R=$(perl -e 'alarm 20; exec @ARGV' $S557 'docker logs glm53-r216-bench 2>&1 | grep -m1 -o "max_total_num_tokens=[0-9]*"' 2>/dev/null | cut -d= -f2)
    KV_A=$(perl -e 'alarm 20; exec @ARGV' $S238 'docker logs glm53-a0-bench 2>&1 | grep -m1 -o "max_total_num_tokens=[0-9]*"' 2>/dev/null | cut -d= -f2)
    log "BOTH_READY r216_kv=${KV_R:-?} a0_kv=${KV_A:-?}"
    READY=1; break
  fi
done
[ -n "$READY" ] || { log "WATCHER_FAIL servers-not-ready-in-25min"; exit 1; }
V=$(perl -e 'alarm 900; exec @ARGV' $SOM 'export PATH=$HOME/lm-eval-venv/bin:$PATH; lm_eval run -M local-completions -a "base_url=http://spark-557f.internal:8000,tokenizer=/home/sero/glm53-tokenizer,api_key=local,num_concurrent=2,max_retries=2,timeout=300" -t gpqa_diamond_zeroshot -L 2 --seed 1234 --output_path /tmp/lmeval-validate 2>&1 | tail -6' 2>&1)
log "VALIDATE_TAIL: $(echo "$V" | tail -3 | head -c 400)"
if echo "$V" | grep -qiE "acc|Error|error|gated|denied"; then
  if echo "$V" | grep -qiE "gated|denied|401|403"; then log "VALIDATE_BLOCKED gpqa-access"; exit 2; fi
fi
if echo "$V" | grep -qE '"acc[^_]*"' ; then
  log "VALIDATE_OK - launching full runs"
  perl -e 'alarm 30; exec @ARGV' $SOM 'nohup bash ~/bench-runner.sh > /dev/null 2>&1 & echo RUNNER_LAUNCHED' >> "$LOG" 2>&1
  log "WATCHER_DONE runs launched; completion tracked via ~/bench-results/status.txt"
else
  log "VALIDATE_UNCLEAR - NOT launching; investigate watcher.log VALIDATE_TAIL"; exit 3
fi
