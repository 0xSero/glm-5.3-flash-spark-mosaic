#!/bin/bash
# bench-runner.sh v2: waits for both bench servers READY (timeout-wrapped ssh probes from omarchy),
# validates lm-eval with a 2-doc GPQA run, then launches the full suite on both points in parallel.
set -u
export PATH=$HOME/lm-eval-venv/bin:$PATH
LOG=$HOME/bench-runner.log
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*" >> "$LOG"; }
probe() { timeout 20 ssh -o ConnectTimeout=8 -o BatchMode=yes "$1" "$2" 2>/dev/null | tail -1; }
log "RUNNER_V2_START"
READY=""
for i in $(seq 1 40); do
  sleep 30
  R=$(probe valentine@spark-557f.internal 'docker logs glm53-r216-bench 2>&1 | grep -c "fired up and ready"')
  A=$(probe sero@spark-2384.internal 'docker logs glm53-a0-bench 2>&1 | grep -c "fired up and ready"')
  log "poll $i r216=${R:-0} a0=${A:-0}"
  if [ "${R:-0}" -ge 1 ] 2>/dev/null && [ "${A:-0}" -ge 1 ] 2>/dev/null; then
    KVR=$(probe valentine@spark-557f.internal 'docker logs glm53-r216-bench 2>&1 | grep -m1 -o "max_total_num_tokens=[0-9]*"')
    KVA=$(probe sero@spark-2384.internal 'docker logs glm53-a0-bench 2>&1 | grep -m1 -o "max_total_num_tokens=[0-9]*"')
    log "BOTH_READY r216_kv=$KVR a0_kv=$KVA"
    READY=1; break
  fi
done
[ -n "$READY" ] || { log "FAIL not-ready"; exit 1; }
V=$(timeout 1500 lm_eval run -M local-completions \
  -a "base_url=http://spark-557f.internal:8000,tokenizer=/home/sero/glm53-tokenizer,api_key=local,num_concurrent=2,max_retries=2,timeout=300" \
  -t gpqa_diamond_zeroshot -L 2 --seed 1234 --output_path /tmp/lmeval-validate 2>&1 | tail -8)
log "VALIDATE_TAIL: $(echo "$V" | tr '\n' ' ' | head -c 500)"
echo "$V" | grep -qiE "gated|denied|401|403" && { log "BLOCKED gpqa-access"; exit 2; }
echo "$V" | grep -qE '"acc' || { log "VALIDATE_UNCLEAR not launching"; exit 3; }
log "VALIDATE_OK launching full runs"
mkdir -p ~/bench-results
ARGS_COMMON="tokenizer=/home/sero/glm53-tokenizer,api_key=local,num_concurrent=16,max_retries=5,timeout=1800"
full() {
  local name=$1 url=$2
  echo "$(utc) start $name" >> ~/bench-results/status.txt
  lm_eval run -M local-completions \
    -a "base_url=$url,$ARGS_COMMON" \
    -t mmlu gpqa_diamond_zeroshot gpqa_diamond_cot_zeroshot \
    --seed 1234 --log_samples \
    --output_path ~/bench-results/$name \
    > ~/bench-results/$name.log 2>&1
  echo "$(utc) exit=$? $name" >> ~/bench-results/status.txt
}
full r216 "http://spark-557f.internal:8000" &
full a0 "http://spark-2384.internal:8000" &
wait
echo "$(utc) RUNNER_DONE" >> "$LOG"
echo "$(utc) RUNNER_DONE" >> ~/bench-results/status.txt
