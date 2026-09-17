#!/bin/bash
# run-f216-quick.sh — F216 quick MMLU, like-for-like vs e216-quick (57 subjects x limit 20 = 1,140 docs, seed 1234).
# Same model_args as e216-quick (receipt results/e216-quick/bench-e216/results_*.json).
# Engine: glm53-f216 on 557f (http://spark-557f.internal:8000).
set -u
NAME=f216-quick; URL=http://spark-557f.internal:8000
V=$HOME/lmeval-mac-venv/bin
B=$HOME/sessions/glm53-single-spark-release-20260911/benchmarks/lm-eval-bench
LOG=$B/run-$NAME.log; OUT=$B/results
mkdir -p $OUT
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*" >> "$LOG"; }
log "RUNNER_START $NAME $URL (quick MMLU limit-20 like-for-like vs e216-quick)"
READY=""
for i in $(seq 1 60); do
  C=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' $URL/health 2>/dev/null)
  if [ "$C" = 200 ]; then
    S=$(curl -sS -m 90 -o /tmp/smoke-$NAME.json -w '%{http_code}' \
      -H 'Content-Type: application/json' \
      -d '{"model":"default","prompt":"hello","max_tokens":1,"temperature":0}' $URL/v1/completions 2>/dev/null)
    log "poll $i health=200 smoke=$S"
    if [ "$S" = 200 ] && grep -q '"choices"' /tmp/smoke-$NAME.json 2>/dev/null; then
      READY=1; log "READY_OK"; break
    fi
  else
    log "poll $i health=$C"
  fi
  sleep 60
done
[ -n "$READY" ] || { log "FAIL not-ready"; exit 1; }

TOK=$B/glm53-tokenizer
$V/lm_eval run -M local-completions \
  -a "model=bench-$NAME,base_url=$URL/v1/completions,tokenizer=$TOK,api_key=local,num_concurrent=4,max_retries=3,timeout=600" \
  -t mmlu --limit 2 --seed 1234 --output_path $OUT/validate-$NAME \
  > $OUT/validate-$NAME.log 2>&1
RC=$?
log "VALIDATE rc=$RC tail: $(tail -c 300 $OUT/validate-$NAME.log | tr '\n' ' ')"
grep -qE '\|acc' $OUT/validate-$NAME.log || { log "VALIDATE_UNCLEAR not launching"; exit 3; }

log "VALIDATE_OK launching quick MMLU (limit 20, 1,140 docs)"
echo "$(utc) start $NAME" >> $OUT/status.txt
$V/lm_eval run -M local-completions \
  -a "model=bench-$NAME,base_url=$URL/v1/completions,tokenizer=$TOK,api_key=local,num_concurrent=16,max_retries=5,timeout=1800" \
  -t mmlu --limit 20 --seed 1234 --log_samples --output_path $OUT/$NAME \
  > $OUT/$NAME.log 2>&1
echo "$(utc) exit=$? $NAME" >> $OUT/status.txt
log "RUNNER_DONE exit-recorded"
