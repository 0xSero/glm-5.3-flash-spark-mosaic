#!/bin/bash
# run-e216-gpqa-mc.sh — evidence-matrix completion: E216 GPQA MC (zeroshot) only,
# no CoT leg. Same gates + same model_args as prereg GPQA runs (run-gpqa.sh).
# Engine: glm53-e216 on 557f (http://spark-557f.internal:8000). 198 docs = 792 requests.
set -u
NAME=e216-gpqa-mc; URL=http://spark-557f.internal:8000
V=$HOME/lmeval-mac-venv/bin
B=$HOME/sessions/glm53-single-spark-release-20260911/benchmarks/lm-eval-bench
LOG=$B/run-$NAME.log; OUT=$B/results
mkdir -p $OUT
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*" >> "$LOG"; }
log "RUNNER_START $NAME $URL (MC-only evidence-matrix leg)"
READY=""
for i in $(seq 1 20); do
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
  --include_path $B/tasks-gpqa-local \
  -t gpqa_diamond_zeroshot --limit 2 --seed 1234 --output_path $OUT/validate-$NAME \
  > $OUT/validate-$NAME.log 2>&1
RC=$?
log "VALIDATE rc=$RC tail: $(tail -c 300 $OUT/validate-$NAME.log | tr '\n' ' ')"
grep -qE '\|acc' $OUT/validate-$NAME.log || { log "VALIDATE_UNCLEAR not launching"; exit 3; }

log "VALIDATE_OK launching GPQA MC (198 docs, no limit)"
echo "$(utc) start $NAME" >> $OUT/status.txt
$V/lm_eval run -M local-completions \
  -a "model=bench-$NAME,base_url=$URL/v1/completions,tokenizer=$TOK,api_key=local,num_concurrent=16,max_retries=5,timeout=1800" \
  --include_path $B/tasks-gpqa-local \
  -t gpqa_diamond_zeroshot \
  --seed 1234 --log_samples --output_path $OUT/$NAME \
  > $OUT/$NAME.log 2>&1
echo "$(utc) exit=$? $NAME" >> $OUT/status.txt
log "RUNNER_DONE exit-recorded"
