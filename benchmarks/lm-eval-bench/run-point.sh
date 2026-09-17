#!/bin/bash
# run-point.sh <name> <url> — Mac-side single-point lm-eval runner.
# Pure HTTP (curl) — NO ssh legs: Mac->tailscale ssh wedges in D-state, HTTP does not.
# Gates: (1) /health 200, (2) real completion smoke returns choices, (3) 2-doc MMLU
# loglikelihood validation (same request path as GPQA MC; catches flag/API breakage),
# then the full preregistered suite. Full-run model_args are EXACTLY PREREG's.
# NOTE 2026-09-14: GPQA is gated and account 0xSero is not yet granted (PREREG-ADDENDUM.json) —
# this runner does MMLU only; run-gpqa.sh (auto-started by watch-gpqa-grant.sh on grant) does GPQA.
# NOTE: lm-eval 0.4.13 api_models POSTs to base_url VERBATIM — base_url for lm_eval must be the
# full completions endpoint (<server-root>/v1/completions); $URL stays the server root for health/smoke.
set -u
NAME=$1; URL=$2; LIMIT=${3:-480}
V=$HOME/lmeval-mac-venv/bin
B=$HOME/sessions/glm53-single-spark-release-20260911/benchmarks/lm-eval-bench
LOG=$B/run-$NAME.log; OUT=$B/results
mkdir -p $OUT
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*" >> "$LOG"; }
log "RUNNER_START $NAME $URL"
READY=""
for i in $(seq 1 480); do
  sleep 60
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
done
[ -n "$READY" ] || { log "FAIL not-ready-after-poll-window"; exit 1; }

TOK=$B/glm53-tokenizer
$V/lm_eval run -M local-completions \
  -a "model=bench-$NAME,base_url=$URL/v1/completions,tokenizer=$TOK,api_key=local,num_concurrent=4,max_retries=3,timeout=600" \
  -t mmlu --limit 2 --seed 1234 --output_path $OUT/validate-$NAME \
  > $OUT/validate-$NAME.log 2>&1
RC=$?
log "VALIDATE rc=$RC tail: $(tail -c 300 $OUT/validate-$NAME.log | tr '\n' ' ')"
grep -qiE "gated dataset|GatedRepoError|Permission denied|Status code: 40[13]" $OUT/validate-$NAME.log && { log "BLOCKED dataset-access"; exit 2; }
grep -qE '\|acc' $OUT/validate-$NAME.log || { log "VALIDATE_UNCLEAR not launching"; exit 3; }

log "VALIDATE_OK launching full suite"
echo "$(utc) start $NAME" >> $OUT/status.txt
$V/lm_eval run -M local-completions \
  -a "model=bench-$NAME,base_url=$URL/v1/completions,tokenizer=$TOK,api_key=local,num_concurrent=16,max_retries=5,timeout=1800" \
  -t mmlu \
  --seed 1234 --log_samples --output_path $OUT/$NAME \
  > $OUT/$NAME.log 2>&1
echo "$(utc) exit=$? $NAME" >> $OUT/status.txt
log "RUNNER_DONE exit-recorded"
