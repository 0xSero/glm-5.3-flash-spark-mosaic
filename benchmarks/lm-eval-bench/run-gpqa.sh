#!/bin/bash
# run-gpqa.sh <name> <url> [poll-min] — GPQA-only suite (gated dataset; launch AFTER access grant).
# Same gates as run-point.sh: /health → completion smoke → 2-doc gpqa_diamond_zeroshot validation
# (this is also what catches the gate being closed again) → full GPQA prereg pair.
set -u
NAME=$1; URL=$2; LIMIT=${3:-480}
V=$HOME/lmeval-mac-venv/bin
B=$HOME/sessions/glm53-single-spark-release-20260911/benchmarks/lm-eval-bench
LOG=$B/run-gpqa-$NAME.log; OUT=$B/results
mkdir -p $OUT
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*" >> "$LOG"; }
log "GPQA_RUNNER_START $NAME $URL"
READY=""
for i in $(seq 1 $LIMIT); do
  sleep 60
  C=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' $URL/health 2>/dev/null)
  if [ "$C" = 200 ]; then
    S=$(curl -sS -m 90 -o /tmp/smoke-gpqa-$NAME.json -w '%{http_code}' \
      -H 'Content-Type: application/json' \
      -d '{"model":"default","prompt":"hello","max_tokens":1,"temperature":0}' $URL/v1/completions 2>/dev/null)
    log "poll $i health=200 smoke=$S"
    if [ "$S" = 200 ] && grep -q '"choices"' /tmp/smoke-gpqa-$NAME.json 2>/dev/null; then
      READY=1; log "READY_OK"; break
    fi
  else
    log "poll $i health=$C"
  fi
done
[ -n "$READY" ] || { log "FAIL not-ready-after-poll-window"; exit 1; }

TOK=$B/glm53-tokenizer
# NOTE 2026-09-14T15:5xZ: upstream Idavidrein/gpqa is gated and 0xSero not granted;
# tasks-gpqa-local points dataset_path at gpqa-local-ds (original-schema mirror
# bdytx5/gpqa_gpqa_diamond, cross-verified 198/198 vs hendrydong/gpqa_diamond).
# Task names, yaml definitions, metrics UNCHANGED from prereg.
$V/lm_eval run -M local-completions \
  -a "model=bench-$NAME,base_url=$URL/v1/completions,tokenizer=$TOK,api_key=local,num_concurrent=4,max_retries=3,timeout=600" \
  --include_path $B/tasks-gpqa-local \
  -t gpqa_diamond_zeroshot --limit 2 --seed 1234 --output_path $OUT/validate-gpqa-$NAME \
  > $OUT/validate-gpqa-$NAME.log 2>&1
RC=$?
log "VALIDATE rc=$RC tail: $(tail -c 300 $OUT/validate-gpqa-$NAME.log | tr '\n' ' ')"
grep -qiE "gated dataset|GatedRepoError|Permission denied|Status code: 40[13]" $OUT/validate-gpqa-$NAME.log && { log "BLOCKED gpqa-access"; exit 2; }
grep -qE '\|acc' $OUT/validate-gpqa-$NAME.log || { log "VALIDATE_UNCLEAR not launching"; exit 3; }

log "VALIDATE_OK launching GPQA suite"
echo "$(utc) start $NAME-gpqa" >> $OUT/status.txt
$V/lm_eval run -M local-completions \
  -a "model=bench-$NAME,base_url=$URL/v1/completions,tokenizer=$TOK,api_key=local,num_concurrent=16,max_retries=5,timeout=1800" \
  --include_path $B/tasks-gpqa-local \
  -t gpqa_diamond_zeroshot gpqa_diamond_cot_zeroshot \
  --seed 1234 --log_samples --output_path $OUT/$NAME-gpqa \
  > $OUT/$NAME-gpqa.log 2>&1
echo "$(utc) exit=$? $NAME-gpqa" >> $OUT/status.txt
log "GPQA_RUNNER_DONE exit-recorded"
