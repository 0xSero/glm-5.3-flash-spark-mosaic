#!/bin/bash
# After the a0 full MMLU exits, run the IDENTICAL quick probe (limit 20/subject)
# on the a0 endpoint so both points have an apples-to-apples subset read.
set -u
B=/Users/sero/sessions/glm53-single-spark-release-20260911/benchmarks/lm-eval-bench
while true; do
  grep -qE "^2026-.*exit=[0-9]+ a0$" $B/results/status.txt 2>/dev/null && break
  sleep 120
done
[ -f $B/.quick-a0-launched ] && exit 0
touch $B/.quick-a0-launched
sleep 60
H=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' http://spark-2384.internal:8000/health 2>/dev/null)
[ "$H" = "200" ] || exit 3
nohup $HOME/lmeval-mac-venv/bin/lm_eval run -M local-completions \
  -a "model=bench-a0,base_url=http://spark-2384.internal:8000/v1/completions,tokenizer=$B/glm53-tokenizer,api_key=local,num_concurrent=16,max_retries=5,timeout=1800" \
  -t mmlu --limit 20 --seed 1234 --output_path $B/results/a0-quick \
  > $B/results/a0-quick.log 2>&1 &
echo "$(date -u +%FT%TZ) a0 quick probe launched post-full-suite" >> $B/results/status.txt
