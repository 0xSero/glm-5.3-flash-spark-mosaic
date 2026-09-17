#!/bin/bash
# Intermittent progress dashboard for the a0-vs-r216 benchmark campaign — 3 TRACKS:
#   Track 1 MMLU           a0 on spark-2384, r216 on spark-557f (suite-exclusive per endpoint once GPQA CoT drains)
#   Track 2 GPQA Diamond   MC done both points; CoT finishing on the same endpoints
#   Track 3 Terminal-Bench 2.1  omarchy x86_64 harness; legs launch per-point on suite-exclusive endpoints
# Every 600s: re-measure, compute rates/ETAs from deltas (results/.progress-state), render PROGRESS.md,
# append PROGRESS-HISTORY.jsonl. Read-only over the network (health curls + ssh probes).
set -u
B="/Users/sero/sessions/glm53-single-spark-release-20260911/benchmarks/lm-eval-bench"
STATE="$B/results/.progress-state"
HIST="$B/PROGRESS-HISTORY.jsonl"
DASH="$B/PROGRESS.md"
PIDFILE="$B/.progress-refresh.pid"
INTERVAL=600

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid $(cat "$PIDFILE")"; exit 0
fi
echo $$ > "$PIDFILE"

num(){ grep -o "| [0-9]*/$2" "$1" 2>/dev/null | tail -1 | awk '{n=$2; sub(/\/.*/,"",n); print n}'; }

hm(){ # key total cur pcur pts now -> R_LAST R_CUM ETA
  R_LAST=""; R_CUM=""; ETA=""
  local key=$1 total=$2 c=$3 pc=$4 pts=$5 now=$6 delta rate
  delta=$((c - pc))
  if [ "$now" -gt "$pts" ] && [ "$delta" -gt 0 ]; then
    rate=$((delta * 60 / (now - pts)))
    [ "$rate" -gt 0 ] && R_LAST="$rate/min"
  fi
  delta=$((c - _first_c))
  if [ "$now" -gt "$firstts" ] && [ "$delta" -gt 0 ]; then
    rate=$((delta * 60 / (now - firstts)))
    [ "$rate" -gt 0 ] && R_CUM="$rate/min"
  fi
  local rest=$((total - c)); [ "$rest" -lt 0 ] && rest=0
  if [ -n "$R_LAST" ]; then
    rate=${R_LAST%/min}
    if [ "$rate" -gt 0 ]; then
      local mins=$((rest / rate))
      if [ "$mins" -ge 60 ]; then ETA="$((mins / 60))h$((mins % 60))m"; else ETA="${mins}m"; fi
    fi
  fi
}

row(){ # label log total key
  local label=$1 log=$2 total=$3 key=$4 c pc pts now
  now=$(date -u +%s)
  c=$(num "$log" "$total"); [ -z "$c" ] && c=0
  pc=$c; pts=$now; firstts=$now; _first_c=0
  local line
  line=$(grep "^$key|" "$STATE" 2>/dev/null | tail -1)
  if [ -n "$line" ]; then
    firstts=$(echo "$line" | cut -d'|' -f3)
    _first_c=$(echo "$line" | cut -d'|' -f2)
    pc=$(echo "$line" | cut -d'|' -f4)
    pts=$(echo "$line" | cut -d'|' -f5)
  fi
  hm "$key" "$total" "$c" "$pc" "$pts" "$now"
  local pct=$((c * 100 / total))
  printf '| %s | %s/%s | %s%% | %s | %s | %s |\n' "$label" "$c" "$total" "$pct" "${R_LAST:-—}" "${R_CUM:-—}" "${ETA:-—}"
  echo "$key|$c|$firstts|$c|$now" >> "$STATE.rw"
  HISTCOUNTS="$HISTCOUNTS \"$key\":$c,"
  HISTRATES="$HISTRATES \"$key\":\"${R_LAST:-0}\","
}


err_lines(){ grep -iaE "traceback|exception|error|failed|refused|oom" "$1" 2>/dev/null | grep -av "Initializing" | tr "\r" "\n" | tail -3 | cut -c1-180; }

remote_traces(){
  echo "[tbench omarchy console]"
  perl -e 'alarm 25; exec @ARGV' ssh -o ConnectTimeout=8 -o BatchMode=yes omarchy 'for f in $(ls -t /home/sero/terminal-bench/runs/full/*.console.log /home/sero/terminal-bench/runs/full/*//harbor-console.log 2>/dev/null | head -2); do echo "--- $f"; tail -3 "$f" 2>/dev/null; done; for f in $(ls -t /home/sero/terminal-bench/runs/full/*/jobs/*/result.json 2>/dev/null | head -2); do python3 -c "import json;d=json.load(open("$f"));s=d.get("stats",{});print("stats:","done",s.get("n_completed_trials"),"err",s.get("n_errored_trials"),"running",s.get("n_running_trials"))" 2>/dev/null; done' 2>/dev/null
  echo "[r216-tbench server 2822]"
  perl -e 'alarm 25; exec @ARGV' ssh -o ConnectTimeout=8 -o BatchMode=yes spark-2822 'tail -4 /tmp/r216-tbench-server.log 2>/dev/null; grep -iE "error|traceback|oom|exceeds" /tmp/r216-tbench-server.log 2>/dev/null | tail -3; curl -sS -m 5 -o /dev/null -w "2822-health %{http_code}\n" http://127.0.0.1:8000/health 2>/dev/null' 2>/dev/null
}

tbench_state(){
  perl -e 'alarm 25; exec @ARGV' ssh -o ConnectTimeout=8 -o BatchMode=yes omarchy 'python3 - <<"PY"
import json,glob
for pat in ("/home/sero/terminal-bench/runs/full/*/jobs/*/result.json","/home/sero/terminal-bench/runs/smoke/*/jobs/*/result.json"):
    for p in sorted(glob.glob(pat)):
        label=p.split("/runs/")[1].split("/jobs/")[0]
        try: d=json.load(open(p))
        except Exception: print(label,"READFAIL"); continue
        s=d.get("stats",{})
        rew={}
        for ev,evd in (s.get("evals",{}) or {}).items():
            for k,v in (evd.get("reward_stats",{}) or {}).items(): rew[k]=len(v)
        mean=None
        for ev,evd in (s.get("evals",{}) or {}).items():
            for m in (evd.get("metrics",[]) or []):
                if isinstance(m,dict) and "mean" in m: mean=m["mean"]
        print(label,"done=%s err=%s running=%s finished=%s"%(s.get("n_completed_trials",0),s.get("n_errored_trials",0),s.get("n_running_trials",0),d.get("finished_at") is not None),"mean_reward:",mean,"rewards:",json.dumps(rew))
PY' 2>/dev/null
}

while :; do
  HISTCOUNTS=""; HISTRATES=""
  : > "$STATE.rw"
  a0h=$(curl -sS -m 6 -o /dev/null -w '%{http_code}' http://spark-2384.internal:8000/health 2>/dev/null)
  r0h=$(curl -sS -m 6 -o /dev/null -w '%{http_code}' http://spark-557f.internal:8000/health 2>/dev/null)
  TB=$(tbench_state)

  {
    echo "# Benchmark campaign progress — a0 vs r216 (3 tracks)"
    echo
    echo "Updated: $(date -u +%FT%TZ). Refreshed every 10 min by \`progress-refresh.sh\` (pid $(cat "$PIDFILE")); rates/ETAs from deltas between refreshes. Receipts under \`benchmarks/lm-eval-bench/\`."
    echo

    echo "## Results so far (a0 = unpruned 2.05bpw reference, r216 = pruned 3.05bpw keep-216)"
    echo
    echo "| Benchmark | Point | Score |"
    echo "|---|---|---|"
    python3 "$B/render-results-table.py" 2>/dev/null | grep -v "^<!--" | grep -v "^| Benchmark" | grep -v "^|---"
    echo
    echo "CoT note: strict = \\boxed{} extraction (near-zero in raw-completions mode, no chat template); flexible = last-number extraction; MC loglikelihood is the clean signal. tbench rows appear when legs land."
    echo
    echo "## Track 1 — MMLU (a0 on spark-2384, r216 on spark-557f; suite-exclusive)"
    echo
    echo "| Leg | Requests | % | Rate (last 10m) | Since start | ETA |"
    echo "|---|---|---|---|---|---|"
    row "a0 @2384" "$B/results/a0.log" 56168 mmlu_a0
    row "r216 @557f" "$B/results/r216.log" 56168 mmlu_r216
    echo
    echo "## Track 2 — GPQA Diamond"
    echo
    echo "| Leg | Requests | % | Rate (last 10m) | Since start | ETA |"
    echo "|---|---|---|---|---|---|"
    row "MC a0 @2384" "$B/results/a0-gpqa.log" 792 gpqa_mc_a0
    row "MC r216 @557f" "$B/results/r216-gpqa.log" 792 gpqa_mc_r216
    row "CoT a0 @2384" "$B/results/a0-gpqa.log" 198 gpqa_cot_a0
    row "CoT r216 @557f" "$B/results/r216-gpqa.log" 198 gpqa_cot_r216
    echo
    echo "## Track 3 — Terminal-Bench 2.1 (omarchy x86_64 harness; legs launch on suite-exclusive endpoints)"
    echo
    echo '```'
    echo "${TB:-omarchy unreachable}"
    echo '```'
    echo
    echo "- Smoke (a0, 1 task): plumbing verified (container+verifier+model wiring, reward computed); trial AgentTimeoutError under suite contention at multiplier 1 — preserved in harbor result.json. Not score-bearing."
    echo "- Full 89-task legs: timeout multiplier 100; a0 leg auto-launches on 2384 after a0-MMLU exits (tbench-auto-a0.sh); r216 leg targeted at spark-2822 dedicated engine after the 2822 purge + staging."
    echo

    echo "## Live traces (rolling, last errors + tails)"
    echo
    echo '```'
    for lg in results/a0.log results/r216.log results/a0-gpqa.log results/r216-gpqa.log; do
      echo "[${lg}]"; err_lines "$B/$lg"
    done
    remote_traces
    echo '```'
    echo
    echo "## Endpoints"
    echo
    echo "| Point | Health (/health) |"
    echo "|---|---|"
    echo "| a0   glm53-a0-bench    spark-2384.internal:8000  (spark-2384) | $a0h |"
    echo "| r216  glm53-r216-bench spark-557f.internal:8000 (spark-557f) | $r0h |"
    echo
    echo "## Status / receipts"
    echo
    echo '```'
    tail -8 "$B/results/status.txt" 2>/dev/null
    echo '```'
    echo
    echo "- Prereg: PREREG.json + PREREG-ADDENDUM.json (3 corrections); runners: run-point.sh, run-gpqa.sh"
    echo "- Logs: results/{a0,r216}.log, results/{a0,r216}-gpqa.log, validate-*.log; history: PROGRESS-HISTORY.jsonl"
    echo "- tbench prereg: /Users/sero/terminal-bench/TBENCH-PREREG.json; harness: /Users/sero/terminal-bench/framework/"
    echo "- 2822 purge receipt: work/glm53-single-spark-release-20260911/receipts/DELETION-20260914-2822.log (on 2822)"
    echo
    echo "## What remains"
    echo
    echo "1. Track 2 completes (CoT ~30 min). Track 1 MMLU completes (~6-7h at isolated rates)."
    echo "2. tbench r216 leg on dedicated spark-2822 engine (staged after purge); a0 leg auto-launches on 2384 post-MMLU."
    echo "3. Assemble a0-vs-r216 comparison with receipts; stop campaign lock refresher; resume ranking queue (E216 rebuild, healing, G5 loop); restore panel-tune serving."
    echo
    echo "Constraints: one GPU job per node; never kill processes; preserve failures; UTC; every number traceable to a receipt."
  } > "$DASH"

  mv "$STATE.rw" "$STATE"
  printf '{"ts":"%s","health":{"a0":%s,"r216":%s},%s "rates":{%s}}\n' \
    "$(date -u +%FT%TZ)" "${a0h:-0}" "${r0h:-0}" "${HISTCOUNTS%,}" "${HISTRATES%, }" >> "$HIST"
  sleep "$INTERVAL"
done
