#!/bin/bash
# E216 orchestrator (driver run 5, queue item 4 tail): build -> census -> served receipt -> serve
# (KV gate + one relaunch) -> smoke -> G4 on 2384. Logs to e216/orchestrator.log. Refreshes
# loop-driver/DRIVER-LOCK.json on every poll so concurrent driver runs stay skipped. Every remote
# call is wrapped in a hard timeout. No process is killed other than via the run-serve attempt
# scripts' own documented docker stop of the glm53-e216 container this orchestrator launched.
set -u
ROOT=/Users/sero/sessions/glm53-single-spark-release-20260911
LOG=$ROOT/e216/orchestrator.log
LOCK=$ROOT/loop-driver/DRIVER-LOCK.json
S557='ssh -o ProxyJump=none -o Hostname=spark-557f.internal -o ConnectTimeout=8 -o BatchMode=yes spark-557f-extension'
S238='ssh -o ConnectTimeout=8 -o BatchMode=yes spark-2384'
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(utc) $*" >> "$LOG"; }
refresh_lock() { printf '{"started_utc": "%s", "run": "driver", "note": "queue item 4: E216 REAP-native build+eval in flight (orchestrator refresh)"}' "$(utc)" > "$LOCK"; }
rx() { perl -e 'alarm 25; exec @ARGV' $S557 "$@"; }
r23() { perl -e 'alarm 25; exec @ARGV' $S238 "$@"; }
fail() { log "ORCHESTRATOR-FAILED stage=$1"; echo "ORCHESTRATOR-FAILED:$1 $(utc)" >> "$LOG"; exit 1; }

log "ORCHESTRATOR-START (E216 build+eval)"
refresh_lock

# ---- stage: build (already launched manually; wait for BUILD_DONE) ----
log "stage=BUILD waiting for BUILD_DONE"
for i in $(seq 1 40); do
  sleep 90
  refresh_lock
  n=$(rx "grep -c BUILD_DONE /home/valentine/e216/build.out 2>/dev/null" 2>/dev/null | tail -1)
  if [ "${n:-0}" -ge 1 ] 2>/dev/null; then log "stage=BUILD done (poll $i)"; break; fi
  [ "$i" -eq 40 ] && fail "build-timeout"
  log "stage=BUILD poll $i not done"
done
packfail=$(rx "grep -c 'pack exit=0' /home/valentine/e216/build.out" 2>/dev/null | tail -1)
[ "${packfail:-0}" -ge 1 ] 2>/dev/null || fail "pack-nonzero-exit"
verif=$(rx "grep -c 'verify exit=0' /home/valentine/e216/build.out" 2>/dev/null | tail -1)
[ "${verif:-0}" -ge 1 ] 2>/dev/null || fail "verify-nonzero-exit"
log "stage=BUILD pack+verify exit=0 confirmed"

# ---- stage: census ----
rx "cd /home/valentine/e216 && nohup bash run-census.sh > census.out 2>&1 &" >/dev/null 2>&1
log "stage=CENSUS launched"
for i in $(seq 1 30); do
  sleep 60
  refresh_lock
  n=$(rx "grep -c CENSUS_DONE /home/valentine/e216/census.out 2>/dev/null" 2>/dev/null | tail -1)
  if [ "${n:-0}" -ge 1 ] 2>/dev/null; then log "stage=CENSUS done (poll $i)"; break; fi
  [ "$i" -eq 30 ] && fail "census-timeout"
done
cexit=$(rx "grep -o 'exit=0' /home/valentine/e216/census.out | wc -l" 2>/dev/null | tail -1)
[ "${cexit:-0}" -ge 2 ] 2>/dev/null || fail "census-nonzero-exit"
log "stage=CENSUS both exits 0 confirmed"

# ---- stage: served receipt (pre-stage before ANY launch) ----
out=$(rx "python3 /home/valentine/e216/make-served-census-receipt-e216.py" 2>&1)
log "stage=RECEIPT: $out"
echo "$out" | grep -q '"verdict_contract_ok": true' || fail "receipt-generation"
log "stage=RECEIPT ok (pre-staged before first launch)"

# ---- stage: serve (attempt 1) ----
rx "cd /home/valentine/e216 && nohup bash run-serve-e216.sh > serve.out 2>&1 &" >/dev/null 2>&1
log "stage=SERVE attempt1 launched"
KV=""
for try in 1 2; do
  for i in $(seq 1 40); do
    sleep 30
    refresh_lock
    kv=$(rx "docker logs glm53-e216 2>&1 | grep -m1 -o 'max_total_num_tokens=[0-9]*'" 2>/dev/null | tail -1 | cut -d= -f2)
    if [ -n "${kv:-}" ] 2>/dev/null; then
      KV=$kv
      if [ "$kv" -ge 262144 ] 2>/dev/null; then log "stage=SERVE attempt$try KV=$kv OK"; break 2; fi
      log "stage=SERVE attempt$try KV=$kv < 262144 (profiling variance)"
      break
    fi
    log "stage=SERVE attempt$try poll $i no KV yet"
    [ "$i" -eq 40 ] && fail "serve-kv-timeout-attempt$try"
  done
  [ "$try" -eq 1 ] || fail "serve-kv-short-after-relaunch"
  rx "cd /home/valentine/e216 && nohup bash run-serve-e216-attempt2.sh > serve-attempt2.out 2>&1 &" >/dev/null 2>&1
  log "stage=SERVE relaunch (attempt 2) issued"
done
# wait for ready + smoke (retry up to 4x, T216 warmup-transient pattern)
SMOKE=""
for i in $(seq 1 4); do
  sleep 45
  refresh_lock
  SMOKE=$(rx "curl -sS -m 120 -X POST http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"/model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"temperature\":0,\"max_tokens\":512}'" 2>&1)
  echo "$SMOKE" | grep -q '"finish_reason"' && { log "stage=SMOKE ok (try $i): $(echo "$SMOKE" | head -c 200)"; break; }
  log "stage=SMOKE try $i no finish_reason; response head: $(echo "$SMOKE" | head -c 120)"
  [ "$i" -eq 4 ] && fail "smoke-no-response"
done
log "stage=SERVE_OK KV=$KV"

# ---- stage: G4 on 2384 ----
r23 "test -f /home/sero/w2port/g4/run-e216-g4.sh" >/dev/null 2>&1 || fail "g4-script-missing"
r23 "cd /home/sero/w2port/g4 && nohup bash run-e216-g4.sh > g4-e216.out 2>&1 &" >/dev/null 2>&1
log "stage=G4 launched on 2384"
for i in $(seq 1 36); do
  sleep 300
  refresh_lock
  n=$(r23 "grep -c E216_G4_DONE /home/sero/w2port/g4/g4-e216.out 2>/dev/null" 2>/dev/null | tail -1)
  if [ "${n:-0}" -ge 1 ] 2>/dev/null; then log "stage=G4 done (poll $i)"; break; fi
  [ "$i" -eq 36 ] && fail "g4-timeout"
  log "stage=G4 poll $i not done"
done
g4c=$(r23 "grep -o 'candidate exit=0\|compare exit=0\|samples exit=0\|summary exit=0' /home/sero/w2port/g4/g4-e216.out | wc -l" 2>/dev/null | tail -1)
[ "${g4c:-0}" -ge 4 ] 2>/dev/null || fail "g4-phase-nonzero"
log "ORCHESTRATOR-COMPLETE KV=$KV all G4 phases exit=0"
echo "ORCHESTRATOR-COMPLETE $(utc)" >> "$LOG"
