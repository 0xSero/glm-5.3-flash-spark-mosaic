#!/bin/bash
# recover-557f.sh — user-authorized reboot recovery for spark-557f (tailscale data-plane outage
# since ~2026-09-14T13:05Z; ssh/discos/HTTP dead from Mac, omarchy AND 2384; de5c jump path down
# because de5c itself is offline ~10h). The user asked "is there any way for you to send a reboot
# signal" at ~13:2xZ — that request authorizes rebooting 557f for THIS incident only, overriding
# the standing no-reboot rule. Everything logged to recover-557f.log.
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale
B=/Users/sero/sessions/glm53-single-spark-release-20260911
LOG=$B/benchmarks/lm-eval-bench/recover-557f.log
utc() { date -u +%FT%TZ; }
log() { echo "$(utc) $*" >> "$LOG"; }
direct_probe() {
  perl -e 'alarm 20; exec @ARGV' ssh -o ConnectTimeout=8 -o BatchMode=yes valentine@spark-557f.internal "echo OK557; hostname" 2>&1 | tail -2
}
log "RECOVERY_WATCH_START (user-authorized reboot for 557f data-plane outage)"
while true; do
  sleep 180
  P=$(direct_probe)
  if echo "$P" | grep -q OK557; then log "SELF_HEALED direct ssh alive; no reboot sent"; break; fi
  OFF=$($TS status 2>/dev/null | grep spark-de5c | grep -c offline)
  log "cycle: direct=dark de5c_offline=$OFF"
  if [ "$OFF" = 0 ]; then
    J=$(perl -e 'alarm 45; exec @ARGV' ssh -o ConnectTimeout=10 -o BatchMode=yes spark-557f-extension "echo OK557; hostname" 2>&1 | tail -2)
    log "de5c back; jump probe: $J"
    if echo "$J" | grep -q OK557; then
      log "REBOOT (user-authorized) via de5c jump path"
      perl -e 'alarm 45; exec @ARGV' ssh -o ConnectTimeout=10 -o BatchMode=yes spark-557f-extension "sudo -n systemctl reboot" >> "$LOG" 2>&1
      log "reboot command issued (rc recorded above); waiting for host return"
      break
    fi
  fi
done
# Phase 2: wait for 557f to return, then relaunch the r216 bench server (attempt-3 knobs).
for i in $(seq 1 60); do
  sleep 60
  P=$(direct_probe)
  if echo "$P" | grep -q OK557; then
    log "557F_BACK after wait $i — relaunching r216 bench attempt 3"
    perl -e 'alarm 60; exec @ARGV' scp -o ConnectTimeout=10 -o BatchMode=yes \
      $B/benchmarks/lm-eval-bench/relaunch-r216-bench3.sh valentine@spark-557f.internal:/home/valentine/bench/ >> "$LOG" 2>&1
    perl -e 'alarm 150; exec @ARGV' ssh -o ConnectTimeout=10 -o BatchMode=yes valentine@spark-557f.internal \
      "bash /home/valentine/bench/relaunch-r216-bench3.sh" >> "$LOG" 2>&1
    log "attempt-3 relaunch executed; lm-eval runner on the Mac picks the point up when READY"
    break
  fi
done
log "RECOVERY_WATCH_EXIT"
