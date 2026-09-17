# PRUNE PROGRAM CLOSED — 2026-09-15T01:4xZ (driver run 19, a0-fallback default executed)

The 25%-prune program (uniform keep-216 of 288 routed experts, EXL3 3.05bpw, SGLang + exl3_plain) is
**CLOSED as a negative result**. Release config = **a0 (q2, 2.05bpw, unpruned, all 288 experts)**.
Trigger: user said "continue" after the fork was presented with a0-fallback as the recommended default
(healing/PLAN.md §6: "Default if the user says nothing: a0-fallback stands"); fork option (b) remains
priced and unlaunched in healing/PLAN.md if ever reopened.

## Why closed (all receipted)

1. Every measured pruned point loses to a0 on every benchmark surface — GPQA MC: a0 43.43±3.53 vs
   R216 40.91±3.50 / E216 37.88±3.46; MMLU-quick: a0 83.42±1.20 vs E216 77.28±1.20 (−6.14pp ~3.6σ);
   GPQA CoT-flex: a0 31.82 vs R216 17.68. Receipts: benchmarks/lm-eval-bench/results/.
2. Both selection criteria tested (route-mass-informed R216; REAP-native E216) fail; the G4
   teacher-agreement panel predicts neither benchmark direction.
3. The prune's only value proposition (3.05bpw fitting 128 GB) buys a WORSE model than the smaller,
   higher-KV-headroom a0. Healing's binding burden of proof (healing/PLAN.md §1) — beat a0 by ≥1σ on
   two suites — would require recovering ~100% of a 0.4-nat gap.

## What is preserved (nothing sealed was deleted; every artifact rebuildable or archived)

| Item | Where | Status |
|---|---|---|
| Sealed base (GLM-5.3-Flash-exl3 3.05bpw, turboderp) | 557f | UNTOUCHED |
| Sealed laguna-reap-saliency-v1 + sealed observations.json | 2822 / Mac | UNTOUCHED |
| E216 artifact (96.58 GB, 31 files, G2-verified) | 557f:/home/valentine/models/glm53-3p05-pruned-e216 | **DELETED 2026-09-16T10:52Z** per user order ("remove the prunes off the sparks"); container glm53-e216 stopped+removed 2026-09-15T01:36:04Z (receipt e216/receipts/e216-pre-stop-receipt.txt); pre-deletion receipt prune-removal-20260916/prune-removal-557f-pre.json |
| S216 + T216 + F216 artifacts (96.58 GB each, G2-verified) | 557f:/home/valentine/models/glm53-3p05-pruned-{s216,t216,f216} | **DELETED 2026-09-16T10:52Z** per user order; receipts prune-removal-20260916/ (pre + listings + post-deletion-receipt.json); F216 artifact on 2822 likewise deleted |
| r216 / u216 / t216 / s216 / e216 / f216 plans + receipts + build tooling | Mac (r216/ u216/ t216/ s216/ e216/ f216/) + 2822:/home/sero/work/f216-build/ | PRESERVED — every deleted artifact is rebuildable (sealed base + frozen plan sha + packer; pack time 80.3 s measured on f216). Whole keep-216 artifact family is now off the fleet; nothing sealed was deleted |
| Plans: u216/s216/t216/r216/e216 + all BUILD-*.md | Mac | COMPLETE incl. BUILD-E216.md (backfilled run 18) |
| Calibration corpora + teacher rows + G4 receipts | 2384:/home/sero/w2port/ + Mac | UNTOUCHED |
| All benchmark receipts (PREREG, results/, partials) | Mac benchmarks/lm-eval-bench/ | COMPLETE for the surfaces run |
| Healing plan (unlaunched) | healing/PLAN.md | WAITING if user reopens fork (b) |

## Release serving (what is running and its exact config)

- Host 2384, container **glm53-a0-bench** (Up since 2026-09-14T14:39:14Z, KV pool 1,057,088 tokens),
  image glm53-flash-sglang-exl3-plain:serve4, endpoint http://spark-2384.internal:8000 — the config that
  produced every a0 benchmark row. Exact flags/env in
  benchmarks/lm-eval-bench/a0-release-serving-inspect.json (docker inspect snapshot, this run):
  entrypoint python3, `-m sglang.launch_server --model-path /model --host 0.0.0.0 --port 8000
  --quantization exl3 --tp-size 1 --ep-size 1 --context-length 262144 --kv-cache-dtype fp8_e4m3
  --attention-backend dsa --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend
  flashinfer_sparse_mla --linear-attn-backend triton --disable-shared-experts-fusion
  --chunked-prefill-size 256 --max-prefill-tokens 256 --max-running-requests 16
  --mem-fraction-static 0.90 --enable-multimodal --chat-template /opt/glm53/chat-template-mm.jinja
  --reasoning-parser glm45 --tool-call-parser glm47 --disable-cuda-graph`, env
  SGLANG_EXL3_MAX_BATCH_TOKENS=256 (+ census/decoder overlay envs).
- Serving slots after closure: 2384 = a0 (release, serving); 557f = IDLE (all pruned artifacts removed
  2026-09-16, disk 89% -> was 99% with 39 GB free; panel-era restore path = the preserved inspects +
  run-serve scripts in e216/; nv-persisted-restore shim still holding the persistenced socket — owner
  should later `sudo systemctl restart nvidia-persistenced` then `docker rm -f nv-persisted-restore`);
  2822 = IDLE (53% disk after f216 removal; ~1.83 TB free).

## Disk removals (2026-09-16, interactive turn)

User order: "remove the prunes off the sparks we can't keep taking up all this space for stuff we don't
use". Removed 5 pruned artifact dirs totalling **482,966,693,370 B (≈483 GB; df freed 482,967,396,352 B)**:
2822 glm53-3p05-pruned-f216; 557f glm53-3p05-pruned-{e216,f216,s216,t216}. Each had a pre-deletion
receipt (path, du -sb bytes, file count, sha256 of every *.json identity file, full per-file listing)
taken before `rm -rf`, and the removal was verified (`gone: true`, rm_rc 0) with df before/after.
Receipts: `prune-removal-20260916/` (pre-deletion JSON + listings + post-deletion-receipt.json).
Preserved: sealed bases, the 2.05bpw quant copy, Q4 observation model, a0, all plans/tooling/receipts,
and 557f's exllamav3 quantizer source (exllamav3-v1.4.9 + src).

## Open items handed to the owner

- ~520 GB root-owned remnants on 2822 need owner sudo (list in DELETION-20260914-2822.log era report).
- de5c powered off since 2026-09-14 ~08:2xZ (incident 2026-09-14T092533Z).
- pop-os (production, observe-only) REBOOTED ~2026-09-15T00:5xZ — observed by watchdog ("up 36
  minutes" at 01:3xZ), cause unestablished, untouched per constraint. Worth the owner's attention.
- Terminal-Bench 2.1 score for a0 remains UNRUN (paused 2026-09-14; smoke = AgentTimeout at
  multipliers 1 and 4 under ~15 tok/s decode; a full leg is impractical on this engine — decision
  with owner if that score is ever needed).
