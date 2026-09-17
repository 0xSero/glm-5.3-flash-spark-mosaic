# Offline reconstruction versus serving kernels

`runtime_parity.py` is prepared tooling; it has not measured serving parity. Run it
only after the full quality capture/comparison exits successfully, with the parent
operator reserving an idle quality GPU. This is an eight-position screen (or32),
not quality acceptance, throughput measurement, or proof at262k context.

Preparation requires the completed `quality-report.json`, bound normalized rows,
original frozen token fixture, exact artifact config/index/manifest, unchanged
BF16 head shard, and a frozen successful capture-container `docker inspect` JSON.
It checks that no GPU compute PID is present before CUDA initialization. This
check does not replace the operator's exclusive reservation.

```bash
python3 acceptance/runtime_parity.py prepare \
  --quality /run/quality-output --artifact /models/candidate \
  --fixture /run/quality-eval --capture-inspect /run/capture-final-inspect.json \
  --rows 8 --output /run/parity-probes.json
```

Rows0,4,8,12,16,20,24,28 are fixed in advance. Hidden position2046 predicts token2047,
so the raw API prefix is exactly2047 input IDs. The BF16 head multiplication uses
all2047 scored positions and4096-token vocabulary chunks, followed by FP32
logsumexp/logaddexp, matching the existing evaluator's GEMM geometry and reduction.
It does not replace this with a single-vector multiplication. Expected top20 IDs,
logprobs, near-tie gap and immutable evidence hashes are saved.

The runtime operator must bind `candidate_manifest_sha256` to that same candidate,
`model` to the real served ID, `logprobs_mode` to verified `raw_logprobs`, and actual
frozen process/log hashes in `evidence_sha256`. Preserve the existing acceptance
receipt's KV dtype, speculative configuration and context settings.

```bash
python3 acceptance/runtime_parity.py api --probes /run/parity-probes.json \
  --runtime-receipt /run/runtime-receipt.json \
  --base-url http://127.0.0.1:18080/v1 --model ACTUAL_SERVED_ID \
  --output /run/parity-measurement.json
```

API phase sends raw `/v1/completions` token-ID lists with special-token insertion
disabled, temperature0, max_tokens1 and raw top20 logprobs. It requires exact
`token_id:N` response keys and token usage. Missing ID support, altered accounting,
or unsupported responses stop the remaining probes and produce an incomplete
report, never a substituted retokenization comparison.

Reports show top1 agreement, top20 intersection and maximum logprob drift over
shared IDs separately. The default explicit numerical tolerance is0.05 natural-log
units; it can be changed with `--logprob-tolerance`. Strict per-probe success also
requires both top1 IDs represented. Near ties (gap<=0.02) are marked without
relaxing the test. FP8 KV and different fused kernel arithmetic can cause drift;
a failed screen needs investigation and a passing screen does not prove general
runtime equivalence. No broad acceptance gate is automatically granted.

CPU verification: four focused tests cover position mapping, terminal-state gate,
exact token-ID handling, missing accounting and tolerance arithmetic. CLI help
and compilation pass. GPU preparation/API phases remain unexecuted.
