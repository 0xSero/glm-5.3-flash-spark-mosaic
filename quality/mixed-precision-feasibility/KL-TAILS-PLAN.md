# Supplemental full-vocabulary KL tails

No GPU run is authorized by this plan alone. Preserve every frozen evaluator and existing quality report.

Use the retained32 normalized BF16 teacher rows and the corresponding32 normalized candidate rows already sealed for originalK2, originalQ3, keep256, or keep192. Verify terminal capture identity, candidate/fixture/teacher manifest hashes, native-head source hash and every normalized tensor's SHA/shape/dtype before GPU initialization. This avoids replaying45 transformer layers.

For each row, use the original all2047-position BF16 F.linear geometry with native BF16 head vocabulary chunks4096, then FP32 logits. First pass computes teacher/candidate log-normalizers with the original sequential logsumexp/logaddexp reduction. Second pass recomputes the same chunk logits and accumulates full-vocabulary token KL: sum_v p(v) × [log p(v) − log q(v)]. Keep token-wise sums in FP64 across vocabulary chunks to reduce summation drift; record this supplemental accumulation choice explicitly. There are only65,504 final token KL scalars, so save them in a separate small artifact.

Report mean, median, p95, p99, p99.9, maximum, raw minimum and nonfinite/negative counts. Declare the quantile method explicitly (linear interpolation). Preserve raw tiny negative rounding values; never silently clamp or discard difficult positions. Cross-check summed per-row KL, teacher/candidate NLL and top1 against the existing report with a declared numerical tolerance for the changed reduction order. A discrepancy outside tolerance fails the supplement rather than replacing the original result.

Bind supplemental output to the exact candidate normalized manifest and original quality report SHA. Add aggregate tails to the study table; do not use this held-out panel's high-KL tokens or layer effects to choose precision allocation. A separate calibration sensitivity experiment must select upgrades. API top20 logprobs alone cannot establish full-vocabulary KL tails.

Peak memory can remain bounded to two normalized hidden rows, one head chunk, two chunk-logit matrices and small reduction vectors. Two passes trade additional head GEMMs for avoiding full-vocabulary resident logits. This is a head-only GPU task on an explicitly idle reserved node, with no live serving changes.
