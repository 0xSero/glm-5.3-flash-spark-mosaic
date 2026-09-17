# Full-model quality for physical GLM-5.3-Flash REAP candidates

This adapter reuses the unmodified `evaluate_low_bpw_quality.py` comparison. It advances all 45 target decoder layers, propagates the actual hidden streams and shared DSA attention token indices, reconstructs candidate EXL3 experts to BF16, applies final RMSNorm, and evaluates all **65,504 next-token positions** in the sealed 32 × 2,048 WikiText-2 fixture. This is not an expert-only error proxy.

The teacher and candidate head use exactly the baseline arithmetic: BF16 `torch.nn.functional.linear`, then FP32 logits, vocabulary chunks of 4,096. KL is BF16-teacher-to-candidate. Cross-entropy, perplexity and top-1 agreement cover the same positions. No historical KLD threshold is silently adopted as release acceptance.

Native MTP is preserved in the candidate but excluded from this target-trunk quality measurement, matching the teacher. Vision, MTP, long-context runtime correctness, and serving-kernel parity need separate acceptance tests.

## Contents and dependencies

`deps/` contains unchanged copies of the existing evaluator and its two helper modules. The evaluator hash is pinned by the adapter. It also needs:

- CUDA-capable PyTorch and safetensors;
- the same Transformers GLM5-Next implementation used for the teacher, including `Glm5NextTextDecoderLayer`;
- the custom `exllamav3.modules.quant.LinearEXL3` implementation and its compiled extension;
- the observation checkout's `src` on `PYTHONPATH` (the unchanged layer loader imports `reap.glm53_virtual_abliteration`).

Use the established GLM Spark conversion/observer environment after verifying these imports. Do not substitute a generic current Transformers build. Record installed package/source identities with the actual run. Each candidate runs independently on one GPU; distribute candidates across idle Sparks instead of changing the comparison geometry.

## Candidate contract

The adapter accepts the physical candidate builder's `EXL3_MANIFEST.json` (`glm53-physical-reap-exl3-v1`, `STRUCTURAL_PASS`) and bound `BUILD_STATUS.json`. It verifies every metadata and payload file hash once per invocation before capture or comparison. It requires all four packed rank slices and four EXL3 fields for every retained expert, renumbered contiguously according to the builder's map. Routers resolve through the candidate's actual index. MTP remains native with 288 experts.

`prev_topk_indices` in this model means shared DSA **attention token indices**, not MoE expert IDs. Independent per-layer expert renumbering does not require remapping this state.

## Commands inside the established Spark environment

Mount or stage the whole `quality/` directory and the existing teacher/fixture directories. Example paths below are container paths, not new host assumptions.

```bash
export PYTHONPATH=/work/quality/deps:/workspace/src:$PYTHONPATH
export GLM53_EXL3_GPU_IDS=0

python3 /work/quality/evaluate_pruned_quality.py \
  --phase preflight \
  --artifact /candidate \
  --fixture /work/quality-eval \
  --teacher /work/bf16-normalized \
  --output /work/results/massmax-keep192

python3 /work/quality/evaluate_pruned_quality.py \
  --phase capture \
  --artifact /candidate \
  --fixture /work/quality-eval \
  --teacher /work/bf16-normalized \
  --output /work/results/massmax-keep192

python3 /work/quality/evaluate_pruned_quality.py \
  --phase compare \
  --artifact /candidate \
  --fixture /work/quality-eval \
  --teacher /work/bf16-normalized \
  --output /work/results/massmax-keep192
```

`--phase all` combines capture and comparison. Capture resumes at the latest verified layer boundary. Use a distinct output directory for every candidate. The identity file refuses reuse after candidate, fixture, teacher, or evaluator changes. Retain approximately 10 GiB of additional disk space for the alternating hidden-state slots and normalized outputs; actual model memory and execution acceptance must be measured.

The report is `quality-report.json`, marked `QUALITY_MEASURED`, not automatically accepted. The adapter has CPU contract tests but has **not yet run a GPU capture**.

## Matched original Q3 control

Use `--source-inventory` explicitly for the full original **0xSero/GLM-5.3-Flash-EXL3-3.0bpw** checkpoint at revision `2a30ad09c15f779a44fa62c216f5dbe5fb0c9223`. This mode requires all 288 experts, the hard-pinned inventory (`a5fb57d1c198d1dcbd58748ed22c1f1cb86d2cee8d04bb5623a7ef1b099350e4`), all **130 weight files / 149,402,871,912 bytes**, and all five original metadata hashes. No missing data or integrity check is bypassed. An original source cannot enter pruned mode, and a pruned source cannot claim the original-control identity.

```bash
python3 /release/quality/evaluate_pruned_quality.py \
  --phase all \
  --source-inventory /release/source-inventory/filehash-inventory.json \
  --artifact /candidate \
  --fixture /release/quality-eval \
  --teacher /release/bf16-normalized \
  --output /release/results/original-q3-control
```

The original and pruned runs use the same capture/reconstruction/head code. Compare their BF16-teacher KL, cross-entropy and top-1 agreement on identical positions. Keep both reports. A difference between their teacher-relative KL values is not itself `KL(original || pruned)`; measure that separately if needed.

The CPU-inspected established image is recorded in `de5c-runtime.json`. Stage the latest whole `quality/` tree before launching. The root coordinator owns actual GPU launches and confirms source transfer completion.

### Established image import path

The image lacks the standalone `flash_attn` Python package, and ExLlama's top-level initializer eagerly imports its own attention implementation. That attention path is not used by this evaluator. The adapter now calls the **existing observer's** quantization-only namespace loader before importing the baseline. This loads the real `LinearEXL3`, reconstruction functions, and compiled extension without importing ExLlama serving attention. It does not substitute an attention implementation or change Transformers.

`smoke_environment.py` passed in the exact image without GPU access: real class and extension imports, unchanged baseline import, and meta-device construction of dense, DSA-MoE and linear-attention-MoE decoder layers. CUDA stayed uninitialized and neither `flash_attn` nor `exllamav3.modules.attn` was imported. GPU execution remains a distinct test.

```bash
PYTHONPATH=/release/quality:/release/quality/deps:/workspace/src \
python3 /release/quality/smoke_environment.py --config /release/source-inventory
```

```bash
python3 -m unittest discover -s quality -p 'test_*.py' -v
```
