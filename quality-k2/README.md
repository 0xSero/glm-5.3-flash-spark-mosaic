# Uniform K2 GLM-5.3-Flash full-trunk quality control

This independent directory is a frozen copy of the Q3 quality adapter and its
unchanged comparison dependencies. It does not import or modify active `quality/`
files. No K2 GPU capture or server mutation has been launched by this work.

The candidate is the unpruned public checkpoint
`0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw`, pinned at
`35b4b580debe2a1510a2eb042d9d6d4cd7fb3a6b`:
133 weight files,111,352,026,456bytes,42 routed layers ×288 experts.

## Identity and native protected weights

`../source-inventory-k2/` contains the pinned HF config, index, EXL3 manifest,
quantization metadata and LFS file-hash inventory. Inventory SHA:
`36b79ee52ace17008eb917588666e3ae7f6ce5f419470df019da8a27576d594b`.

All49 retained-shard SHA/byte entries match the original Q3 protected source,
including the unchanged LM head and all native MTP weights. The proof is
`protected-q3-parity.json`. These2482 tensors total33,835,039,608bytes.
The source manifest retains native precision; this quality run does not prune
or re-encode anything.

Fresh read-only SSH inspection found the following existing557f directory:
`/home/valentine/flash-experimental-staging-20260906/model`.
Its config, index, quantization config and EXL3 manifest are **byte-identical** to
the pinned HF files. No local runtime metadata override was observed. This does
not establish full local weight hashes; the validator performs that gate before
capture. `observed-557f-metadata.json` records the limited live audit.

The validator fails if any future local metadata differs. It never changes a
running server's files. If runtime overrides appear, create a separate canonical
metadata view and preserve the serving view. Capture always uses the pinned
metadata bundle's canonical configuration.

## Read-only validation

Pure standard-library metadata/header/size inspection (not a full payload seal):

```bash
python3 validate_source.py \
  --source /models/k2 \
  --inventory /release/source-inventory-k2/filehash-inventory.json \
  --metadata-only --output /release/results/k2-metadata.json
```

Omit `--metadata-only` to hash all133 source files. The validator binds the
original metadata, exact index/header closure,145152 actual K2 trellis tensors,
all four archive-rank fragments, and identical Q3 native protected shards.
A metadata-only result never claims full payload verification.

## Full target-trunk measurement

Use an otherwise idle Spark with the established observer image, not the557f
server while another agent owns it. Preserve Pop inference. Example container
paths follow; the coordinator owns actual mounts and GPU launch.

```bash
export PYTHONPATH=/release/quality-k2/deps:/workspace/src:$PYTHONPATH
export GLM53_EXL3_GPU_IDS=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 /release/quality-k2/evaluate_k2_quality.py \
  --phase all \
  --source-inventory /release/source-inventory-k2/filehash-inventory.json \
  --artifact /models/k2 \
  --fixture /release/quality-eval \
  --teacher /release/bf16-normalized \
  --output /release/results/original-k2-control
```

Each invocation first validates all source payload hashes. `--phase preflight`
validates source, imports, teacher and token fixture without GPU capture.
`capture` and `compare` can run separately; `all` avoids an extra validation pass.
Outputs are bound to source/teacher/token/dependency identity and resume only at
verified layer boundaries. Use a distinct output directory from every Q3 run.

The full45-layer target trunk is evaluated on the exact same32×2048 WikiText2 test
fixture and65,504 next-token positions as the Q3 control. Each K2 projection is
reconstructed by the **same real LinearEXL3.get_weight_tensor()** path and copied
into the same BF16 native layer geometry. The original decoder attention state
and final normalization are preserved. The head uses the unchanged BF16 F.linear
then FP32 logits, vocabulary chunks4096. This is not an expert error proxy.

Teacher manifest SHA:
`a1604015f76d6a6b6f032ffb7e3359a770e8dd2564ca89e3b00eb8bcb4d93314`.
Fixture manifest SHA:
`46255621cd2da8e05546edc3498cfbee518da092af2f44ae243f79da97c0f00b`.
Frozen comparison implementation SHA:
`0c02949a77d28fe7c45786e964a1241cbba8d33b076335ca6a08a4fcac28729a`.

`quality-report.json` reports measured KL(BF16 teacher || K2), perplexity and top1
agreement. It is not automatically marked accepted. Compare it with the fresh Q3
and pruned-Q3 controls on this same fixture. A difference between two
teacher-relative KL numbers is not KL(Q3 || K2).

MTP remains preserved but is excluded from this target-trunk comparison, matching
the teacher. Native MTP execution, vision/video,262144-context, CUDA graphs and
sustained serving speed remain separate runtime acceptance gates.

## Validation performed

Five CPU tests pass: pinned metadata and protected parity, runtime metadata drift
rejection without mutation, K2/K3 header discrimination, payload-vs-size-only
verification and CLI import independence. `smoke_environment.py` also passed
inside the exact observer image
`sha256:54acf16064726697ab079dfbd297c1c1c260cb28867f8287f5ae2ec3bbb0a382`,
with no GPU exposed, no network,2GiB memory limit and read-only observer source.
It imported the real reconstruction class/compiled extension, constructed dense,
DSA and linear-attention layers on `meta`, and verified CUDA was uninitialized.
This is CPU import/geometry evidence, not a GPU reconstruction or inference test.

```bash
python3 -m unittest discover -s quality-k2 -p 'test_*.py' -v
```
