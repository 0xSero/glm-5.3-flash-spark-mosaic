# Single-Spark GLM study, 2026-09-11

These are experiment records, not acceptance of the final release. The target is
GLM-5.3-Flash on one 128 GB DGX Spark, with 262,144 total tokens per request,
working images/video and the model's MTP algorithm. No power settings were changed.

[Quality measurements](QUALITY.md) compare the same 65,504 next-token positions
against one frozen BF16 teacher. [Baseline speeds](LEGACY-SPEED.md) report
actual prompt lengths, separate total and per-request decode, and server request
prefill. The baseline uses legacy EXL3/FlashInfer, not B12x.

| Experiment | Observed result | Decision |
|---|---|---|
| Original Q3, 288 experts | KL 0.152204; PPL 3.49691 | Quality control; unpruned payload exceeds single-Spark capacity |
| Original K2, 288 experts | KL 0.438986; PPL 4.54687 | Quality control; full target with smaller MTP draft under test |
| Q3 REAP, 176 experts | KL 1.343988; PPL 11.27267 | Rejected for large quality loss |
| K2 REAP, 256 experts | KL 0.687907; PPL 5.82429; legacy text/MTP passed | Lower quality than unpruned K2; pruning no longer needed for measured C1 capacity |
| Full K2 plus BF16 MTP | Weights loaded at 103.86 GiB; subsequent KV budget was −1.13 GiB | Failed KV admission; no ready API |
| K2 keep256 plus BF16 MTP | Two sustained C1 samples measured 12.576 and 13.304 total decode tok/s | Legacy baseline only |
| Full K2 plus FP8 MTP routed experts | 97.11 GiB loaded; 460,208 aggregate KV tokens allocated; 262,144 request limit; fresh exact text and MTP counters passed | Legacy serve pass; long-context, media and final B12x acceptance pending |
| Q3 REAP, 192 experts | Structural pass; quality run in progress | Not accepted |
| Latest pinned B12x runtime | Reached final CUDA build unit; missing cusparse.h header under repair with objects retained | No B12x speed claim yet |

The first baseline request had no separate matched-shape warmup; the cause of its
larger prefill time was not isolated. Its actual
prompt was 1,023 tokens; the second was 1,024. Each generated 2,048 tokens.
Speculative counters on the second request advanced by 1,173 drafted tokens and
874 accepted tokens. This proves MTP execution, not an MTP-on versus MTP-off gain.

Failed loader and build attempts are also retained:

| Failure | Cause established from evidence | Correction or status |
|---|---|---|
| Initial native-MTP loads | Draft inherited target EXL3 quantization; V2 used a distinct loader path | Bind the draft's model and quantization config in both paths; real strict native load subsequently passed |
| Memory fraction 0.94 startup | 114.03 GiB free versus 114.39 GiB reservation | 0.93 passed initial admission; later KV allocation tested separately |
| Thinking-off exact-answer smoke | Template always opens `<think>`; false flag disabled parser separation | Explicit thinking enabled; exact text then passed |
| First benchmark client | Transformers returned a BatchEncoding; its key count was mistaken for token count | Explicit `return_dict=False`; seven prompt-size comparisons with the server passed |
| B12x build, initial attempts | NVRTC linker entry missing, then stale CMake lookup cache | Corrected linker discovery and only the failed cache |
| B12x final CUDA unit | DeepGEMM wrapper could not find cusparse.h under CUDA_HOME | Repairing matching installed header search path; expensive objects retained |
| B12x build, 20 GiB cap | Compiler exceeded the cap and exited 137 | Controlled resume at 64 GiB; observed peak 29.62 GiB at this stage |

Original source files and recoverable experiment outputs were retained. Sealing
32 normalized quality rows proves final-result integrity; early monitoring
records were unavailable for layer 0 in the Q3 control and layers 0–4 in K2.
No missing per-layer timings are reconstructed. See [data provenance](DATA_PROVENANCE.md)
for dataset identity, observation coverage and redistribution boundaries.
