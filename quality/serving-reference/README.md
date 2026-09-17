# Private BF16 serving reference

Do not publish this directory: the token IDs recover the private frozen evaluation corpus.

`reference.json` contains `input_ids[32][2048]` and `teacher_top1[32][2047]`, plus immutable fixture/teacher/native-head identity, arithmetic and per-row teacher NLL. Position p in teacher_top1 predicts input token p+1, for p=0..2046. Raw serving requests must not insert special tokens.

SHA256: `582c80ba9f9780be10a5064c4952bbdc6a71bc09b21ec353b5043fc2fa6ee4c3`.

Preparation verified all32 frozen normalized BF16 row hashes and shapes, the exact token fixture, and the unchanged native BF16 head shard. It used BF16 F.linear with the full2047-position GEMM and vocabulary chunks4096, followed by FP32 logits. Per-chunk torch.max selects the first token ID; strict-greater updates between chunks retain the lowest ID on ties. A CPU tie test passed without CUDA initialization.

All32 teacher NLL sums exactly reproduced the existing matched baseline, with maximum absolute difference0.0. Teacher perplexity is3.199529500063886. The bounded GPU phase took14.43seconds and peaked at176,216,064 allocated bytes; the container exited0 without OOM. These are reference preparation facts, not model serving benchmarks.

`identity.json` binds separate array hashes, all row hashes, source and runtime arithmetic settings. `final-inspect.json`, `launch.json`, and `prepare.log` retain execution evidence. `combine.py` verifies both arrays before producing the combined API measurement input. The parent owns serving requests and scoring separately.
