# Upstream changes inspected on September12

The running GLM baseline remains pinned to B12x3b862805 / vLLM3aada677. Latest fetched B12x864b630 adds two DeepSeekV4.1 commits: lagged mHC parallel decode and W4A8 MoE microkernels/benchmarks. The new W4A8 kernels do not establish EXL3 support, two-Spark topology performance, or a GLM speedup. mHC changes include an explicit no-alias output contract and need the supplied numerical oracle and CUDA-graph checks on GB10 before use.

vLLMd783e291 changes the DS4.1 TP4 serving example from5 to7 DSpark draft tokens and removes expert-parallel; its remaining change fixes test import locations. Those TP4 defaults must not be copied blindly into a two-Spark EXL3 recipe.

Fetched source is staged, not deployed. Current calibration remains on its bitwise-qualified native path. The DeepSeek runtime owner has these exact new pins for serving qualification. Pop remains untouched. Raw diffs and commit IDs are retained beside this report.
