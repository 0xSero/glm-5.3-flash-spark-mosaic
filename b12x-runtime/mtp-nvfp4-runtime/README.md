# Native MTP NVFP4 packed-route runtime candidate

This is an isolated staged overlay over the pinned Jovian/B12x projection image.
It has not passed full-model serving, speed, quality, or multimodal acceptance.
The built Docker image is
`sha256:49e834851ff47ff01f3601122936f692cde75f253670aef0f895e44e40174ae0`,
tagged `glm53-b12x-exl3:mtp-nvfp4-packed-staged1` on the assigned test node.
Eight installed-import/config/cache/warmup CPU tests and three installer tests
passed. `BUILD_RECEIPT.json` binds the source, image and terminal CPU evidence.
The preceding prototype passed the eight-expert packed-route component checks
and separately loaded all 288 native draft experts. Those receipts do not prove
this new subclass's full 288-expert sparse execution; that is the next GPU gate.

`SOURCE_PINS.json` fixes the existing vLLM native-MTP dispatchers, FP8 policy,
B12x adapter, online NVFP4 quantizer, backend selector and B12x planned API.
The installer rejects unknown source hashes before writing anything. It modifies
only the V1/V2 native-draft policy call and installs one new module. Target EXL3,
native protected tensors, shared embeddings/heads and multimodal code are unchanged.

The NVFP4 branch requires:

```sh
GLM53_MTP_EXPERT_NVFP4=1
GLM53_MTP_EXPERT_FP8=0
VLLM_B12X_MOE_FP4_FORCE_A16=1
VLLM_MTP_NVFP4_LM_HEAD=0
VLLM_MXFP8_LM_HEAD=0
```

Both precision flags default to zero and are mutually exclusive. The existing
FP8 branch remains available when NVFP4 is off. This adapter accepts only the
separate native GLM MTP config with all 288 experts, SM121, TP1/EP1/DP1 without
EPLB, hidden size 4096, intermediate size 2048, top-8 and BF16 activations.
Only `model.layers.45.mlp.experts` is quantized. Dense layers remain unquantized.

`PackedDraftB12xExperts._plan` resolves an ordinary provisional plan, copies its
policy context with only `w4a16_route_mode='packed'` changed, and caches the new
packed plan. The provisional plan is never warmed or launched. Fast math stays
true. No test launch recorder or global backend monkeypatch is installed.

`workspace_shapes` and `apply` are inherited unchanged, so packed scratch sizes
come from the normal workspace manager. Warmup calls the upstream implementation
one capacity at a time; it retains its lifetime protection and device completion
barrier. Capture rejects missing plans and plans without completed eager warmup.
The upstream compiled-launch cache independently rejects compilation in capture.
No existing captured plan can be replaced through this policy.

Build only against a separately verified pinned base:

```sh
docker build --network=none --pull=false \
  --build-arg BASE_IMAGE=glm53-b12x-exl3:projection-qualified-source \
  -t glm53-b12x-exl3:mtp-nvfp4-packed-staged1 .
```

The base image recorded in SOURCE_PINS is a local evidence identity, not a public
pull reference. Public publication still needs the parent reproducible base and
new serving acceptance. This directory does not change launch contexts, KV dtype,
vision limits, chunk size, memory fraction or speculative depth.

For the next isolated full-loader/sparse-route proof, use the installed import:

```python
from vllm.model_executor.layers.quantization.glm_mtp_expert_precision import (
    maybe_enable_mtp_expert_precision,
)
draft_config = maybe_enable_mtp_expert_precision(native_draft_config)
# The existing strict native loader then constructs and loads the real draft.
# Its routed method has the same B12xNvfp4DraftMethod class name as the prototype.
kernel = routed.quant_method.moe_kernel.fused_experts
kernel.warmup_launches(routed, token_counts=(1, 2, 4, 8))
metadata = kernel.packed_plan_metadata()
```

For a proof-local import on the old image, import the same function from the
mounted `glm_mtp_expert_precision.py`. Its dependencies already exist in the pinned
base. Do not call the old prototype's `enable_for_probe` after this function.
`packed_plan_metadata()` reports actual cached policy/capacity/scratch and warmup
coverage. It explicitly does not claim to observe physical launch flags.

CPU checks use actual installed imports with CUDA uninitialized. The plan factory,
hardware constructor and model-load boundary are mocked and labeled as such.
`test_install.py` separately checks exact-source application, idempotence,
check-only behavior and rejection of unknown modifications without partial edits.

Next gates: full 288-expert strict production load; sparse top-8 addressing across
the entire expert range; independent decoded-weight reference; changed-input and
repeated CUDA-graph replay; actual serving load, native head/vision preservation,
KV admission, speculative acceptance and matched speed sweeps. Original numerical
tolerances remain unchanged.
