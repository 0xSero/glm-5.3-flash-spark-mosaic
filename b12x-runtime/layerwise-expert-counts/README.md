# Layerwise GLM routed-expert counts: isolated prototype

This adds different retained expert counts per routed layer to the pinned Local Inference Lab Jovian vLLM source. It is independent of the separate `layer_bits` overlay. The running Spark2384 server is unchanged; no full pruned model or quality acceptance is claimed.

## Artifact contract

Place both maps in the text config. Keep global `n_routed_experts:288` and the original native source config intact:

```json
{
  "n_routed_experts": 288,
  "routed_experts_per_layer": {"5": 8, "32": 16},
  "retained_expert_ids_by_layer": {
    "5": [32, 33, 34, 35, 36, 37, 38, 39],
    "32": [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
  }
}
```

This is the small smoke-test selection, not a quality recommendation. Each ID list must contain exactly the declared number of unique, sorted native IDs. Omitted layers retain all288 original experts. Native MTP layer45 retains all288. No dense layer can appear in these maps.

An artifact builder must select router weight rows, correction-bias rows and expert tensors using the **same ordered original IDs**. It must rename the selected expert tensor IDs to contiguous0..N-1 and retain all four archive ranks, three projections and four EXL3 fields. These are physical artifact operations; the runtime does not silently reinterpret an unpruned checkpoint. Builder/source hash and row-equality verification remain separate requirements.

## Runtime scope

`layer_config` validates the plan and creates a layer-local config copy. If a multimodal wrapper is passed, its text config is also copied, avoiding shared nested-config mutation. The existing decoder then passes this local config to the existing gate, bias and FusedMoE constructors. Router scaling, top-k8 and normalization are unchanged.

Initial support is TP1/PP1/DP1/CP1, EP off, EPLB off, no redundant experts, no sequence parallelism, one routing group. Unsupported modes fail before routed-layer allocation. Global MoE/EPLB metadata has scalar-count assumptions and is not extended to heterogeneous counts. Native MTP has its own single288-expert layer and remains unchanged.

The existing target loader's native288 expert mapping is an upper bound containing every valid compact ID. A new per-tensor guard rejects routed IDs beyond that layer's count and wrong router/bias shapes. The plan is validated once per load, not for every tensor. Existing EXL3 rank/shape/MCG/post-load checks remain active. This prototype does not replace a complete artifact index/hash validator.

## Apply and tests

Use the local virtual environment created through `uv`:

```bash
.venv/bin/python -m unittest discover -s . -p 'test_*.py' -v
.venv/bin/python apply_patch.py --model-file /staged/vllm/models/glm5next/nvidia/model.py --check-only
.venv/bin/python apply_patch.py --model-file /staged/vllm/models/glm5next/nvidia/model.py
```

Patch application accepts only the pinned original model SHA or its exact known patched bytes. It is idempotent and rejects unrelated edits. `model.original.py`, `model.patched.py`, the reviewable diff and `source-contract.json` are staged copies only.

Nine CPU tests cover all45 target layers, nested config isolation, unchanged top-k/global geometry, MTP protection, malformed retained-ID maps, unsupported parallelism, loader bounds/router shapes, patch guards, and the actual pinned MoE constructor's gate/bias/FusedMoE arguments. GPU allocations in that constructor unit test are explicit CPU recorders, not kernel evidence.

The actual GPU smoke uses real source experts32..39 at layer5 and80..95 at layer32: two standalone MoEs with8 and16 experts, full hidden4096/intermediate2048 geometry and four-rank K2 archives. It checks retained router/bias rows, native shared weights, all384+768 packed expert fields, actual EXL3 fused versus individual-kernel and reconstructed-weight references, production router/shared MoE execution, and CUDA graphs. It does not load attention, vision, KV cache, MTP or a full language model. Therefore it demonstrates implementation feasibility only.

## Rollback preservation

The parent authorized stopping the idle experimental557f legacy server for this bounded test. Its original container used `--rm`, so inspect/log/metrics were saved and a same-image/config/mounts **unstarted rollback clone** was created with AutoRemove disabled before stopping the original. The original container did not survive its automatic removal. The retained clone starts with:

```bash
docker start glm53-legacy-k2-fp8mtp-rollback-20260911
```

This remains a cold model reload. The image and source weights are preserved. Spark2384 B12x, PopOS serving and the other research lanes were not stopped or modified.
