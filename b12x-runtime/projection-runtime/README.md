# Projection-specific EXL3 runtime overlay

Status: built on DGX Spark de5c and passed CPU import/actual candidate metadata checks. Full-model loading, quality and serving acceptance are pending. No running model was changed.

The overlay installs the exact wrapper/helper already used by the real K(2,2,3) MoE GPU and CUDA-graph test. It preserves native target weights, supports a per-layer expert projection map, and rejects unknown installed wrapper hashes before modifying the new image. All base layers, environment, entrypoint and command were preserved. Docker reported 28522 additional image bytes; this is not a model-memory measurement.

Built image: `sha256:368820997e1146e9d7843367478b53ce18db708e79861f7ea269860e9a1bda4b`.
Local tag: `glm53-b12x-exl3:projection-qualified-source`.
Qualified local R3 base: `sha256:c0d5f76237a5fe1262291eb23b8086ef0b2637e6d8d45ccdbb9c5706df321399`.

From this directory, after independently verifying the base image identity:

```sh
docker build --pull=false --network=none   --build-arg BASE_IMAGE=glm53-b12x-exl3:jovian-3aada677-r3   -t glm53-b12x-exl3:projection-qualified-source .
```

This local overlay is not a standalone public release. The parent B12x runtime Dockerfile contains the source-pinned base build. Publication needs the selected weight artifact, real serving qualification and a fresh public reload.

`BUILD_RECEIPT.json` binds source, build log, image metadata and CPU gate evidence. `cpu_import.py` parsed the actual six-layer candidate configuration, verified all 42 routed layer precisions and preserved the default native Linear path with CUDA uninitialized. The GPU microproof covers eight real experts in one MoE, not a full model. Both boundaries are explicit.

First CPU invocation failed because the host has no named `nvidia` Docker runtime. Attempt 2 mounted driver libraries through the supported GPU request mechanism, hid all devices with empty `CUDA_VISIBLE_DEVICES`, and verified no CUDA initialization. Both logs are preserved.
