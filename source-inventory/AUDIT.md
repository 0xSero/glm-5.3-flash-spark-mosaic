# GLM-5.3-Flash source and candidate audit

Inspected 2026-09-11. Read-only remote inventory and public pinned metadata;
no weight downloads, GPU allocations, process changes or candidate builds.

Source: `0xSero/GLM-5.3-Flash-EXL3-3.0bpw` at
`2a30ad09c15f779a44fa62c216f5dbe5fb0c9223`.
`filehash-inventory.json` contains every original weight SHA/size and metadata
SHA. Original `EXL3_MANIFEST.json` SHA:
`05e0ff9cc6a3f87fbd8e27b46bb679e114579dfea3bc4afcc2d724b58be3d1ee`.

## Byte inventory

| Component | Tensor bytes |
|---|---:|
| All Q3 routed experts | 115,490,479,104 |
| Native MTP layer 45 | 14,865,185,408 |
| Native vision | 1,127,254,016 |
| Other native protected | 17,842,600,184 |
| Total protected | 33,835,039,608 |

All 49 protected-shard headers were inspected through bounded HTTP range
requests. Exact MTP count889, vision347, other1246; total2482.
Routed payload is42 layers ×288 experts ×3 projections ×4 virtual ranks.
Physical Q3 weight files130;149,402,871,912bytes including safetensors headers.

| Keep per layer | Q3 payload with complete native protected, GiB | K2 equivalent, GiB |
|---:|---:|---:|
|160|91.266|71.579|
|176|97.242|75.586|
|192|103.217|79.592|
|208|109.193|83.599|
|224|115.168|87.606|
|240|121.144|91.612|
|256|127.119|95.619|
|288|139.070|103.633|

These are tensor payloads, not runtime fits. KV, activation peaks, allocator,
CUDA graphs/workspaces and OS reserve are additional. MTP weights are already
included here. The old89.89GiB K2 loaded measurement skipped native MTP and
cannot establish the new capacity requirement.

## Reusable local files

- de5c: `/home/valentine/glm53-full-observations-20260907/models/q3-smoke`,84 weight files,78,129,966,960bytes; routed layers3–22.
-557f: `/home/valentine/glm53-full-observations-20260907/models/q3-stage1`,70 weight files,68,665,538,016bytes; routed layers23–44.
- Union128/130 names and sizes match pinned HF.58 files exist only on de5c;
26 duplicates. Full SHA validation is required after staging.
- Only missing source files are `retained/retained-00104-of-00120.safetensors`
and `retained/retained-00105-of-00120.safetensors`,5,368,754,272 and
5,368,754,264bytes respectively. See exact digests in inventory.
- Original HF config/index must replace altered observation-stage metadata;
never overwrite an original through a metadata hardlink.

## Quality baseline

| Artifact | Positions | KL BF16 to candidate | Top1 agreement | PPL change |
|---|---:|---:|---:|---:|
| Q3 |65,504|0.152512806|87.2847%|+9.2969%|
| Q4 |65,504|0.065793622|91.6997%|+2.3797%|
| Hot-route mixed control |51,175|0.115908355|89.0161%|+5.5398%|

Hot-route result is a different25-row panel and a mixed K4 candidate. It is
not proof of a better complete uniform3bpw artifact. Exact K2 BF16 KLD remains
unmeasured in the inspected prior evidence. Q3/Q4 historical panels are matched;
re-run the Q3 control with the recovered teacher before claiming new deltas.

Reusable teacher on Omarchy:
`/home/sero/glm53-exl3/quality-reference-v2/bf16-normalized/`.
All32 normalized hidden-state files were freshly SHA/size verified;
536,873,728bytes. Manifest SHA:
`a1604015f76d6a6b6f032ffb7e3359a770e8dd2564ca89e3b00eb8bcb4d93314`.
Use unchanged native LM head to reconstruct full-vocabulary logits. Historical
head calculation is BF16 F.linear followed by float logits, not FP32 matmul.

Token file `/home/sero/glm53-exl3/quality-eval-v1/token_rows.safetensors`, SHA
`5b77e320eaccc959e5f731639aa4d9b908027e7647192305e374c945b44c7d44`.
Evaluation manifest SHA
`46255621cd2da8e05546edc3498cfbee518da092af2f44ae243f79da97c0f00b`.
32×2048 WikiText2 test rows; source BF16 revision
`a6c167b62691b2bac901344b65cb651a70f53e43`.

## Builder contract

`../build_reap_checkpoint.py` consumes this inventory, sealed original source,
and sealer `candidates.json`. It copies selected expert bytes without decode or
requantization, keeps all4 virtual ranks, renumbers sorted retained expert IDs,
and slices trunk router/bias rows in the identical order. It preserves native
MTP45 with288 experts and saves original `source-config.json` independently.
Candidate index groups original shard-relative paths. No legacy per-layer
expert sidecars are emitted; use the candidate index.

Per-shard SHA checks, atomic receipts/resume, protected payload rereads, final
2398 unchanged native +84 sliced router +889 MTP subset closure are structural
gates. Quality, CUDA graphs, vision, MTP and262144-token serving remain separate.
