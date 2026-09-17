# Q3 keep192 CPU build (private operations)

Completed on de5c with the unchanged shared builder. The Docker container exited0,
all130 files completed, and the monitor verified the bound completion seal and
metadata hashes. Two-CPU/8GiB/no-extra-swap limits and no GPU devices were confirmed
from Docker inspect. The source remained mounted read-only.

| Field | Verified value |
|---|---|
| Manifest SHA256 | `98e72202d67acc158e908701d04b6d228de74b01d8c1c66d3aa0df6a998b1cbc` |
| Tensor payload | 110795646072 bytes /103.186486GiB |
| Safetensors file bytes | 110847041992 |
| Files | 130 |
| Trunk experts per routed layer | 192 |
| Routed payload bitrate | Original K3; no requantization |
| Native tensors unchanged | 2398 |
| Router tensors sliced | 84 |
| Native MTP experts | Original288 |
| Initial disk admission | 243.88GiB free;150GiB minimum gate passed |
| Quality/runtime acceptance | Not run; no fit or speed claim |

`completion.json`, `EXL3_MANIFEST.json`, `final-inspect.json` and `build.log` retain
the exact result. `builder.py` is the frozen unchanged source SHA
`b5cd69f87c106c161b1349707a3b4bdcf8cd70c504d615cac76d2443f929f734`.
The same sealed massmax-domain maps SHA
`4fd8de1e8d0d3e05158b0989a6b18813430f30b280f5499870e7e75b06d02ecd`
were used. The keep176 fallback and K2keep256 quality queue were preserved.

Artifact: `/home/valentine/glm53-single-spark-release-20260911/q3-massmax-k192`.
These operational files contain private paths and should not be uploaded raw.
