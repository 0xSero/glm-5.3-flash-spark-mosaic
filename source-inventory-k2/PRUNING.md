# K2 REAP physical selection

The shared `../build_reap_checkpoint.py` accepts the two explicit pinned source
inventories. K2 inventory bytes are pinned to SHA256
`36b79ee52ace17008eb917588666e3ae7f6ce5f419470df019da8a27576d594b`;
source revision is `35b4b580debe2a1510a2eb042d9d6d4cd7fb3a6b` from
`0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw`. Any source metadata override or cross-source
inventory mismatch fails before writing candidate shards.

Parent/operator launch only; this command has not been executed by the builder
implementation task:

```bash
python3 build_reap_checkpoint.py \
  --source /home/valentine/flash-experimental-staging-20260906/model \
  --output /home/valentine/glm53-single-spark-release-20260911/candidate-k2-keep256 \
  --candidates /PATH/TO/SEALED/candidates.json \
  --inventory source-inventory-k2/filehash-inventory.json \
  --metric massmax_domain --keep 256
```

The same sealed observation keep maps select256 logical experts per trunk layer.
All four virtual-rank fragments retain original K2 bytes. IDs are remapped
contiguously with router rows/correction biases sliced in the same order. Other
weights retain original precision, including all889 MTP tensors and288 native MTP
experts. The original config remains in `source-config.json` for the native MTP
view. The source is immutable; outputs/receipts are atomic and resumable only under
an identical build contract, including builder SHA. Existing completed Q3 artifacts
are not rewritten or invalidated by this implementation change.

Projected payload before router slicing reduction is102670375800 bytes
(95.6192387GiB): protected33835039608 bytes plus routed77439753216×256/288.
It excludes headers and runtime/KV/activation/OS memory, so this is not a proven
single-Spark fit. Actual final tensor bytes are measured in the build manifest.

CPU tests cover both real pinned metadata inventories, source confusion rejection,
Q3/K2 bitrate fields, native MTP contract, exact K2 trellis bytes/shape for all four
rank fragments, router remapping, protected NaN payloads, duplicate keep rejection,
and empty-shard handling. Eight tests pass. No full build or GPU load was launched
as part of implementing this support.
