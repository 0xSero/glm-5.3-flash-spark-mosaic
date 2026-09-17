# Mixed5+32 frozen quality measurement

Separate snapshot of the original all-45-layer adapter and byte-identical evaluator dependencies. No frozen baseline files were edited. The new validator accepts only mixed candidate manifest `73291ede5c0f2ad13dcd9fbdfe3887f133ef713d56c589c59c6392bdb887b45c`, rehashes every model/metadata file, checks source tiers and packed trellis shapes, and verifies indexed and native protected coverage.

Every routed slice reconstructed during capture checks both its packed shape and the real LinearEXL3 module's inferred K against the immutable layer map. `probe_reconstruction.py` first reconstructs all three projections and four ranks for expert0 in layers3,5,32,44 (48 slices) on the real GPU extension; it checks BF16 output shapes and finiteness. This smoke is not a quality score.

The supervisor runs CPU preflight, then the bounded reconstruction smoke, then all45 trunk layers and the original BF16 head comparison. Each GPU phase requires an idle device. Capture is capped at four CPUs and64GiB, with no network. Results use the new `results/mixed-k2-k3-l5-l32` directory; all container receipts and failures are retained.

Comparison remains the same32×2048 input rows,65,504 scored positions, frozen BF16 normalized teacher and unchanged native head. The two upgraded layers were chosen from independent600-row calibration proxy evidence. MTP is preserved in the artifact and excluded from target-trunk quality as in the baseline. This does not establish actual serving parity, context admission, vision, MTP behavior or release acceptance.

## Actual launch gates

The full source/hash preflight passed. The first reconstruction probe failed because it incorrectly expected the raw EXL3 return tensor to be BF16. Actual reconstruction returns FP16; the frozen evaluator explicitly casts into native BF16 expert storage. The isolated R2 probe uses that same cast and passed all48 slices with both K2 andK3, recording rawFP16 and targetBF16. Peak allocation was55,396,864 bytes. The first failed container and logs remain in `launch-receipts/`; the adapter and its arithmetic were unchanged.

Capture CID: `91355999a29cb044e0454c0474d16ad95026c6003d0da9a30f3363c8fbe0ef75`. All45 raw layer receipts are monitored from launch. The separate terminal supervisor will verify every final normalized row and create an integrity seal after a successful exit. No agreement/KL result exists until that comparison completes.
