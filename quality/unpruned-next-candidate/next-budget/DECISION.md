# Next unpruned precision budget

Projection6 completed with77.9769% teacher top1agreement, KL0.4198133, PPL4.4422095. That is not accepted as the requested high-quality endpoint. The prospective allocations below are independently derived from pinned paired calibration errors; the script does not read held-out quality reports or token labels. Proxy size does not establish a proportional agreement benefit: the first measured projection allocation already shows why that extrapolation would be unreliable.

The next justified experiment is **projection14 with draft-only NVFP4**, conditional on a real full-draft resident saving of at least2.75GiB,262144-token post-graph admission, and sufficient OS reserve. It upgrades down projections in layers5,6,7,20,22,23,26,27,28,31,32,35,36,37. Gate/up and all other experts stay K2; target protected tensors remain unchanged. If actual savings are smaller, projection12 is the conservative fallback. No new checkpoint or GPU job was started by this assessment.

The native MTP has864 expert matrices,7,247,757,312 weights. Current measured FP8 expert parameters including two scale vectors occupy7,247,759,616bytes. NVFP4 nominal data uses3,623,878,656packed-weight bytes +452,984,832group-16FP8scale bytes +4,608bytes for fourFP32vectors =4,076,868,096bytes. Nominal saving:3,170,891,520bytes (2.953123GiB). Its other25native tensors occupy369,670,784bytes and remain unchanged. Backend reordering/padding, transient conversion, workspaces, graph buffers and allocator effects are excluded until measured. Shared target head/embedding must not be counted twice.

| K3 down layers | Extra target GiB vs K2 | Net GiB vs K2+FP8 draft after nominal NVFP4 saving | Source artifact tensor bytes including native MTP |
|---:|---:|---:|---:|
| 6 | 1.6875 | -1.2656 | 113,086,732,152 |
| 12 | 3.3750 | 0.4219 | 114,898,671,480 |
| 14 | 3.9375 | 0.9844 | 115,502,651,256 |
| 18 | 5.0625 | 2.1094 | 116,710,610,808 |
| 20 | 5.6250 | 2.6719 | 117,314,590,584 |

Projection14 adds2.25GiB over projection6 and retains~0.703GiB of nominal draft savings as margin. Projection18/20 are later capacity frontiers, not admission claims. The baseline's~6.24GiB MemAvailable and5.84GiB KV are observed totals; reclaimable KV is not free memory until a new runtime proves262k and required concurrency after graph capture. Do not spend the entire~3.8GiB theoretical KV slack or call it an automatic weight allowance.

**Why not immediately redo K2?** The paired MIT path already includes calibrated Hessians, output/global scale search, LDLQ feedback and exact decode checks. No missing numerical setting has been demonstrated; frozen evaluators disable TF32. A worthwhile K2 pilot requires preserved independent fitting and checking activations plus exact source/quantizer pins, then seed/damping/objective comparisons on those independent data. Existing available per-part proxy summaries are not substitutes for those activations. Changing the MCG codebook needs kernel compatibility proof. A large recollection/requantization is not justified by an assumed easy quality gain.

EXL3 MTP compression could nominally reduce expert payload further, but the existing paired expert calibration covers only target layers3–44, not MTP45. Reusing target-layer quantized experts is invalid. Native MTP calibration must use its actual draft hidden states and the loader must accept separate serialized MTP precision without inheriting target settings. NVFP4 uses the native MTP source directly and the separate B12x SM121 W4A16 path, so it is the concrete next bounded runtime experiment. Lower draft precision may reduce acceptance/speed; target-distribution preservation and actual performance need separate validation.
