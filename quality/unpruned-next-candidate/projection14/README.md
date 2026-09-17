# Projection14 immutable candidate

This CPU build preserves all288experts in each target MoE layer, all router IDs/rows, all native target protected components, and all288native MTP experts. It upgrades only down_proj to paired K3 in layers5,6,7,20,22,23,26,27,28,31,32,35,36,37. Gate/up remain K2. No held-out quality label was used to choose the order.

Selection SHA256:57926af4ddf6aa5f5cf0180ec6cf57b867f4ec62457d806ecab1310cddf65c78. BaseK2HFrevision35b4b580debe2a1510a2eb042d9d6d4cd7fb3a6b; pairedMITK3status4a4402ab7b47f91145db841ce0105b54c3f7fcc3663a110fa712ceb85d5d1f0f. The same paired calibration errors and immutable source hashes used for projection6 are retained.

Six previously rewritten layers are linked from manifest590ec08e84a66767e03964f7482e84c952e6b98c26cddc7d45983592c4e491f5 after contributor/hash verification. Eight new layers require16pairedK3sourcefiles (22,012,843,776bytes) and16rewrittenparts (~17.18GB). The bridge streams bytes only. Neither prior weights nor calibration files are deleted or edited. Builder2CPU/8GiB; noGPU. Structural success is separate from quality/serving acceptance.

## Exact artifact frontier

|K3 down layers|Extra target GiB vsK2|Net GiB after nominaldraftNVFP4saving|Tensor bytes including native sourceMTP|
|---:|---:|---:|---:|
|6|1.6875|-1.2656|113,086,732,152|
|12|3.3750|0.4219|114,898,671,480|
|14|3.9375|0.9844|115,502,651,256|
|16|4.5000|1.5469|116,106,631,032|
|18|5.0625|2.1094|116,710,610,808|
|20|5.6250|2.6719|117,314,590,584|
|22|6.1875|3.2344|117,918,570,360|
|24|6.7500|3.7969|118,522,550,136|
|42|11.8125|8.8594|123,958,368,120|

The third column assumes nominal3,170,891,520byte draft savings; it does not measure them. Actual admission must reserve the successfully exercised262144-token KV requirement, full-graph buffers, measured prefill/vision activation peaks, backend workspaces and OS headroom using a fresh full runtime load. Baseline excessKV is not automatically credited. The nativeMTP in this artifact remains unchanged; draft-only NVFP4 is a separate launch-time experiment owned by the parent.
