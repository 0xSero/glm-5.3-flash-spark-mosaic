# K2 keep256 staging and queued quality (private operations)

Source candidate manifest:
`7c7a2e47c7cde8996ac54c6f2ca5c3f7532b682d5d7296408c97a14e8950f73c`.
133 safetensors files total102727970296 file bytes; tensor payload102659360376 bytes.
The completed source stays immutable on557f. A tar stream passes through2822 to
de5c without storing the weight copy on2822.

`stage_and_queue.py` runs on2822 under the new release's `staging-k2-k256/` folder.
It verifies the exact source completion/manifest before and after transfer, checks
de5c free space (payload plus60GiB floor), and refuses an existing destination.
Interrupted transfers retain their partial destination and logs for explicit
inspection; the script never deletes/restarts a partial transfer automatically.

After staging, `validate_stage.py` runs in the existing pinned observer image with
no GPUs/network,2GiB memory and one CPU. It freezes the existing evaluator code,
verifies exact adapter/dependency hashes, then reuses its structural validator to
hash every candidate payload and bind config/index/MTP protection. The copied
candidate is mounted read-only for all validation and inference work.

`queue_quality.py` then waits for exact original Q3 attempt2 container
`4f6816f2653e217ef0cd0a552cba7a9be5b2c4c7e7f5e3b864934af5e7aec3ef`.
It cannot launch if that process is running or exits nonzero. Its successful report
must have65504 positions and all32 normalized row hashes verified against the
report, fixture and teacher identities. Final original logs/inspect are copied
into the new queue directory, without editing original results.

The queue then checks both GPU process IDs and Docker GPU reservations. Only once
both are clear does it launch `glm53-quality-k2-massmax-k256-attempt1` with the frozen
adapter, all45 layers,256 experts and the same BF16 head arithmetic/held-out panel.
Candidate output is separate: `results/k2-massmax-k256`. On termination it retains
all logs/inspect; success additionally seals the complete report and normalized
rows. Failure is retained without automatic overwriting or changing settings.

Inspect `status.json`/`stage.log` on2822 and `queue-status.json`/`queue.log` on de5c.
Receipts are `stage-receipt.json`, `quality-launch.json`,
`original-q3-results-seal.json` and eventually `candidate-results-seal.json`.
Do not publish these operational files directly: they contain private host paths
and network details. Publish sanitized score/provenance reports that reference the
sealed artifact and evidence hashes instead.

Two CPU guard tests prove running or failed original controls cannot cause a
candidate GPU launch. The source payload and de5c free capacity were checked live
before staging. Runtime acceptance is separate from this offline quality run.

Metadata privacy scan in `public-metadata-audit.json` covers146 JSON/text/Jinja
files (79,217,878 bytes), including full index/tokenizer. It found no private paths,
known fleet hostnames or private-LAN/Tailscale address strings. It does not grant
permission to publish operational receipts, or change any sealed artifact.

## Completed candidate quality

The candidate exited successfully and both result seals passed. This records measured quality, not release acceptance.

| Matched 65,504-position measure | K2 keep256 |
|---|---:|
| KL(BF16 → candidate) | 0.6879073281 |
| Top-1 agreement | 71.349841% |
| Candidate perplexity | 5.824290376 |
| BF16 perplexity | 3.199529500 |
| Perplexity increase vs BF16 | 82.035839% |

`candidate-results-seal.json` binds candidate/fixture/teacher identities and all32 normalized row hashes. `quality-integrity-seal.json` additionally binds successful terminal container status and verifies the normalized BF16 tensor shapes. The quality report SHA256 is `deaee2accaefcfd2afe3089fb645dd4e4c7c925610c42acd600bb3fa288cf5bb`.

Raw rotating layer receipts33–44 were retained by the monitoring sidecar. Receipts0–32 had already rotated before that sidecar started and are explicitly recorded as unavailable; no per-layer timing claims are made for those IDs. This limitation is separate from successful verification of all final normalized rows. Compare the matched unpruned K2 control to isolate the pruning penalty.
