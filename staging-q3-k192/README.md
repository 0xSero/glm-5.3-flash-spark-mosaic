# Q3 keep192 staging and guarded quality

This is private operational evidence, not public model metadata.

Only the temporary keep176 indexed weight files on 2822 were eligible for reclamation. Before removal, the original 557f copy was verified against manifest `0230ee28b89f0eab7b24bebca8ddd10ff3e96783f408efe17ca1a9bbe0f39a13`, including all payload sizes and SHA256 hashes. `preserved-copy-proof.json` records that proof; `reclamation.json` records the exact deletion list and before/after disk space. All keep176 quality outputs and metadata remain. The original 557f artifact remains immutable.

Q3 keep192 was copied from the completed de5c build. Its pinned manifest is `98e72202d67acc158e908701d04b6d228de74b01d8c1c66d3aa0df6a998b1cbc`; its 130 weight files contain 110,847,041,992 bytes including file headers. The validator independently checks all payload hashes and the 192-expert index contract before creating `stage-receipt.json`. It freezes the existing quality adapter and dependencies, preserving the original BF16 head arithmetic and matched 65,504-position fixture/teacher panel.

The quality queue requires no GPU process or Docker GPU owner, at least 35 GiB MemAvailable and 20 GiB free storage. It additionally honors `HOLD_GPU_LAUNCH`, installed under parent direction before launch to reserve the GPU for the native-MTP probe. Only explicit parent release may remove that hold. The quality queue has not established a quality result merely by staging or waiting.

Read `status.json`, `queue-status.json`, and `launch.json` for current evidence. On actual completion, the queue retains container logs, final inspect data and a normalized-result integrity seal. Errors preserve evidence. The staging script intentionally is not an automatic resume tool: inspect partial state before recovery rather than rerunning deletion/staging blindly.

Validation: three CPU unit tests cover rejection of an unverified backup, rejection of a different preserved-artifact pin, and prevention of GPU launch while the explicit hold exists even when all resource gates pass.
