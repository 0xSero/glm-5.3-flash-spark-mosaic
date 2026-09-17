# GLM release data provenance

The matched quantization comparison uses Salesforce/wikitext, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, configuration `wikitext-2-raw-v1`,
test split. The frozen fixture selects the first 32 contiguous complete blocks of
2048 tokens. It scores 65,504 next-token positions after dropping the final
unpredictable position from each block. The fixture records no exact row overlap
with the calibration rows; this does not establish semantic or document-level
decontamination from every training or calibration source.

The pinned [dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md)
lists CC-BY-SA 3.0 and GFDL. Publication should include source attribution, fixture
selection, hashes and reproduction code. Raw corpus text, recoverable token IDs,
teacher hidden states and parity-probe prefixes stay outside the public release
payload unless their redistribution terms are separately satisfied. Benchmark
results may be published with the exact source and evaluation procedure.

Fixture manifest SHA256:
`46255621cd2da8e05546edc3498cfbee518da092af2f44ae243f79da97c0f00b`.
Token rows SHA256:
`5b77e320eaccc959e5f731639aa4d9b908027e7647192305e374c945b44c7d44`.
BF16 teacher manifest SHA256:
`a1604015f76d6a6b6f032ffb7e3359a770e8dd2564ca89e3b00eb8bcb4d93314`.

REAP observations are a separate dataset and purpose: 332 sealed shards,
21,248 records and 32,602,850 tokens from `0xSero/reap-calibration-data-v1`,
revision `115e754ab8835025e5b59df4b0f30735d8a40ce8`. Science was absent and
CUDA coverage partial. Eight boundary shards mix domains and contribute only
to pooled scoring. These limitations must remain visible in candidate cards;
the observation token budget is not the held-out quality sample size.

The serving speed fixture is synthetic repeated archive text followed by a
request for Python debugging comments. It is a throughput workload, not an
accuracy benchmark. Paired image/video fixtures separately check OCR, counts,
colors and temporal direction. Publish their source and provenance alongside
semantic receipts; passing them does not establish arbitrary video duration,
resolution or broad multimodal accuracy.
