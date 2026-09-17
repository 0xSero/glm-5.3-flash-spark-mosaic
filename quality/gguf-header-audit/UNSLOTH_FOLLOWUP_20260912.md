# Unpruned lead: fresh source check, September 12

Primary source: https://unsloth.ai/docs/models/glm-5.3-flash#faster-inference-and-mtp-support
Post: https://x.com/UnslothAI/status/2095852388888522890

The guide advertises 3-bit on128GB devices. Its detailed table distinguishes120.37GB UD-IQ3_XXS (81.63%top1,KL0.283772) from147.54GB UD-Q3_K_XL (86.25%,KL0.159697). Do not attach the opening rounded87% claim to the120.37GB artifact. External quality panels are not directly comparable to our frozen WT2 results.

The published speed table uses oneB200 and UD-IQ1_S, so its3.3x claim is not a measured Spark/EXL3 multiplier. Its testedMTPdepth2 performs better than depths3/5 at4k; this supports measuring depth, not transferring results across runtimes.

Our pinned GGUF header audit records all288experts retained, mixed gate/up/down types, and quantized target nonexperts/MTP. Our target protected weights remain native. UnprunedK2 already passed separate262k,vision/video and nativeMTP tests, but77.38%matchedagreement remains insufficient. Projection14 improves to78.38%; originalQ3reference87.38%remains a separate memory/quality reference.

Continue unpruned precision-allocation and draft-memory work first. Full new-server NVFP4 admission is unverified because its host is unreachable. No claim of high-quality unpruned3bpw plus262k fit; no serving migration or public promotion performed.
