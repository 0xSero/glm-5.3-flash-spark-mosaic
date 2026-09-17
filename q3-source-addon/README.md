# Exact Q3 auxiliary serving addon

Source: `0xSero/GLM-5.3-Flash-EXL3-3.0bpw`, immutable revision
`2a30ad09c15f779a44fa62c216f5dbe5fb0c9223`.
Addon manifest SHA256:
`68136b1646610d2c04efb4c8761e5d374d25096594e4344629e846e04fbacc03`.

`HUB_INVENTORY.json` records the actual pinned Hub file listing. Its complete
runtime auxiliary closure is the five files in `payload/`: tokenizer JSON/config,
processor config, generation config and chat template. No extra vocab/merges,
special-tokens file, separate image/video processor config or auto_map code is
present in this source. The native `Glm5NextProcessor` class is required from the
matching installed Transformers/runtime environment.

Each file was verified against its Hub LFS SHA256 or Git blob SHA1 as applicable,
then assigned a SHA256 in `ADDON_MANIFEST.json`. The exact source config is kept
only under `reference/`; original MIT license, notices and provenance stay under
`provenance/`. These are not candidate runtime-acceptance claims.

Real CPU smoke in the existing observer image loaded `TokenizersBackend`
(vocabulary154820) and `Glm5NextProcessor` with trust_remote_code=false and offline
mode, using the payload plus source reference config in a temporary view. No CUDA
context was initialized. `cpu-smoke.json` and stdout/stderr preserve that evidence.
This proves metadata/class loading, not semantic vision/video inference.

The Q3 chat template is10644bytes with SHA256
`34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891`;
it differs from the modified8757-byte template in the K2 source. Do not substitute
that K2 template when declaring exact Q3 source reproduction.

The addon is staged separately under the release's `q3-source-addon/` directory on
2822,557f and de5c. No candidate/source core config, index, manifest or weights were
changed. A future runtime view may combine an accepted candidate with `payload/`
while separately recording both manifest hashes.
