#!/usr/bin/env bash
# Download the published M288-12L mosaic (mixed 2.05/3.05 bpw EXL3, codebook mul1).
#
#   repo: 0xSero/GLM-5.3-Flash-EXL3-Spark   (~96.1 GB, 12 shards)
#   pin:  2642851741fc833764e77d03039117be559dc83e   (the revision this repo's
#         measured results were produced from)
#
# Verifies every downloaded file against the sha256-manifest.txt the model repo
# ships, so a truncated or substituted download cannot reach the server.
#
# Usage:  ./download-mosaic.sh [dest_dir]      (default ~/models/mosaic-12l)
set -euo pipefail
DEST="${1:-$HOME/models/mosaic-12l}"
PIN="${GLM53_MOSAIC_PIN:-2642851741fc833764e77d03039117be559dc83e}"
REPO="${GLM53_MOSAIC_REPO:-0xSero/GLM-5.3-Flash-EXL3-Spark}"

python3 - "$DEST" "$PIN" "$REPO" <<'EOF'
import sys
from huggingface_hub import snapshot_download
dest, pin, repo = sys.argv[1], sys.argv[2], sys.argv[3]
p = snapshot_download(repo, revision=pin, local_dir=dest, ignore_patterns=[".cache/*"])
print("downloaded:", p)
EOF

cd "$DEST"
if [ -f sha256-manifest.txt ]; then
  echo "### verifying files against sha256-manifest.txt"
  sha256sum -c sha256-manifest.txt
  echo "### all files verified"
else
  echo "WARNING: sha256-manifest.txt not present — cannot verify the download" >&2
  exit 1
fi