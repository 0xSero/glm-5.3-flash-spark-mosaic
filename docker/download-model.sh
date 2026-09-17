#!/usr/bin/env bash
# Download the public a0 EXL3 quant (2.05bpw, codebook mul1) used by every
# receipted result in this repo.
#   repo:  turboderp/GLM-5.3-Flash-exl3, branch "2.05bpw"
#   pin:   51058cd551c7e570d87bd32a4adee720edce2349  (the program's recorded revision)
set -euo pipefail
DEST="${1:-$HOME/models/GLM-5.3-Flash-exl3-2.05bpw}"
PIN="${GLM53_2P05_PIN:-51058cd551c7e570d87bd32a4adee720edce2349}"

python3 - "$DEST" "$PIN" <<'EOF'
import sys
from huggingface_hub import snapshot_download
dest, pin = sys.argv[1], sys.argv[2]
p = snapshot_download("turboderp/GLM-5.3-Flash-exl3", revision=pin, local_dir=dest)
print("downloaded:", p)
EOF

# Record what we actually got, so any run is attributable to bytes.
( cd "$DEST" && find . -type f ! -path './.cache/*' -exec sha256sum {} \; | sort -k2 > sha256-manifest.txt )
echo "wrote $DEST/sha256-manifest.txt"
