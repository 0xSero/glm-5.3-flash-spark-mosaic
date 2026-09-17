#!/usr/bin/env bash
set -euo pipefail
python3 /opt/verify_bootstrap_exllama.py
root=$(python3 -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent.parent)')
# The verified bootstrap has one legacy overlay file outside wheel RECORD.
# A normal package uninstall therefore leaves it behind. Retire only that
# exact known file before the source-pinned port refuses arbitrary overwrites.
python3 - "$root" <<'PYCODE'
import hashlib, pathlib, sys
path = pathlib.Path(sys.argv[1]) / 'vllm/model_executor/layers/quantization/exl3.py'
if path.exists():
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = '41fc9bb42f3a555dbcef05b8b6c4fb1fa63dfe80e844f66078d160d4f3235db9'
    if digest != expected:
        raise ValueError('Unexpected existing EXL3 overlay: refuse removal')
    path.unlink()
    print('Removed verified legacy EXL3 overlay outside wheel RECORD:', digest)
PYCODE
python3 /patches/exl3-port/apply_port.py --root "$root"
python3 /patches/apply_native_mtp_port.py --root "$root"
# Actual native import gates require the real host CUDA driver. Run
# /opt/verify_runtime.sh under NVIDIA_VISIBLE_DEVICES=none after building.
python3 -m compileall -q "$root/vllm/models/glm5next" "$root/vllm/model_executor/layers/quantization/exl3.py"
