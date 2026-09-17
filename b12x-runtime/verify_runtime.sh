#!/usr/bin/env bash
set -euo pipefail
python3 /patches/exl3-port/cpu_import_smoke.py
python3 /opt/smoke_build.py --output /opt/build-identity.json
python3 /opt/test_mtp_external_embeddings.py --output /opt/mtp-embedding-cpu-evidence.json
python3 /opt/test_mtp_fp8_scope.py --output /opt/mtp-fp8-scope-cpu-evidence.json
python3 /opt/test_b12x_cache_contract.py --output /opt/b12x-cache-contract.json
