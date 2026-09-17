#!/usr/bin/env bash
set -euo pipefail
root=/home/valentine/glm53-single-spark-release-20260911
python3 - "$root/native-mtp/runtime-receipt-k2full-fp8draft-r4.json" <<'PY'
import json,subprocess,sys
r=json.load(open(sys.argv[1]))
i=json.loads(subprocess.check_output(['docker','inspect',r['container_id']]))[0]
assert i['State']['Running'] and i['Image']==r['image_id']
b=json.loads(subprocess.check_output(['docker','inspect','glm53-k2full-fp8-legacy-baseline-client-r4']))[0]
assert not b['State']['Running'] and b['State']['ExitCode']==0
assert r['raw_logprobs_api_contract_passed']
PY
docker run -d --name glm53-k2full-fp8-parity-client-r4 \
  --network host --cpus=2 --memory=2g --memory-swap=2g \
  -v "$root/acceptance-k2full-fp8-r4":/acceptance:ro \
  -v "$root/native-mtp/runtime-receipt-k2full-fp8draft-r4.json":/runtime-receipt.json:ro \
  -v "$root/benchmarks":/results \
  --entrypoint python3 \
  sha256:4f06a26d872ddd6f268a1f99e85940ea630a3cbecc33c7cc8b0038faf8c0851b \
  /acceptance/runtime_parity.py api \
  --base-url http://127.0.0.1:18080/v1 --model glm-5.3-flash \
  --probes /acceptance/original-k2-runtime-parity-probes.json \
  --runtime-receipt /runtime-receipt.json \
  --output /results/k2full-fp8-runtime-parity-r4.json
