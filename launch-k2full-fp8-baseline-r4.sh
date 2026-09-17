#!/usr/bin/env bash
set -euo pipefail
root=/home/valentine/glm53-single-spark-release-20260911
python3 - "$root/native-mtp/runtime-receipt-k2full-fp8draft-r4.json" <<'PY'
import json,subprocess,sys
r=json.load(open(sys.argv[1]))
i=json.loads(subprocess.check_output(['docker','inspect',r['container_id']]))[0]
assert i['State']['Running'] and i['Image']==r['image_id']
assert r['fresh_text_passed'] and r['thinking_enabled']
assert r['candidate_manifest_sha256']=='501641b947fa56afc9ed098bdc034e59c2bd229d4fa1a1d7b80e3727f10c8ba3'
PY
mkdir -p "$root/benchmarks"
docker run -d --name glm53-k2full-fp8-legacy-baseline-client-r4 \
  --network host --cpus=2 --memory=3g --memory-swap=3g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$root/acceptance-k2full-fp8-r4":/acceptance:ro \
  -v "$root/native-mtp/runtime-receipt-k2full-fp8draft-r4.json":/runtime-receipt.json:ro \
  -v "/home/valentine/flash-experimental-staging-20260906/model":/model:ro \
  -v "$root/benchmarks":/results \
  --entrypoint python3 \
  sha256:4f06a26d872ddd6f268a1f99e85940ea630a3cbecc33c7cc8b0038faf8c0851b \
  /acceptance/run.py --base-url http://127.0.0.1:18080/v1 \
  --metrics-url http://127.0.0.1:18080/metrics --model glm-5.3-flash \
  --tokenizer /model --runtime-receipt /runtime-receipt.json \
  --max-num-seqs 1 --kv-capacity-tokens 460208 \
  --lengths 1024 --concurrencies 1 --repeats 2 \
  --template-json '{"enable_thinking":true}' \
  --output /results/k2full-fp8-legacy-c1-1024-r4
