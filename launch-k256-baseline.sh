#!/usr/bin/env bash
set -euo pipefail
root=/home/valentine/glm53-single-spark-release-20260911
python3 - "$root/native-mtp/runtime-receipt-k256-r3-m93.json" <<'PY'
import json,subprocess,sys
r=json.load(open(sys.argv[1]))
i=json.loads(subprocess.check_output(['docker','inspect',r['container_id']]))[0]
assert i['State']['Running'] and i['Image']==r['image_id']
assert r['fresh_text_passed'] and r['thinking_enabled']
assert r['candidate_manifest_sha256']=='7c7a2e47c7cde8996ac54c6f2ca5c3f7532b682d5d7296408c97a14e8950f73c'
PY
mkdir -p "$root/benchmarks"
docker run -d --name glm53-k256-legacy-baseline-client \
  --network host --cpus=2 --memory=3g --memory-swap=3g \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$root/acceptance-k256-baseline1":/acceptance:ro \
  -v "$root/native-mtp/runtime-receipt-k256-r3-m93.json":/runtime-receipt.json:ro \
  -v "$root/k2-massmax-k256":/model:ro \
  -v "$root/benchmarks":/results \
  --entrypoint python3 \
  sha256:cd6ce6e10276f856d9ec617c702362ffc45114cad6f26ddcb6a39ef5a2ac7e37 \
  /acceptance/run.py --base-url http://127.0.0.1:18080/v1 \
  --metrics-url http://127.0.0.1:18080/metrics --model glm-5.3-flash \
  --tokenizer /model --runtime-receipt /runtime-receipt.json \
  --max-num-seqs 1 --kv-capacity-tokens 693225 \
  --lengths 1024 --concurrencies 1 --repeats 2 \
  --template-json '{"enable_thinking":true}' \
  --output /results/k256-legacy-c1-1024-r1
