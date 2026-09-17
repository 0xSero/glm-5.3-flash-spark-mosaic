#!/usr/bin/env bash
set -euo pipefail
python3 /acceptance/behavior_acceptance.py \
  --endpoint http://127.0.0.1:18080 --model glm-5.3-flash \
  --output /results/k2full-fp8-behavior-r4.json
python3 /acceptance/vision_acceptance.py \
  --base-url http://127.0.0.1:18080/v1 --model glm-5.3-flash \
  --template-json '{"enable_thinking":true}' --kind all \
  --output /results/k2full-fp8-vision-r4.json
