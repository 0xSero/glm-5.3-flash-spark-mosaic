#!/usr/bin/env bash
set -euo pipefail
export GLM53_MODEL_ROOT=/home/valentine/glm53-single-spark-release-20260911/k2-massmax-k256
export GLM53_NATIVE_SOURCE=/home/valentine/flash-experimental-staging-20260906/model
export GLM53_MTP_ROOT=/home/valentine/glm53-single-spark-release-20260911/native-mtp-k2-control
export GLM53_IMAGE=sha256:cd6ce6e10276f856d9ec617c702362ffc45114cad6f26ddcb6a39ef5a2ac7e37
export GLM53_MEMORY_FRACTION=0.93
export GLM53_MTP_EXPERT_FP8=0
exec bash /home/valentine/glm53-single-spark-release-20260911/native-mtp/rollback-r3-native/start-native-mtp-candidate.sh
