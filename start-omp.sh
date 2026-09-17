#!/usr/bin/env bash
# Launch the local agent on this handoff bundle. Any extra args are passed through
# (e.g. ./start-omp.sh -p "@HANDOFF.md Continue from section 8, item #1").
cd "$(dirname "$0")" || exit 1
MODEL="${OMP_MODEL:-deepseek-v4.1-flash}"
if ! command -v omp >/dev/null 2>&1; then
  echo "omp not found on PATH (expected /home/ser/.local/bin/omp)" >&2
  exit 1
fi
echo "==> omp --model=$MODEL  (cwd: $PWD)"
echo "==> brief: HANDOFF.md   results: repo/mosaic-gatea/RESULTS-CONFIRMING-RUN.md"
exec omp --model="$MODEL" "$@"
