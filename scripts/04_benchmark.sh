#!/usr/bin/env bash
# Phase 4 and 8: measure the real speed and memory use, and find a stable setup.
#
#   scripts/04_benchmark.sh                 test the configured model
#   scripts/04_benchmark.sh --auto-tune     step down to a smaller context or model if it is unstable
#   scripts/04_benchmark.sh --model llama3.2:3b --ctx 4096
#
# For a fair test close the browser and other heavy programs first.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
if [ -x venv/bin/python ]; then PY="venv/bin/python"; fi

exec "$PY" -m benchmarks.benchmark "$@"
