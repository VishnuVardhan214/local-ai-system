#!/usr/bin/env bash
# Phase 2 and 3: download the models. Run this once while online. After that the system works offline.
#
#   scripts/02_pull_models.sh            main model, small fallback and the embedding model
#   scripts/02_pull_models.sh --compare  also pull the other candidates so they can be benchmarked
set -euo pipefail

MAIN="${LAI_BACKEND__MODEL:-qwen3:4b}"
EMBED="${LAI_BACKEND__EMBED_MODEL:-nomic-embed-text}"
FALLBACKS=("llama3.2:3b")
COMPARE=("qwen3:8b" "qwen3:1.7b")

pull() {
  if ollama list | awk 'NR>1 {print $1}' | grep -qx "$1"; then
    echo "already installed: $1"
  else
    echo "pulling $1"
    ollama pull "$1"
  fi
}

pull "$MAIN"
pull "$EMBED"
for m in "${FALLBACKS[@]}"; do pull "$m"; done
if [ "${1:-}" = "--compare" ]; then
  for m in "${COMPARE[@]}"; do pull "$m"; done
fi

echo
ollama list
echo
echo "Note: qwen3:4b is listed as 2.5 GB on the Ollama library page at the time of writing."
echo "Check the SIZE column above for the real size on your disk."
