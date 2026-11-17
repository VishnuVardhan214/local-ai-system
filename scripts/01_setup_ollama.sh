#!/usr/bin/env bash
# Phase 3: make sure Ollama is installed and tuned for 16 GB RAM and a 4 GB GPU.
# It does not reinstall Ollama if it is already there.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
if [ -x venv/bin/python ]; then PY="venv/bin/python"; fi

echo "Step 1: current ollama"
if command -v ollama >/dev/null 2>&1; then
  echo "Ollama is already installed: $(ollama --version 2>&1 | tail -1)"
else
  echo "Ollama is not installed. Installing with the official script (needs internet and sudo)."
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo
echo "Step 2: apply the server settings from config/config.yaml"
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files 2>/dev/null | grep -q '^ollama.service'; then
  sudo "$PY" scripts/ollama_env.py --systemd
  sleep 3
  systemctl show ollama -p Environment || true
else
  echo "No ollama systemd service found. Start the server yourself with these settings:"
  "$PY" scripts/ollama_env.py --print
  echo "Paste those lines into your shell, then run: ollama serve"
fi

echo
echo "Step 3: check the server answers on localhost"
if curl -fsS http://127.0.0.1:11434/ >/dev/null; then
  echo "Ollama is running."
else
  echo "Ollama is not answering yet. Check: systemctl status ollama"
  exit 1
fi

echo
echo "Done. Next: scripts/02_pull_models.sh"
