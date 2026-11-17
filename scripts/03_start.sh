#!/usr/bin/env bash
# Start the web app. It listens on localhost only.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
if [ -x venv/bin/python ]; then PY="venv/bin/python"; fi

if ! curl -fsS http://127.0.0.1:11434/ >/dev/null 2>&1; then
  echo "The Ollama server is not running. Start it with: systemctl start ollama   (or: ollama serve)"
  exit 1
fi

PORT="$("$PY" -c 'from config.settings import load_settings; print(load_settings().ui.port)')"
echo "Opening http://127.0.0.1:${PORT}"
exec "$PY" -m app.server
