#!/usr/bin/env bash
# Optional: run the same GGUF model directly with llama.cpp for comparison.
#
# The web app talks to Ollama, not to llama-server, so this script is for testing:
# it gives you llama.cpp's own web page on localhost and lets you compare speed and memory.
#
#   MODEL_GGUF=/path/to/Qwen3-4B-Q4_K_M.gguf scripts/run_llamacpp.sh
#
# All values can be changed from the environment, for example NGL=24 CTX=2048.
set -euo pipefail

: "${MODEL_GGUF:?Set MODEL_GGUF to the path of a .gguf file}"
SERVER="${LLAMA_SERVER:-llama-server}"
NGL="${NGL:-99}"            # layers on the GPU. Lower it if the GPU runs out of memory. 99 means all.
CTX="${CTX:-4096}"          # context window
THREADS="${THREADS:-6}"     # CPU threads
KV="${KV:-q8_0}"            # KV cache type: f16, q8_0 or q4_0
PORT="${PORT:-8081}"

# mmap: the model file is mapped from the SSD and pages are loaded as they are used, so the
# first run is slower and the operating system can drop pages again under memory pressure.
# Add --no-mmap to read the whole file into RAM instead, or --mlock to pin it in RAM.
EXTRA=()
[ "${NO_MMAP:-0}" = "1" ] && EXTRA+=(--no-mmap)
[ "${MLOCK:-0}" = "1" ] && EXTRA+=(--mlock)

exec "$SERVER" \
  -m "$MODEL_GGUF" \
  --host 127.0.0.1 --port "$PORT" \
  -ngl "$NGL" \
  -c "$CTX" \
  -t "$THREADS" \
  -np 1 \
  --flash-attn on \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  ${EXTRA[@]+"${EXTRA[@]}"}
