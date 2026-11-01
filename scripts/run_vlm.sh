#!/usr/bin/env bash
# =============================================================================
#  FloodScout: start llama-server (Qwen3.5-4B + vision projector).
#
#  Usage:   scripts/run_vlm.sh            # Q4_K_M
#           QUANT=Q8_0 scripts/run_vlm.sh # ablation
#
#  Thinking is switched per request with
#     "chat_template_kwargs": {"enable_thinking": true|false}
#  (see src/flood_vlm/NOTES.md for what was verified).
# =============================================================================
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen35}"
QUANT="${QUANT:-Q4_K_M}"
PORT="${PORT:-8080}"
THREADS="${THREADS:-$(( $(nproc) / 2 ))}"

BIN="$MODEL_DIR/llama.cpp/llama-server"
command -v llama-server >/dev/null 2>&1 && [ ! -x "$BIN" ] && BIN="$(command -v llama-server)"

export LD_LIBRARY_PATH="$MODEL_DIR/llama.cpp:${LD_LIBRARY_PATH:-}"
exec "$BIN" \
  -m "$MODEL_DIR/Qwen3.5-4B-${QUANT}.gguf" \
  --mmproj "$MODEL_DIR/mmproj-F16.gguf" \
  -c 8192 -t "$THREADS" --host 127.0.0.1 --port "$PORT" --jinja \
  --no-warmup "$@"
