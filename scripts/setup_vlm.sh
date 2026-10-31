#!/usr/bin/env bash
# =============================================================================
#  FloodScout-VLM: download the llama.cpp CPU build and the Qwen3.5-4B GGUFs.
#
#  Usage:   scripts/setup_vlm.sh            # Q4_K_M + mmproj (default)
#           scripts/setup_vlm.sh --q8       # also fetch Q8_0 (Phase 4 ablation)
#
#  Everything lands in ~/models/qwen35/ (never committed). Re-running skips
#  files that are already complete.
# =============================================================================
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen35}"
LLAMA_TAG="${LLAMA_TAG:-b12017}"          # pinned llama.cpp release
HF_REPO="https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

fetch() {  # fetch <url> <file>
  echo ">> $2"
  curl -L --fail --retry 5 -C - -o "$2" "$1"
}

# ---- llama.cpp (CPU build, tar.gz since ~b6000; the plan said .zip) --------
if [ ! -x "$MODEL_DIR/llama.cpp/llama-server" ]; then
  tarball="llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz"
  fetch "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${tarball}" "$tarball"
  mkdir -p llama.cpp
  tar -xzf "$tarball" -C llama.cpp --strip-components=1
  rm -f "$tarball"
fi

# ---- model weights ---------------------------------------------------------
fetch "$HF_REPO/Qwen3.5-4B-Q4_K_M.gguf" Qwen3.5-4B-Q4_K_M.gguf
fetch "$HF_REPO/mmproj-F16.gguf"        mmproj-F16.gguf
if [ "${1:-}" = "--q8" ]; then
  fetch "$HF_REPO/Qwen3.5-4B-Q8_0.gguf" Qwen3.5-4B-Q8_0.gguf
fi

echo "Done. llama-server: $MODEL_DIR/llama.cpp/llama-server"
ls -lh "$MODEL_DIR"
