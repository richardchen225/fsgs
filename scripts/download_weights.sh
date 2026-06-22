#!/usr/bin/env bash
set -euo pipefail

WEIGHTS_DIR="${1:-weights}"

mkdir -p "$WEIGHTS_DIR"

download() {
  local output="$1"
  local url="$2"

  if [[ -z "$url" || "$url" == TODO:* ]]; then
    echo "Skip $output: fill its URL in scripts/download_weights.sh first."
    return 0
  fi

  if [[ -f "$WEIGHTS_DIR/$output" ]]; then
    echo "Skip $output: already exists."
    return 0
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -O "$WEIGHTS_DIR/$output" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$WEIGHTS_DIR/$output"
  else
    echo "wget or curl is required to download weights." >&2
    exit 1
  fi
}

# Replace TODO values with the released checkpoint URLs you want to use.
download "pre_wm.safetensors" "https://huggingface.co/tencent/HunyuanWorld-Mirror/resolve/main/model.safetensors"
download "pre_zipmap.pt" "https://huggingface.co/coast01/ZipMap/resolve/main/checkpoint_online.pt"
download "model.pt" "https://huggingface.co/Ruicheng/moge-2-vitl/resolve/main/model.pt"
download "pre_dav3.safetensors" "https://huggingface.co/depth-anything/DA3-LARGE/resolve/main/model.safetensors"
echo "Weights are under $WEIGHTS_DIR"
