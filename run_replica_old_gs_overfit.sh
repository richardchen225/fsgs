#!/usr/bin/env bash
set -euo pipefail

# PyTorch reads gsplat's CUDA/C++ sources before JIT compilation. Some cluster
# locales default Python's file encoding to ASCII, while those sources contain
# UTF-8 punctuation.
export PYTHONUTF8=1

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_replica_old_gs_overfit.sh DATA_DIR [HEAD_CHECKPOINT] [HYDRA_OVERRIDES...]" >&2
  exit 2
fi

DATA_DIR="$(cd "$1" && pwd)"
export REPLICA_TOY_ROOT="$DATA_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
shift

ARGS=(
  -m src.experiments.overfit_old_gs_residual
  +experiment=replica_old_gs_overfit
)

if [[ $# -gt 0 && "$1" != *=* ]]; then
  CHECKPOINT_DIR="$(cd "$(dirname "$1")" && pwd)"
  CHECKPOINT="$CHECKPOINT_DIR/$(basename "$1")"
  ARGS+=("toy.checkpoint=$CHECKPOINT")
  shift
fi

ARGS+=("$@")

python "${ARGS[@]}"
