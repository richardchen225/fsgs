#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/train_then_bench.sh TRAIN_ROOT BENCH_ROOT [Hydra training overrides...]" >&2
  exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
TRAIN_ROOT="$1"
BENCH_ROOT="$2"
shift 2
GPU_NUM="${GPU_NUM:-4}"
TEST_GPU_NUM="${TEST_GPU_NUM:-$GPU_NUM}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
EXPECTED_SCENES="${EXPECTED_SCENES:-140}"
RUN_DIR="${RUN_DIR:-output/train_bench_$(date +%Y%m%d_%H%M%S)_$$}"
for value in "$GPU_NUM" "$TEST_GPU_NUM" "$TRAIN_STEPS" "$EXPECTED_SCENES"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU counts, TRAIN_STEPS, and EXPECTED_SCENES must be positive integers" >&2
    exit 2
  fi
done
TRAIN_ROOT="$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$TRAIN_ROOT")"
BENCH_ROOT="$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$BENCH_ROOT")"
RUN_DIR="$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$RUN_DIR")"
for index in train_index.json test_index.json; do
  if [[ ! -f "$TRAIN_ROOT/$index" ]]; then
    echo "Training/validation index missing: $TRAIN_ROOT/$index" >&2
    exit 2
  fi
done
"$PYTHON" scripts/benchmark_workflow.py preflight "$BENCH_ROOT" --expected-scenes "$EXPECTED_SCENES"
# A fresh run directory prevents accidentally testing an old final.ckpt/result.
mkdir -p -- "$(dirname -- "$RUN_DIR")"
mkdir -- "$RUN_DIR"
trap 'echo "Pipeline failed at line $LINENO; logs are under $RUN_DIR" >&2' ERR
echo "Experiment directory: $RUN_DIR"

"$PYTHON" -u -m src.main +experiment=dl3dv wandb.mode=offline "$@" \
  mode=train \
  "dataset.dl3dv.roots=[\"$TRAIN_ROOT\"]" \
  trainer.devices="$GPU_NUM" trainer.num_nodes=1 trainer.max_steps="$TRAIN_STEPS" \
  checkpointing.load=null checkpointing.save_final=true \
  "hydra.run.dir=$RUN_DIR/train" hydra.output_subdir=.hydra \
  2>&1 | tee "$RUN_DIR/train.log"

COMPARE_ARGS=()
if [[ "${SAVE_COMPARE:-0}" == "1" ]]; then
  COMPARE_ARGS+=(--save-compare)
fi
"$PYTHON" scripts/benchmark_workflow.py prepare-test \
  "$RUN_DIR/train" "$BENCH_ROOT" "$RUN_DIR/bench_dl3dv140" \
  "$RUN_DIR/bench_config/config.yaml" \
  --devices "$TEST_GPU_NUM" --expected-scenes "$EXPECTED_SCENES" "${COMPARE_ARGS[@]}"

# Reuse the exact resolved training configuration, including all GIR switches.
"$PYTHON" -u -m src.main --config-path "$RUN_DIR/bench_config" --config-name config \
  "hydra.run.dir=$RUN_DIR/bench_run" \
  2>&1 | tee "$RUN_DIR/test.log"

"$PYTHON" scripts/benchmark_workflow.py show-result \
  "$RUN_DIR/bench_dl3dv140/metrics.json" --expected-scenes "$EXPECTED_SCENES" \
  | tee "$RUN_DIR/result.txt"
