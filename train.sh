#!/usr/bin/env bash
set -euo pipefail

GPU_NUM="${GPU_NUM:-4}"
NUM_NODES="${NUM_NODES:-1}"

python -m src.main \
  +experiment=dl3dv \
  trainer.devices="$GPU_NUM" \
  trainer.num_nodes="$NUM_NODES" \
  +hydra.job.config.store_config=false
