#!/bin/sh
# run in current working directory
#$ -cwd

#$ -l node_f=1
# maximum run time
#$ -l h_rt=24:00:00
#$ -N fsgs

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

conda activate fsgs

export WANDB_DIR=/home/9/ug04729/tanyixin/fsgs

python -m src.main +experiment=dl3dv wandb.mode=offline \
  'hydra.run.dir=/gs/bs/tga-mdl/tanyixin-mdl/models/fsgs/outputs/exp_dl3dv/${now:%Y-%m-%d_%H-%M-%S}' \
  'checkpointing.load=/gs/bs/tga-mdl/tanyixin-mdl/models/fsgs/outputs/exp_dl3dv/2026-06-19_23-06-19/checkpoints/epoch_16-step_20500.ckpt'

