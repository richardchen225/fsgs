#!/bin/sh
# run in current working directory
#$ -cwd

#$ -l node_o=1
# maximum run time
#$ -l h_rt=24:00:00
#$ -N download

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"

conda activate fsgs

bash /home/9/ug04729/tanyixin/fsgs/scripts/download_dl3dv.sh /gs/bs/tga-mdl/tanyixin-mdl/dataset/dl3dv
