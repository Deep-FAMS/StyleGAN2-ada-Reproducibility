#!/bin/bash
module load anaconda
module load compiler/gcc/4.7
module load cuda
$WORK/.conda/envs/ada-env/bin/python $WORK/ADA_Project/StyleGAN2-ada__source_code/train.py --outdir=/work/chaselab/malyetama/ADA_Project/training_runs/AFHQ_training-runs --gpus=2 --data=/work/chaselab/malyetama/ADA_Project/datasets/AFHQ_custom --snap=1 --kimg=1