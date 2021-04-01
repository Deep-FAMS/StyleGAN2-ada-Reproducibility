#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name='ADA-ANIME-FACES'
#SBATCH --partition='gpu'
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu_32gb&gpu_v100
#SBATCH --mem=64gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user='malyetama@unomaha.edu'
#SBATCH --error=/work/chaselab/malyetama/ADA_Project/jobs_log/%J.err
#SBATCH --output=/work/chaselab/malyetama/ADA_Project/jobs_log/%J.out

cd $WORK/ADA_Project
module load cuda
module load anaconda
module load compiler/gcc/6.1
conda activate $WORK/.conda/envs/ada-env
$WORK/.conda/envs/ada-env/bin/python $WORK/ADA_Project/py_scripts/ADA-ANIME-FACES.py