#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --job-name='30K-ADA-FFHQ'
#SBATCH --partition='gpu'
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu_32gb&gpu_v100
#SBATCH --mem=64gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user='en4byxffy93v6uy@pipedream.net'
#SBATCH --error=/work/chaselab/malyetama/ADA_Project/jobs_log/%J.err
#SBATCH --output=/work/chaselab/malyetama/ADA_Project/jobs_log/%J.out

cd $WORK/ADA_Project
module load anaconda
module load compiler/gcc/6.1
module load cuda/10.0
conda activate $WORK/.conda/envs/ada-env
$WORK/.conda/envs/ada-env/bin/python $WORK/ADA_Project/py_scripts/ADA-FFHQ.py '30K'
