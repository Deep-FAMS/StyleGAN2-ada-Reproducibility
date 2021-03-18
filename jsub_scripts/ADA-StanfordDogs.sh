#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name='ADA-StanfordDogs'
#SBATCH --partition='gpu'
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu_32gb&gpu_v100
#SBATCH --mem=32gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user='en4byxffy93v6uy@pipedream.net'
#SBATCH --error='/work/chaselab/malyetama/ADA_Project/jobs_log/%J.err'
#SBATCH --output='/work/chaselab/malyetama/ADA_Project/jobs_log/%J.out'

cd $WORK/ADA_Project
module load cuda
module load anaconda
module load compiler/gcc/6.1
conda activate /work/chaselab/malyetama/.conda/envs/ada-env
/work/chaselab/malyetama/.conda/envs/ada-env/bin/python /work/chaselab/malyetama/ADA_Project/py_scripts/ADA-StanfordDogs.py
