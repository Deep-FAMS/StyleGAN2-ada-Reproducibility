#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name='backup_pkls'
#SBATCH --mem=2gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user='en4byxffy93v6uy@pipedream.net'
#SBATCH --error='/work/chaselab/malyetama/ADA_Project/jobs_log/%J.err'
#SBATCH --output='/work/chaselab/malyetama/ADA_Project/jobs_log/%J.out'

cd $WORK
module load rclone
module load cuda
module load anaconda
module load compiler/gcc/6.1
conda activate $WORK/.conda/envs/ada-env
while true; do $WORK/.conda/envs/ada-env/bin/python  python $WORK/ADA_Project/backup_pkls.py; sleep 3600; done
