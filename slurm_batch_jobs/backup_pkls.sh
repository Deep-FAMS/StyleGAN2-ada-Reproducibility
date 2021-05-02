#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --job-name='backup_pkls'
#SBATCH --constraint=opa
#SBATCH --mail-type=ALL
#SBATCH --mail-user='en4byxffy93v6uy@pipedream.net'
#SBATCH --error=/work/chaselab/malyetama/ADA_Project/jobs_log/%J.err
#SBATCH --output=/work/chaselab/malyetama/ADA_Project/jobs_log/%J.out


cd $WORK
module load rclone
module load anaconda
conda activate $WORK/.conda/envs/ada-env
while true; do python $WORK/ADA_Project/backup_pkls.py; sleep 14400; done  # every 4 hrs
