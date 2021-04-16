#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --job-name='backup_pkls'
#SBATCH --mem=1gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user='en4byxffy93v6uy@pipedream.net'
#SBATCH --error=/work/chaselab/malyetama/ADA_Project/jobs_log/%J.err
#SBATCH --output=/work/chaselab/malyetama/ADA_Project/jobs_log/%J.out

cd $WORK
module load rclone
module load anaconda
conda activate $WORK/.conda/envs/tensorflow-env2
while true; do python $WORK/ADA_Project/backup_pkls.py; sleep 3600; done
