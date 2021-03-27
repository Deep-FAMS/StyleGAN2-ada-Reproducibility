#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name='rclone'
#SBATCH --mem=16gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user='en4byxffy93v6uy@pipedream.net'
#SBATCH --error='/work/chaselab/malyetama/ADA_Project/jobs_log/%J.err'
#SBATCH --output='/work/chaselab/malyetama/ADA_Project/jobs_log/%J.out'

cd $WORK
module load rclone
rclone copy /work/chaselab/malyetama/ADA_Project GoogleDrive:/ADA_Project_backup
