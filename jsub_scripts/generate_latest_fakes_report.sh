#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --job-name='fakes_report'
#SBATCH --mem=1gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user='en4byxffy93v6uy@pipedream.net'
#SBATCH --error='$WORK/ADA_Project/jobs_log/%J.err'
#SBATCH --output='$WORK/ADA_Project/jobs_log/%J.out'

cd $WORK
module load cuda
module load anaconda
module load compiler/gcc/6.1
conda activate $WORK/.conda/envs/ada-env
while true; do $WORK/.conda/envs/ada-env/bin/python $WORK/ADA_Project/generate_latest_fakes_report.py; sleep $(( 4 * 3600 )); done
