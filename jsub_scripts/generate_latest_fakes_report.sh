#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name='fakes_report'
#SBATCH --mem=4gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user='en4byxffy93v6uy@pipedream.net'
#SBATCH --error='/work/chaselab/malyetama/ADA_Project/jobs_log/%J.err'
#SBATCH --output='/work/chaselab/malyetama/ADA_Project/jobs_log/%J.out'

cd $WORK
module load cuda
module load anaconda
module load compiler/gcc/6.1
conda activate /work/chaselab/malyetama/.conda/envs/ada-env
while true; do /work/chaselab/malyetama/.conda/envs/ada-env/bin/python /work/chaselab/malyetama/ADA_Project/generate_latest_fakes_report.py; sleep $(( 8 * 3600 )); done
