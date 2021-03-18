#!/bin/bash

$WORK/.conda/envs/ada-env/bin/python3 -c "from pathlib import Path
import os

WORK = os.environ['WORK']
os.chdir(f'{WORK}/ADA_Project/jsub_scripts')
cwd = os.getcwd()

time = int(input('time (in hrs)? '))
partition = input('partition (default: gpu)? ')
constraint = input('GPU name (k20, k40, p100, v100)? ')


if constraint.lower() == 'k20':
        no_gpus = int(input('number of GPUs (MAX: 2 or 3)? '))
        const_line = '#SBATCH --constraint=' + 'gpu_' + constraint

elif constraint.lower() == 'k40':
        no_gpus = int(input('number of GPUs (MAX: 2 or 4)? '))
        const_line = '#SBATCH --constraint=' + 'gpu_' + constraint

elif constraint.lower() == 'p100':
        no_gpus = int(input('number of GPUs (MAX: 2)? '))
        mem = int(input('memory per GPU (MAX: 12)? '))
        const_line = '#SBATCH --constraint=' + 'gpu_' + str(mem) + 'gb&gpu_' + constraint

elif constraint.lower() == 'v100':
        no_gpus = int(input('number of GPUs (MAX: 2 or 4)? '))
        if no_gpus == 2:
                mem = int(input('memory per GPU (MAX: 32)? '))
                const_line = '#SBATCH --constraint=' + 'gpu_' + str(mem) + 'gb&gpu_' + constraint
        elif no_gpus == 4:
                mem = int(input('memory per GPU (MAX: 16)? '))
                const_line = '#SBATCH --constraint=' + 'gpu_' + str(mem) + 'gb&gpu_' + constraint

file_name = input('python file name (WITHOUT EXTENSION!)? ')

with open(f'{Path(file_name).stem}.sh', 'w') as f:
        f.write(f'''#!/bin/bash
#SBATCH --time={time}:00:00
#SBATCH --job-name='{file_name}'
#SBATCH --partition='{partition}'
#SBATCH --gres=gpu:{no_gpus}
{const_line}
#SBATCH --mail-type=ALL
#SBATCH --mail-user='en4byxffy93v6uy@pipedream.net'
#SBATCH --error='{WORK}/ADA_Project/jobs_log/%J.err'
#SBATCH --output='{WORK}/ADA_Project/jobs_log/%J.out'

module load cuda
module load anaconda
module load compiler/gcc/6.1
conda activate $WORK/.conda/envs/ada-env
$WORK/.conda/envs/ada-env/bin/python $WORK/ADA_Project/py_scripts/{file_name}.py
''')

print(
'=' * 60,
'\n',
'Run this command to submit the job:\n',
f'$ sbatch {cwd}/{file_name}.sh')
print('=' * 60)
"

