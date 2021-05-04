import os
import json
import subprocess
import shlex
from pathlib import Path


def upload_best_snapshots():
    def execute(command: str):
        command = shlex.split(command)
        stdout = subprocess.run(command,
                                capture_output=True,
                                text=True,
                                check=True).stdout
        lines = "\n".join(list(stdout.strip().splitlines()))
        print(lines)

    WORK = os.environ['WORK']

    with open(f'{WORK}/ADA_Project/FID_of_best_snapshots.json') as f:
        d = json.load(f)

    pkls = [
        WORK + 'ADA_Project/training_runs/' +
        d[k]['file'].replace('log.txt', '') + d[k]['snapshot'] + '.pkl'
        for k, v in d.items()
    ]

    with open('best_pkls.sh', 'w') as f:
        print('#!/bin/bash \n' + '#SBATCH --nodes=1 \n' +
              '#SBATCH --partition=tmp_anvil \n' +
              '#SBATCH --ntasks-per-node=4 \n' +
              '#SBATCH --time=4:00:00 \n\n' + 'module load rclone',
              file=f)

        print(f'rclone mkdir GoogleDrive:/best_pkls', file=f)

        for x in pkls:
            ds = Path(x).parents[1].name.replace("_training-runs", "")
            file = Path(x).name
            print(f'rclone copyto {x} GoogleDrive:/best_pkls/{ds}_{file} -P\n',
                  file=f)


if __name__ == '__main__':
    upload_best_snapshots()
