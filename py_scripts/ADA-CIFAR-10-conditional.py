from glob import glob
from tqdm import tqdm
from pathlib import Path
import urllib.request
import tarfile
import sys
import dotenv
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

dotenv.load_dotenv(override=True)
WORK = os.getenv('WORK')
sys.path.insert(0, f'{WORK}/ADA_Project')
sys.path.insert(0, f'{WORK}/ADA_Project/StyleGAN2-ADA')

import DeepFAMS

WORK, PROJ_DIR = DeepFAMS.utils.set_env()


url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
output_file = f'{PROJ_DIR}/datasets/cifar-10-python.tar.gz'
urllib.request.urlretrieve(url, output_file)

tarf = tarfile.open(output_file)
tarf.extractall(path=f'{PROJ_DIR}/datasets')

# class-conditional

DeepFAMS.utils.executePopen(
    f'$WORK/.conda/envs/ada-env/bin/python $WORK/ADA_Project/StyleGAN2-ada/dataset_tool.py \
    create_cifar10 --ignore_labels=0 \
    {PROJ_DIR}/datasets/cifar10c {PROJ_DIR}/datasets/cifar-10-batches-py', PROJ_DIR)


_, _, _, TRAIN_RUNS_DIR = DeepFAMS.utils.return_dirs(PROJ_DIR, 'CIFAR-10')

DATA_CUSTOM_DIR_c = f'{PROJ_DIR}/datasets/cifar10c'
TRAIN_RUNS_DIR_c = f'{Path(TRAIN_RUNS_DIR).parent}/conditional_{Path(TRAIN_RUNS_DIR).name}'

DeepFAMS.utils.execute('nvidia-smi')


# conditional training
DeepFAMS.utils.executePopen(f'''#!/bin/bash
module load anaconda
module load compiler/gcc/4.7
module load cuda
$WORK/.conda/envs/ada-env/bin/python $WORK/ADA_Project/StyleGAN2-ada/train.py \
--outdir={TRAIN_RUNS_DIR_c} \
--gpus=2 \
--data={DATA_CUSTOM_DIR_c} \
--snap=1 \
--kimg=1 \
--cfg=cifar''', PROJ_DIR)


for num in range(-1, -10, -1):
    files = DeepFAMS.utils.last_snap(num, TRAIN_RUNS_DIR_c)
    if files != []:
        break

latest_snap = sorted(files)[-1]
print(latest_snap)


run_desc, training_options = DeepFAMS.setup_training_options(
    gpus       = 2,
    snap       = 50,
    data       = DATA_CUSTOM_DIR_c,
    resume     = latest_snap
)


DeepFAMS.RunTraining(outdir=TRAIN_RUNS_DIR_c, seed=1000,
             dry_run=True, run_desc=run_desc, training_options=training_options)


tf.compat.v1.disable_eager_execution()
DeepFAMS.RunTraining(outdir=TRAIN_RUNS_DIR_c, seed=1000,
             dry_run=False, run_desc=run_desc, training_options=training_options)
