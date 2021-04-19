from glob import glob
from pathlib import Path
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

RAW_IMGS_DIR, RESIZED_IMGS_DIR, DATA_CUSTOM_DIR, TRAIN_RUNS_DIR = DeepFAMS.utils.return_dirs(
    PROJ_DIR, 'FFHQ')

TRAIN_RUNS_DIR_30K = f'{Path(TRAIN_RUNS_DIR)}_30K'
DATA_CUSTOM_DIR_30K = f'{Path(DATA_CUSTOM_DIR)}_30K'

raw_imgs = glob(f'{RAW_IMGS_DIR}/*')
print(len(raw_imgs))
print(f'Raw: {len(raw_imgs)}, Resized: {len(glob(f"{RESIZED_IMGS_DIR}/*"))}')

DeepFAMS.preprocessing.tf_record_exporter(tfrecord_dir=DATA_CUSTOM_DIR_30K,
                                          image_dir=RESIZED_IMGS_DIR,
                                          shuffle=1,
                                          subset=30000)

# DeepFAMS.utils.executePopen(f'''#!/bin/bash
# module load anaconda
# module load compiler/gcc/4.7
# module load cuda
# $WORK/.conda/envs/ada-env/bin/python $WORK/ADA_Project/StyleGAN2-ada/train.py \
# --outdir={TRAIN_RUNS_DIR_30K} \
# --gpus=2 \
# --data={DATA_CUSTOM_DIR_30K} \
# --snap=1 \
# --kimg=1''', PROJ_DIR)

for num in range(-1, -10, -1):
    files = DeepFAMS.utils.last_snap(num, TRAIN_RUNS_DIR_30K)
    if files != []:
        break

latest_snap = sorted(files)[-1]
print(latest_snap)

run_desc, training_options = DeepFAMS.setup_training_options(
    gpus=2, snap=30, data=DATA_CUSTOM_DIR_30K, resume=latest_snap)

DeepFAMS.RunTraining(outdir=TRAIN_RUNS_DIR_30K,
                     seed=1000,
                     dry_run=True,
                     run_desc=run_desc,
                     training_options=training_options)

tf.compat.v1.disable_eager_execution()
DeepFAMS.RunTraining(outdir=TRAIN_RUNS_DIR_30K,
                     seed=1000,
                     dry_run=False,
                     run_desc=run_desc,
                     training_options=training_options)
