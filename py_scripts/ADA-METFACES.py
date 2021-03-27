import os
WORK = os.environ["WORK"]
os.chdir(f'{WORK}/ADA_Project')
print(os.getcwd())

import sys
sys.path.insert(0, f'{WORK}/ADA_Project')
sys.path.insert(0, f'{WORK}/ADA_Project/DeepFAMS')

from glob import glob
import numpy as np
import PIL.Image
from tqdm import tqdm
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import DeepFAMS


PROJ_DIR = f'{WORK}/ADA_Project'
RAW_IMGS_DIR = f'{PROJ_DIR}/datasets/metfaces-release/images'
RESIZED_IMGS_DIR = f'{PROJ_DIR}/datasets/metfaces_resized_imgs'
DATA_CUSTOM_DIR = f'{PROJ_DIR}/datasets/metfaces_custom'
TRAIN_RUNS_DIR = f'{PROJ_DIR}/training_runs/metfaces_training-runs'


# # data was downloaded from https://drive.google.com/open?id=1w-Os4uERBmXwCm7Oo_kW6X3Sd2YHpJMC
# raw_imgs = glob(f'{RAW_IMGS_DIR}/*')

# for x in tqdm(raw_imgs):
#     DeepFAMS.preprocessing.resize_imgs(x, (512, 512), RESIZED_IMGS_DIR)

# DeepFAMS.preprocessing.tf_record_exporter(tfrecord_dir=DATA_CUSTOM_DIR, image_dir=RESIZED_IMGS_DIR, shuffle=1)


# # Needs to be run through the command line at least once to compile the model

# ! module load anaconda && \
#     module load compiler/gcc/4.7 && \
#     module load cuda && \
#     $WORK/.conda/envs/ada-env/bin/python $WORK/ADA_Project/StyleGAN2-ada__source_code/train.py \
#     --outdir=$WORK/ADA_Project/training_runs/metfaces_training-runs \
#     --gpus=2 \
#     --data=$WORK/ADA_Project/datasets/metfaces_custom \
#     --snap=1


for num in range(-1, -10, -1):
    files = DeepFAMS.utils.last_snap(num, TRAIN_RUNS_DIR)
    if files != []:
        break

latest_snap = sorted(files)[-1]
print(latest_snap)


run_desc, training_options = DeepFAMS.setup_training_options(
    gpus       = 2,
    snap       = 1,
    data       = DATA_CUSTOM_DIR,
    resume     = latest_snap
)

DeepFAMS.utils.execute('nvidia-smi')


DeepFAMS.RunTraining(outdir=TRAIN_RUNS_DIR, seed=1000,
             dry_run=True, run_desc=run_desc, training_options=training_options)

tf.compat.v1.disable_eager_execution()
DeepFAMS.RunTraining(outdir=TRAIN_RUNS_DIR, seed=1000,
             dry_run=False, run_desc=run_desc, training_options=training_options)
