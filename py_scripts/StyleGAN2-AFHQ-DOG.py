# Here, we are training a StyleGAN2 model from scratch to compare to StyleGAN2-ADA on the same dataset

from glob import glob
from pathlib import Path
import sys
import dotenv
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

dotenv.load_dotenv(override=True)
WORK = os.getenv('WORK')
sys.path.insert(0, f'{WORK}/ADA_Project')
sys.path.insert(0, f'{WORK}/ADA_Project/stylegan2')
import DeepFAMS

WORK, PROJ_DIR = DeepFAMS.utils.set_env()

DATASET_NAME = 'AFHQ-DOG'

RAW_IMGS_DIR, RESIZED_IMGS_DIR, DATA_CUSTOM_DIR, TRAIN_RUNS_DIR = DeepFAMS.utils.return_dirs(
    PROJ_DIR, DATASET_NAME)

subset_folder = f'{DATASET_NAME.replace("AFHQ", "").replace("-", "").lower()}'
RAW_IMGS_DIR_ = f'{PROJ_DIR}/datasets/AFHQ_images_raw/afhq/train/{subset_folder}'
TRAIN_RUNS_DIR = f'{Path(TRAIN_RUNS_DIR).parent}/StyleGAN2_{Path(TRAIN_RUNS_DIR).name}'

raw_imgs = glob(f'{RAW_IMGS_DIR_}/*')
print(len(raw_imgs))

for x in tqdm(raw_imgs):
    DeepFAMS.preprocessing.resize_imgs(x, (256, 256), RESIZED_IMGS_DIR)

print(f'Raw: {len(raw_imgs)}, Resized: {len(glob(f"{RESIZED_IMGS_DIR}/*"))}')

DeepFAMS.preprocessing.tf_record_exporter(tfrecord_dir=DATA_CUSTOM_DIR,
                                          image_dir=RESIZED_IMGS_DIR,
                                          shuffle=1)

# Initial run to create the first checkpoint
DeepFAMS.utils.executePopen(
    f'''#!/bin/bash
module load anaconda
module load compiler/gcc/6.1
module load cuda/10.0
conda activate stylegan2
python3 {PROJ_DIR}/stylegan2/run_training.py \
    --num-gpus=2 \
    --data-dir={PROJ_DIR}/datasets \
    --result-dir={TRAIN_RUNS_DIR} \
    --config=config-f \
    --mirror-augment=true \
    --total-kimg=1 \
    --dataset={Path(DATA_CUSTOM_DIR).name}''', PROJ_DIR)

for num in range(-1, -10, -1):
    files = DeepFAMS.utils.last_snap(num, TRAIN_RUNS_DIR)
    if files != []:
        break

latest_snap = sorted(files)[-1]
print(latest_snap)

DeepFAMS.utils.executePopen(
    f'''#!/bin/bash
module load anaconda
module load compiler/gcc/6.1
module load cuda/10.0
conda activate stylegan2
/work/chaselab/malyetama/.conda/envs/stylegan2/bin/python {PROJ_DIR}/stylegan2/run_training.py \
    --num-gpus=2 \
    --data-dir={PROJ_DIR}/datasets \
    --result-dir={TRAIN_RUNS_DIR} \
    --config=config-f \
    --mirror-augment=true \
    --dataset={Path(DATA_CUSTOM_DIR).name}
    --resume-pkl={latest_snap}''', PROJ_DIR)
