# Here, we are training a StyleGAN2 model from scratch to compare to StyleGAN2-ADA on the same dataset
from glob import glob
from tqdm import tqdm
from pathlib import Path
import zipfile
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


def StyleGAN2_FFHQ(subset):

    WORK, PROJ_DIR = DeepFAMS.utils.set_env()

    RAW_IMGS_DIR, RESIZED_IMGS_DIR, DATA_CUSTOM_DIR_, TRAIN_RUNS_DIR_ = DeepFAMS.utils.return_dirs(
        PROJ_DIR, 'FFHQ')

    if subset != '':
        subset = '_' + subset.upper()

    DATA_CUSTOM_DIR = f'{Path(DATA_CUSTOM_DIR_)}' + subset
    TRAIN_RUNS_DIR = f'{Path(TRAIN_RUNS_DIR_).parent}/StyleGAN2_{Path(TRAIN_RUNS_DIR_).name}' + subset

    # ! $WORK/.conda/envs/ada-env/bin/kaggle datasets download -d \
    #     arnaud58/flickrfaceshq-dataset-ffhq -p $WORK/ADA_Project/datasets

    # downloaded_file = f'{Path(RAW_IMGS_DIR).parent}/flickrfaceshq-dataset-ffhq.zip'

    # with zipfile.ZipFile(downloaded_file) as zf:    
    #     for member in tqdm(zf.infolist(), desc='Extracting'):
    #         try:
    #             zf.extract(member, RAW_IMGS_DIR)
    #         except zipfile.error as e:
    #             pass

    raw_imgs = glob(f'{RAW_IMGS_DIR}/*')
    print(len(raw_imgs))

    # for x in tqdm(raw_imgs):
    #     DeepFAMS.preprocessing.resize_imgs(x, (256, 256), RESIZED_IMGS_DIR)


    print(f'Raw: {len(raw_imgs)}, Resized: {len(glob(f"{RESIZED_IMGS_DIR}/*"))}')


    # DeepFAMS.preprocessing.tf_record_exporter(
    #     tfrecord_dir=DATA_CUSTOM_DIR, image_dir=RESIZED_IMGS_DIR, shuffle=1)


    DeepFAMS.utils.executePopen(f'''#!/bin/bash
    module load anaconda
    module load compiler/gcc/6.1
    module load cuda/10.0
    $WORK/.conda/envs/stylegan2/bin/python3 $WORK/ADA_Project/stylegan2/run_training.py \
        --num-gpus=2 \
        --data-dir=$WORK/ADA_Project/datasets \
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


    DeepFAMS.utils.executePopen(f'''#!/bin/bash
    module load anaconda
    module load compiler/gcc/6.1
    module load cuda/10.0
    $WORK/.conda/envs/stylegan2/bin/python3 $WORK/ADA_Project/stylegan2/run_training.py \
        --num-gpus=2 \
        --data-dir=$WORK/ADA_Project/datasets \
        --result-dir={TRAIN_RUNS_DIR} \
        --config=config-f \
        --mirror-augment=true \
        --dataset={Path(DATA_CUSTOM_DIR).name}
        --resume-pkl={latest_snap}''', PROJ_DIR)


if __name__ == "__main__":
    subset = sys.argv[1]  # one of: ['', '2K', '5K', '30K']
    StyleGAN2_FFHQ(subset)
