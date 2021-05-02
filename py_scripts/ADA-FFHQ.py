from glob import glob
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


def ADA_FFHQ(subset):

    WORK, PROJ_DIR = DeepFAMS.utils.set_env()

    RAW_IMGS_DIR, RESIZED_IMGS_DIR, DATA_CUSTOM_DIR, TRAIN_RUNS_DIR_ = DeepFAMS.utils.return_dirs(
        PROJ_DIR, 'FFHQ')

    if subset != '':
        subset = '_' + subset.upper()

    DATA_CUSTOM_DIR = f'{Path(DATA_CUSTOM_DIR_)}' + subset
    TRAIN_RUNS_DIR = f'{Path(TRAIN_RUNS_DIR_).parent}/{Path(TRAIN_RUNS_DIR_).name}' + subset

    DeepFAMS.utils.execute(
        '$WORK/.conda/envs/ada-env/bin/kaggle datasets download -d \
        arnaud58/flickrfaceshq-dataset-ffhq -p $WORK/ADA_Project/datasets')

    downloaded_file = f'{Path(RAW_IMGS_DIR).parent}/flickrfaceshq-dataset-ffhq.zip'

    with zipfile.ZipFile(downloaded_file) as zf:
        for member in tqdm(zf.infolist(), desc='Extracting'):
            try:
                zf.extract(member, RAW_IMGS_DIR)
            except zipfile.error as e:
                pass

    raw_imgs = glob(f'{RAW_IMGS_DIR}/*')
    print(len(raw_imgs))

    for x in tqdm(raw_imgs):
        DeepFAMS.preprocessing.resize_imgs(x, (256, 256), RESIZED_IMGS_DIR)

    print(
        f'Raw: {len(raw_imgs)}, Resized: {len(glob(f"{RESIZED_IMGS_DIR}/*"))}')

    DeepFAMS.preprocessing.tf_record_exporter(tfrecord_dir=DATA_CUSTOM_DIR,
                                              image_dir=RESIZED_IMGS_DIR,
                                              shuffle=1)

    DeepFAMS.utils.execute('nvidia-smi')

    DeepFAMS.utils.executePopen(
        f'''#!/bin/bash
    module load anaconda
    module load compiler/gcc/4.7
    module load cuda
    $WORK/.conda/envs/ada-env/bin/python $WORK/ADA_Project/StyleGAN2-ada/train.py \
    --outdir={TRAIN_RUNS_DIR} \
    --gpus=2 \
    --data={DATA_CUSTOM_DIR} \
    --snap=1 \
    --kimg=1''', PROJ_DIR)

    for num in range(-1, -10, -1):
        files = DeepFAMS.utils.last_snap(num, TRAIN_RUNS_DIR)
        if files != []:
            break

    latest_snap = sorted(files)[-1]
    print(latest_snap)

    run_desc, training_options = DeepFAMS.setup_training_options(
        gpus=2, snap=30, data=DATA_CUSTOM_DIR, resume=latest_snap)

    DeepFAMS.RunTraining(outdir=TRAIN_RUNS_DIR,
                         seed=1000,
                         dry_run=True,
                         run_desc=run_desc,
                         training_options=training_options)

    tf.compat.v1.disable_eager_execution()
    DeepFAMS.RunTraining(outdir=TRAIN_RUNS_DIR,
                         seed=1000,
                         dry_run=False,
                         run_desc=run_desc,
                         training_options=training_options)


if __name__ == "__main__":
    subset = sys.argv[1]  # one of: ['', '2K', '5K', '30K']
    ADA_FFHQ(subset)
