import DeepFAMS
import tensorflow as tf
import PIL.Image
import urllib.request
import sys
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

tf.compat.v1.enable_eager_execution()

WORK = os.getenv('WORK')
sys.path.insert(0, f'{WORK}/ADA_Project')
sys.path.insert(0, f'{WORK}/ADA_Project/StyleGAN2-ADA')


WORK, PROJ_DIR = DeepFAMS.utils.set_env()


# Dataset download link: https://unomaha.box.com/shared/static/emdbt8qdkq6p5841o1dbcwe01zlzn7p2.zip

RAW_IMGS_DIR, RESIZED_IMGS_DIR, DATA_CUSTOM_DIR, TRAIN_RUNS_DIR = DeepFAMS.utils.return_dirs(
    PROJ_DIR, 'ANIME-FACES')


# RAW_IMGS_DIR_ = glob(f'{RAW_IMGS_DIR}/waifus/images/*')
# RAW_IMGS_DIR_ = [x for x in RAW_IMGS_DIR_ if Path(x).suffix in ['.jpg', '.png', '.jpeg']]
# print(len(RAW_IMGS_DIR_))


# def select_imgs(img):
#     image = PIL.Image.open(img)
#     if image.size[0] >= 256 and image.size[1] >= 256:
#         return img

# num_cores = multiprocessing.cpu_count() - 1
# results = Parallel(n_jobs=num_cores)(delayed(select_imgs)(img) for img in RAW_IMGS_DIR_)

# fraw_imgs = [x for x in results if x is not None]
# print(len(fraw_imgs))

# for x in tqdm(fraw_imgs):
#     DeepFAMS.preprocessing.resize_imgs(x, (256, 256), RESIZED_IMGS_DIR)

# print(f'Raw: {len(fraw_imgs)}, Resized: {len(glob(f"{RESIZED_IMGS_DIR}/*"))}')


# DeepFAMS.preprocessing.tf_record_exporter(tfrecord_dir=DATA_CUSTOM_DIR, image_dir=RESIZED_IMGS_DIR, shuffle=1)

DeepFAMS.utils.execute('nvidia-smi')

# DeepFAMS.utils.executePopen(f'''#!/bin/bash
# module load anaconda
# module load compiler/gcc/4.7
# module load cuda
# $WORK/.conda/envs/ada-env/bin/python $WORK/ADA_Project/StyleGAN2-ada/train.py \
# --outdir={TRAIN_RUNS_DIR} \
# --gpus=1 \
# --data={DATA_CUSTOM_DIR} \
# --snap=1 \
# --kimg=1''', PROJ_DIR)

for num in range(-1, -10, -1):
    files = DeepFAMS.utils.last_snap(num, TRAIN_RUNS_DIR)
    if files != []:
        break

latest_snap = sorted(files)[-1]
print(latest_snap)


run_desc, training_options = DeepFAMS.setup_training_options(
    gpus=2,
    snap=1,
    data=DATA_CUSTOM_DIR,
    resume=latest_snap
)


DeepFAMS.RunTraining(outdir=TRAIN_RUNS_DIR, seed=1000,
                     dry_run=True, run_desc=run_desc, training_options=training_options)


tf.compat.v1.disable_eager_execution()
DeepFAMS.RunTraining(outdir=TRAIN_RUNS_DIR, seed=1000,
                     dry_run=False, run_desc=run_desc, training_options=training_options)
