from glob import glob
import PIL.Image
from pathlib import Path
import urllib.request
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
import DeepFAMS

WORK, PROJ_DIR = DeepFAMS.utils.set_env()

RAW_IMGS_DIR, RESIZED_IMGS_DIR, DATA_CUSTOM_DIR, TRAIN_RUNS_DIR = DeepFAMS.utils.return_dirs(
    PROJ_DIR, 'POKEMON')

url = 'https://unomaha.box.com/shared/static/g5l1kfbgmaj2v7rfazp0q6g8nfepqigp.zip'
output_file = f'{Path(RAW_IMGS_DIR).parents[0]}/POKEMON.zip'
urllib.request.urlretrieve(url, output_file)

with zipfile.ZipFile(output_file, 'r') as zip_ref:
    zip_ref.extractall(RAW_IMGS_DIR)

raw_imgs = glob(f'{RAW_IMGS_DIR}/**/*', recursive=True)
print(len(raw_imgs))

raw_imgs_ = [
    x for x in raw_imgs if Path(x).suffix in ['.jpg', '.jpeg', '.png']
]
len(raw_imgs_)

Path(f'{RAW_IMGS_DIR}/renamed').mkdir(exist_ok=True)

for n, x in enumerate(raw_imgs_):
    os.rename(x, f'{RAW_IMGS_DIR}/renamed/{n}.jpg')

RAWimgs = glob(f'{RAW_IMGS_DIR}/renamed/*')
print(len(RAWimgs))

for x in tqdm(RAWimgs):
    DeepFAMS.preprocessing.resize_imgs(x, (256, 256), RESIZED_IMGS_DIR)

print(
    f'Raw: {len(raw_imgs_)}, Resized: {len(glob(f"{RESIZED_IMGS_DIR}/**/*", recursive=True))}'
)

DeepFAMS.preprocessing.tf_record_exporter(
    tfrecord_dir=DATA_CUSTOM_DIR, image_dir=RESIZED_IMGS_DIR, shuffle=1)

DeepFAMS.utils.executePopen(
f'''#!/bin/bash
module load anaconda
module load compiler/gcc/6.1
module load cuda
$WORK/.conda/envs/ada-env/bin/python $WORK/ADA_Project/StyleGAN2-ada/train.py \
--outdir={TRAIN_RUNS_DIR} \
--gpus=1 \
--data={DATA_CUSTOM_DIR} \
--snap=1 \
--kimg=1''', PROJ_DIR
)

for num in range(-1, -10, -1):
    files = DeepFAMS.utils.last_snap(num, TRAIN_RUNS_DIR)
    if files != []:
        break

latest_snap = sorted(files)[-1]
print(latest_snap)

DeepFAMS.utils.execute('nvidia-smi')

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
