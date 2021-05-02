import DeepFAMS
import tensorflow as tf
import warnings
import sys
import os
WORK = os.environ["WORK"]
PROJ_DIR = f'{WORK}/ADA_Project'
os.chdir(WORK)
print(os.getcwd())

sys.path.insert(0, f'{WORK}/ADA_Project')
sys.path.insert(0, f'{WORK}/ADA_Project/DeepFAMS')

warnings.filterwarnings("ignore", category=FutureWarning)

tf.compat.v1.enable_eager_execution()

PROJ_DIR = f'{WORK}/ADA_Project'
RAW_IMGS_DIR = f'{PROJ_DIR}/datasets/cars_train'
RESIZED_IMGS_DIR = f'{PROJ_DIR}/datasets/cars196_resized'
DATA_CUSTOM_DIR = f'{PROJ_DIR}/datasets/cars196_custom'
TRAIN_RUNS_DIR = f'{PROJ_DIR}/training_runs/cars196_training-runs'

url = 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz'
urllib.request.urlretrieve(url, f'{PROJ_DIR}/datasets/cars_train.tgz')

tarf = tarfile.open(f'{PROJ_DIR}/datasets/cars_train.tgz')
tarf.extractall(path=RAW_IMGS_DIR)

raw_imgs = glob(f'{RAW_IMGS_DIR}/*')

for x in tqdm(raw_imgs):
    DeepFAMS.preprocessing.resize_imgs(x, (256, 256), RESIZED_IMGS_DIR)

DeepFAMS.preprocessing.tf_record_exporter(tfrecord_dir=DATA_CUSTOM_DIR,
                                          image_dir=RESIZED_IMGS_DIR,
                                          shuffle=1)

# Needs to be run through the command line at least once to compile the model

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

DeepFAMS.utils.execute('nvidia-smi')

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
