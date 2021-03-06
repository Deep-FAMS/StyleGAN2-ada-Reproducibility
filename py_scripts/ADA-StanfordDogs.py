import DeepFAMS
import tensorflow as tf
import warnings
from glob import glob
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

RAW_IMGS_DIR, RESIZED_IMGS_DIR, DATA_CUSTOM_DIR, TRAIN_RUNS_DIR = DeepFAMS.utils.return_dirs(
    PROJ_DIR, 'StanfordDogs')

url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'

DeepFAMS.utils.Get_Raw_Data(url=url,
             datasets_dir=f'{PROJ_DIR}/datasets',
             RAW_IMGS_DIR=RAW_IMGS_DIR,
             file_name='StanfordDogs_images.tar')

raw_imgs = glob(f'{RAW_IMGS_DIR}/**/**/**/*')
print(len(raw_imgs))

for x in tqdm(raw_imgs):
    DeepFAMS.preprocessing.resize_imgs(x, (256, 256), RESIZED_IMGS_DIR)

print(f'Raw: {len(raw_imgs)}, Resized: {len(glob(f"{RESIZED_IMGS_DIR}/*"))}')

DeepFAMS.preprocessing.tf_record_exporter(tfrecord_dir=DATA_CUSTOM_DIR, image_dir=RESIZED_IMGS_DIR, shuffle=1)

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
--kimg=1''', PROJ_DIR
)

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
