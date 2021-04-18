#!/usr/bin/env python
# coding: utf-8

# ## Here, we are training a StyleGAN2 model from scratch to compare to StyleGAN2-ADA on the same dataset

# In[1]:

import DeepFAMS
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

WORK, PROJ_DIR = DeepFAMS.utils.set_env()

# In[2]:

DATASET_NAME = 'metfaces'

# In[3]:

RAW_IMGS_DIR, RESIZED_IMGS_DIR, DATA_CUSTOM_DIR, TRAIN_RUNS_DIR = DeepFAMS.utils.return_dirs(
    PROJ_DIR, DATASET_NAME)

# In[4]:

RAW_IMGS_DIR_ = f'{PROJ_DIR}/datasets/metfaces-release/images'
RESIZED_IMGS_DIR_ = f'{PROJ_DIR}/datasets/metfaces_resized_imgs'
TRAIN_RUNS_DIR_ = f'{Path(TRAIN_RUNS_DIR).parent}/StyleGAN2_{Path(TRAIN_RUNS_DIR).name}'

# In[5]:

raw_imgs = glob(f'{RAW_IMGS_DIR_}/*')
print(len(raw_imgs))

# In[6]:

# for x in tqdm(raw_imgs):
#     DeepFAMS.preprocessing.resize_imgs(x, (256, 256), RESIZED_IMGS_DIR)

# In[7]:

print(f'Raw: {len(raw_imgs)}, Resized: {len(glob(f"{RESIZED_IMGS_DIR_}/*"))}')

# In[ ]:

# DeepFAMS.preprocessing.tf_record_exporter(
#     tfrecord_dir=DATA_CUSTOM_DIR, image_dir=RESIZED_IMGS_DIR, shuffle=1)

# In[8]:

# # Initial run to create the first checkpoint
# DeepFAMS.utils.executePopen(f'''#!/bin/bash
# module load anaconda
# module load compiler/gcc/6.1
# module load cuda/10.0
# conda activate stylegan2
# python3 {PROJ_DIR}/stylegan2/run_training.py \
#     --num-gpus=2 \
#     --data-dir=$WORK/ADA_Project/datasets \
#     --result-dir={TRAIN_RUNS_DIR_} \
#     --config=config-f \
#     --mirror-augment=true \
#     --total-kimg=1 \
#     --dataset={Path(DATA_CUSTOM_DIR).name}''', PROJ_DIR)

# In[10]:

for num in range(-1, -10, -1):
    files = DeepFAMS.utils.last_snap(num, TRAIN_RUNS_DIR_)
    if files != []:
        break

latest_snap = sorted(files)[-1]
print(latest_snap)

# In[ ]:

DeepFAMS.utils.executePopen(
    f'''#!/bin/bash
module load anaconda
module load compiler/gcc/6.1
module load cuda/10.0
conda activate stylegan2
python3 {PROJ_DIR}/stylegan2/run_training.py \
    --num-gpus=2 \
    --data-dir={PROJ_DIR}/datasets \
    --result-dir={TRAIN_RUNS_DIR_} \
    --config=config-f \
    --mirror-augment=true \
    --dataset={Path(DATA_CUSTOM_DIR).name}
    --resume-pkl={latest_snap}''', PROJ_DIR)

# In[ ]:
