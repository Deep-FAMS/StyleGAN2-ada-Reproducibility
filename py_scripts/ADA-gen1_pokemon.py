#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
WORK = os.environ["WORK"]
PROJ_DIR = f'{WORK}/ADA_Project'
os.chdir(PROJ_DIR)
print(os.getcwd())

import sys
sys.path.insert(0, f'{WORK}/ADA_Project')
sys.path.insert(0, f'{WORK}/ADA_Project/DeepFAMS')


# In[2]:


from glob import glob
import numpy as np
import PIL.Image
from tqdm import tqdm
from pathlib import Path
import urllib.request
import tarfile
import zipfile
import subprocess

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import DeepFAMS


# In[3]:


RAW_IMGS_DIR, RESIZED_IMGS_DIR, DATA_CUSTOM_DIR, TRAIN_RUNS_DIR = DeepFAMS.utils.return_dirs(
    PROJ_DIR, 'gen1_pokemon')


# In[4]:


# ! wget 'https://unomaha.box.com/shared/static/b2quhszs8d7qfh40nnt2sojcztoxmdop.zip' -O datasets/gen1_pokemon.zip


# In[5]:


# import zipfile
# with zipfile.ZipFile('datasets/gen1_pokemon.zip', 'r') as zip_ref:
#     zip_ref.extractall(RAW_IMGS_DIR)


# In[6]:


raw_imgs = glob(f'{RAW_IMGS_DIR}/**/**/*')
print(len(raw_imgs))


# In[7]:


# for x in tqdm(raw_imgs):
#     DeepFAMS.preprocessing.resize_imgs(x, (256, 256), RESIZED_IMGS_DIR)


# In[8]:


print(f'Raw: {len(raw_imgs)}, Resized: {len(glob(f"{RESIZED_IMGS_DIR}/*"))}')


# In[9]:


# DeepFAMS.preprocessing.tf_record_exporter(
#     tfrecord_dir=DATA_CUSTOM_DIR, image_dir=RESIZED_IMGS_DIR, shuffle=1)


# In[10]:


# DeepFAMS.utils.executePopen(
# f'''#!/bin/bash
# module load anaconda
# module load compiler/gcc/4.7
# module load cuda
# $WORK/.conda/envs/ada-env/bin/python $WORK/ADA_Project/StyleGAN2-ada__source_code/train.py \
# --outdir={TRAIN_RUNS_DIR} \
# --gpus=2 \
# --data={DATA_CUSTOM_DIR} \
# --snap=1 \
# --kimg=1''', PROJ_DIR
# )


# In[11]:


for num in range(-1, -10, -1):
    files = DeepFAMS.utils.last_snap(num, TRAIN_RUNS_DIR)
    if files != []:
        break

latest_snap = sorted(files)[-1]
print(latest_snap)


# In[ ]:


DeepFAMS.utils.execute('nvidia-smi')


# In[ ]:


run_desc, training_options = DeepFAMS.setup_training_options(
    gpus       = 2,
    snap       = 1,
    data       = DATA_CUSTOM_DIR,
    resume     = latest_snap
)


# In[ ]:


DeepFAMS.RunTraining(outdir=TRAIN_RUNS_DIR, seed=1000,
             dry_run=True, run_desc=run_desc, training_options=training_options)


# In[ ]:


tf.compat.v1.disable_eager_execution()
DeepFAMS.RunTraining(outdir=TRAIN_RUNS_DIR, seed=1000,
             dry_run=False, run_desc=run_desc, training_options=training_options)

