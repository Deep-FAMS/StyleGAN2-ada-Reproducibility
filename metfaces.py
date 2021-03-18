import os
from glob import glob
import numpy as np
import PIL.Image
from tqdm import tqdm
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from DeepFAMS import preprocessing, utils
from DeepFAMS import setup_training_options, RunTraining

os.chdir('/lustre/work/chaselab/malyetama/METFACES_PROJECT')

WORK = os.environ["WORK"]
PROJ = 'METFACES_PROJECT'
RAW_IMGS_DIR = f'{WORK}/metfaces-release/images/'
DS_NAME = 'metfaces'
RESIZED_IMGS_DIR = f'{WORK}/{PROJ}/resized_imgs'
DATA_CUSTOM = f'{WORK}/{PROJ}/metfaces_custom'
training_runs_dir = f'{WORK}/{PROJ}/metfaces_training-runs'


for n in range(-1, -10, -1):
    files = utils.last_snap(WORK, training_runs_dir, n)
    if files != []:
        break

latest_snap = sorted(files)[-1]
print(latest_snap)


run_desc, training_options = setup_training_options(
    gpus       = 2,
    snap       = 1,
    data       = DATA_CUSTOM,
   resume     = latest_snap
)


utils.execute('nvidia-smi')


RunTraining(outdir=training_runs_dir, seed=1000,
             dry_run=True, run_desc=run_desc, training_options=training_options)


tf.compat.v1.disable_eager_execution()
RunTraining(outdir=training_runs_dir, seed=1000,
             dry_run=False, run_desc=run_desc, training_options=training_options)
