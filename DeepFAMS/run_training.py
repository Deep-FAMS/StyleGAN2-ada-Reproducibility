from metrics import metric_defaults
from training import dataset
from training import training_loop
import dnnlib.tflib as tflib
import dnnlib
import tensorflow as tf
import re
import json
import argparse
import os
import sys

WORK = os.environ["WORK"]
sys.path.insert(0, f'{WORK}/ADA_Project/StyleGAN2-ada')


"""
The `RunTraining()` function is forked from https://github.com/NVlabs/stylegan2-ada,
    and edited to work on Crane.

    "3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under this License, (b) you include a complete copy of this License with your distribution, and (c) you retain without modification any copyright, patent, trademark, or attribution notices that are present in the Work.""
    – https://nvlabs.github.io/stylegan2-ada/license.html
"""

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


def RunTraining(outdir, seed, dry_run, run_desc, training_options):
    # Setup training options.
    tflib.init_tf({'rnd.np_random_seed': seed})
#     run_desc, training_options = setup_training_options(**hyperparam_options)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(
            outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    training_options.run_dir = os.path.join(
        outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(training_options.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(training_options, indent=2))
    print()
    print(f'Output directory:  {training_options.run_dir}')
    print(f'Training data:     {training_options.train_dataset_args.path}')
    print(f'Training length:   {training_options.total_kimg} kimg')
    print(
        f'Resolution:        {training_options.train_dataset_args.resolution}')
    print(f'Number of GPUs:    {training_options.num_gpus}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Kick off training.
    print('Creating output directory...')
    os.makedirs(training_options.run_dir)
    with open(os.path.join(training_options.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(training_options, f, indent=2)
    with dnnlib.util.Logger(os.path.join(training_options.run_dir, 'log.txt')):
        training_loop.training_loop(**training_options)
