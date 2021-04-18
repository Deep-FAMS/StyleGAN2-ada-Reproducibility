import os
from glob import glob
import re
import json
from pathlib import Path
import numpy as np
import itertools
import ast

# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def calc_time(t, unit):
    s = t.partition(unit)[0][-2:].replace(' ', '')
    if t.partition(unit)[0] != t:
        return float(s)
    else:
        return 0


WORK = os.environ["WORK"]
PROJ_DIR = f'{WORK}/ADA_Project'
os.chdir(PROJ_DIR)

TRfolders = f'{PROJ_DIR}/training_runs/'
TRfolders_ = glob(f'{PROJ_DIR}/training_runs/*')
datasets = [
    x.replace(TRfolders, '').replace('_training-runs', '') for x in TRfolders_
]
datasets = ['AFHQ-CAT' if x == 'AFHQ' else x for x in datasets]

d = {}

for folder, dataset in zip(TRfolders_, datasets):
    files = sorted(glob(folder + "/**/*"))
    files = [x for x in files if 'log' in x]
    if files == []:
        continue
    else:
        d[dataset] = {}
        d[dataset]['files'] = files
        d[dataset]['training_time'] = []

findWholeWord = lambda w, s: re.compile(rf'\b({w})\b', flags=re.IGNORECASE
                                        ).search(s)

FID_history = {}

for dataset, values in d.items():

    FID_history[dataset] = {}
    FID_history[dataset]['FID'] = []
    FID_history[dataset]['times'] = {}

    for v in values['files']:
        snapshot_n = 'snapshot_' + Path(v).parent.name.partition('-')[0]
        FID_history[dataset]['times'][snapshot_n] = []

        if 'StyleGAN2' in dataset:
            options_file = glob(f'{Path(v).parent}/submit_config.txt')
            if options_file == []:
                continue
            with open(options_file[0]) as dfile:
                options = dfile.read()
                options = options.replace('>', '"').replace('<', '"')
                options = ast.literal_eval(options)
                network_snapshot_ticks = options['run_func_kwargs'][
                    'network_snapshot_ticks']
            metric = 'fid50k'

        else:
            options_file = glob(f'{Path(v).parent}/training_options.json')
            if options_file == []:
                continue
            with open(options_file[0]) as jf:
                options = json.load(jf)
                network_snapshot_ticks = options['network_snapshot_ticks']
            metric = 'fid50k_full'

        with open(v, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if metric in line and 'Calculating' not in line:
                    sp = findWholeWord(metric, line).span()
                    FID_history[dataset]['FID'].append(
                        float(line[sp[1] + 1:sp[1] + 9]))
                elif 'tick' in line:
                    tick = findWholeWord('tick', line).span()
                    tick = int(line[tick[1] + 1:tick[1] + 5].replace(' ', ''))

                    if not tick % network_snapshot_ticks:
                        sp = findWholeWord('time', line).span()
                        t = line[sp[1] + 1:sp[1] + 12]
                        last = t.partition('s')[-1]
                        t = t.replace(last, '')
                        T = (calc_time(t, 'd') * 24) + calc_time(
                            t, 'h') + (calc_time(t, 'm') /
                                       60) + (calc_time(t, 's') / 3600)
                        FID_history[dataset]['times'][snapshot_n].append(
                            round(T, 2))

    all_FIDs = FID_history[dataset]['FID']

    i = 0
    for x in FID_history[dataset]['times'].keys():
        try:
            LAST = FID_history[dataset]['times'][x][-1]
            FID_history[dataset]['times'][x] = list(
                np.array(FID_history[dataset]['times'][x]) + i)
            i += LAST
        except IndexError:
            continue

    all_times = [
        FID_history[dataset]['times'][x]
        for x in FID_history[dataset]['times'].keys()
        if FID_history[dataset]['times'][x] != []
    ]
    all_times = list(itertools.chain.from_iterable(all_times))
    all_times = [round(x, 2) for x in all_times]

    all_times = FID_history[dataset]['times'] = all_times

    if len(all_FIDs) != len(all_times):
        lens = [len(all_times), len(all_FIDs)]
        if lens[0] < lens[1]:
            FID_history[dataset]['FID'] = all_FIDs[:lens[0]]
        else:
            FID_history[dataset]['times'] = all_times[:lens[1]]


# Compare between StyleGAN2-ada and baseline StyleGAN2 (AFHQ-WILD dataset)
fig = plt.figure(figsize=(12, 6))
ax = plt.axes()

ds1 = 'AFHQ-WILD'
ds2 = 'StyleGAN2_WILD-AFHQ'

ds1_t = [
    x for x in FID_history[ds1]['times'] if x <= max(FID_history[ds2]['times'])
]
ds1_t = [int(x) for x in ds1_t]

ds1_f = FID_history[ds1]['FID'][:len(ds1_t)]

ds2_t = FID_history[ds2]['times']
ds2_f = FID_history[ds2]['FID']

ax.plot(ds1_t, ds1_f, '-g', label='StyleGAN2-ada')
ax.plot(ds2_t, ds2_f, '-.b', label='Baseline StyleGAN2')

ax.set(xlabel='Time (in hrs)', ylabel='FID score', title=ds1)
plt.xticks(np.arange(0, max(ds1_t), 2))
plt.yticks(np.arange(0, max(ds1_f), 5))
plt.ylim(0, 60)
plt.legend()
