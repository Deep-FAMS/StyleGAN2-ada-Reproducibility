import os
WORK = os.environ["WORK"]
PROJ_DIR = f'{WORK}/ADA_Project'
os.chdir(PROJ_DIR)

from glob import glob
import re
# import pandas as pd
from tabulate import tabulate


def training_time(measure: str):    # 'days' or 'hrs'
    TRfolders = f'{PROJ_DIR}/training_runs/'
    TRfolders_ = glob(f'{PROJ_DIR}/training_runs/*')
    datasets = [x.replace(TRfolders, '').replace('_training-runs', '') for x in TRfolders_]
    datasets = ['AFHQ-CAT' if x == 'AFHQ' else x for x in datasets]

    d = {}

    for folder, dataset in zip(TRfolders_, datasets):
        files = sorted(glob(folder + "/**/*"))
        files = [x for x in files if 'metric' in x]
        if files == []:
            continue
        else:
            d[dataset] = {}
            d[dataset]['files'] = files
            d[dataset]['training_time'] = []

    findWholeWord = lambda w, s: re.compile(rf'\b({w})\b', flags=re.IGNORECASE).search(s)

    for metric, values in d.items():
        for v in values['files']:
            with open(v, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    sp = findWholeWord('time', line).span()
                    t = line[sp[1]+1:sp[1]+7]
                    T = int(t[0]) * 60 + int(t[3:-1])
                    values['training_time'].append(T)

    TTs = []
    dvby, measure = (3600, 'hrs') if measure == 'hrs' else (86400, 'days')
    for (x, y) in d.items():
        TTs.append(sum(y["training_time"]) / dvby)

#     df = pd.DataFrame(TTs, index=False, columns=[f'Training time (in {measure})']).round(2)
#     df.index.name = 'Dataset'
    
    table = tabulate([[x, y] for x, y in zip(datasets, [round(i, 2) for i in TTs])],
               headers=['Dataset', f'Training time (in {measure})'],
               tablefmt='github')
    
    return table
