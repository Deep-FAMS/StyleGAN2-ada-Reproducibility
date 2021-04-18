import os

WORK = os.environ["WORK"]
PROJ_DIR = f'{WORK}/ADA_Project'
os.chdir(PROJ_DIR)

from glob import glob
import re
import json
# import pandas as pd
from tabulate import tabulate


def training_time():
    TRfolders = f'{PROJ_DIR}/training_runs/'
    TRfolders_ = glob(f'{PROJ_DIR}/training_runs/*')
    datasets = [
        x.replace(TRfolders, '').replace('_training-runs', '')
        for x in TRfolders_
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

    def calc_time(t, unit):
        s = t.partition(unit)[0][-2:].replace(' ', '')
        if t.partition(unit)[0] != t:
            return int(s)
        else:
            return 0

    TTs = {}
    for dataset, values in d.items():
        for v in values['files']:
            with open(v, 'r') as f:
                lines = f.readlines()
                if 'Exiting...' in lines[-1]:
                    line = lines[-5]
                elif 'Evaluating metrics...' in lines[-1]:
                    line = lines[-2]
                elif 'Exporting sample images...' in lines[-1]:
                    continue
                else:
                    line = lines[-1]
                try:
                    sp = findWholeWord('time', line).span()
                    t = line[sp[1] + 1:sp[1] + 12]
                    last = t.partition('s')[-1]
                    t = t.replace(last, '')
                    T = (calc_time(t, 'd') * 24) + calc_time(t, 'h') + (
                        calc_time(t, 'm') / 60) + (calc_time(t, 's') / 3600)
                    values['training_time'].append(T)

                except AttributeError:
                    continue

        TTs[dataset] = sum(d[dataset]["training_time"])

    # df = pd.DataFrame.from_dict(TTs, orient='index',
    #                        columns=['Training time (in hrs)'])

    # print(df)

    days = [round(j / 24, 1) for i, j in TTs.items()]

    with open(f'{PROJ_DIR}/FID_results.json') as f:
        FID_res = json.load(f)

    FIDs = []
    loc_FID = lambda x, y: round(float(y.partition(x)[-1].replace(' ', '')), 2)

    for x in FID_res.keys():
        y = FID_res[x]['FID'][0]
        if y.find('timfid50k_full') != -1:
            FIDs.append(loc_FID('timfid50k_full', y))
        else:
            FIDs.append(loc_FID('timfid50k', y))

    table = tabulate([[x, round(y, 2), z, f]
                      for (x, y), z, f in zip(TTs.items(), days, FIDs)],
                     headers=[
                         'Dataset', 'Training time (in hrs)',
                         'Training time (in days)', 'FID'
                     ],
                     tablefmt='github')

    return table
