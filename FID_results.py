import os
from glob import glob
import re


def FID_results(PROJ_DIR=f'{os.environ["WORK"]}/ADA_Project'):
    
    os.chdir(PROJ_DIR)
    
    TRfolders = f'{PROJ_DIR}/training_runs/'
    TRfolders_ = glob(f'{PROJ_DIR}/training_runs/*')
    datasets = [x.replace(TRfolders, '').replace('_training-runs', '') for x in TRfolders_]
    datasets = ['AFHQ-CAT' if x == 'AFHQ' else x for x in datasets]

    d = {}

    for folder, dataset in zip(TRfolders_, datasets):
        files = sorted(glob(folder + "/**/*"))
        files = [x for x in files if 'metric' in x and 'pokemon' not in x]
        if files == []:
            continue
        else:
            d[dataset] = {}
            d[dataset]['file'] = files[-1]
            d[dataset]['FID'] = []

    findWholeWord = lambda w, s: re.compile(rf'\b({w})\b', flags=re.IGNORECASE).search(s)

    for metric, values in d.items():
        with open(values['file'], 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                sp_loc = findWholeWord('time', line).span()
                sp = line[sp_loc[1]-1:sp_loc[1]+14]
                line = line.replace(sp, '').replace('\n', '')
                line = line.replace('        ', ': ')
                values['FID'].append(line)
                break

    for skey in d.keys():
        del d[skey]['file']

    with open("FID_results.json", "w") as outfile: 
        json.dump(d, outfile, indent = 4)


FID_results()
