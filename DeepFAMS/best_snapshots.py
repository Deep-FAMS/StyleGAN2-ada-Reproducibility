import os
import re
from glob import glob
from pathlib import Path
import json


def best_snapshots(export=False):
    WORK = os.environ["WORK"]
    PROJ_DIR = f'{WORK}/ADA_Project'
    tr = f'{WORK}/ADA_Project/training_runs'

    fid_logs = {}

    tf_folders = glob(f'{tr}/*')
    for f in tf_folders:
        folder = sorted(glob(f'{f}/*'))[-1]
        fid_file = glob(f'{folder}/metric-*.txt')
        if fid_file != []:
            fid_file = fid_file[0]
            dataset = Path(fid_file).parents[1].name.replace(
                '_training-runs', '')
            fid_logs[dataset] = fid_file

    fid_logs['AFHQ-CAT'] = fid_logs.pop('AFHQ')

    findWholeWord = lambda w, s: re.compile(rf'\b({w})\b', flags=re.IGNORECASE
                                            ).search(s)

    snapshots = {}

    for k, v in fid_logs.items():
        with open(v) as f:
            lines = f.readlines()
            snapshots[k] = {}
            snapshots[k]['scores'] = []
            for line in lines:
                if 'StyleGAN2' in k:
                    string = 'fid50k'
                else:
                    string = 'fid50k_full'
                sp = findWholeWord(string, line).span()
                snapshot = line[:23]
                score = float(line[sp[-1] + 1:sp[-1] + 7])
                snapshots[k]['scores'].append({f'{snapshot}': score})

    best_snapshots = {}

    for ds in snapshots:
        d = snapshots[ds]['scores']
        keys = [list(x.keys()) for x in d]
        vals = [list(x.values()) for x in d]
        best_snapshots[ds] = {
            f'{keys[vals.index(min(vals))][0]}': min(vals)[0]
        }

    print(json.dumps(best_snapshots, indent=4, sort_keys=True))

    if export is True:
        with open('best_snapshots.json', 'w') as out_file:
            json.dump(best_snapshots, out_file, indent=4)


best_snapshots(export=True)
