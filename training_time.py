import os
from glob import glob
import re
import json
from tabulate import tabulate
from pathlib import Path


WORK = os.environ["WORK"]
PROJ_DIR = f'{WORK}/ADA_Project'

def training_time(PROJ_DIR, best_snapshots, verbose=False):
    datasets = glob(f'{PROJ_DIR}/training_runs/*')
    datasets = [str(Path(x).name).replace('_training-runs', '') for x in datasets]
    
    d = {}

    for DATASET in datasets:
        d[DATASET] = {}
        d[DATASET]['times'] = []

        try:
            file = f'{PROJ_DIR}/training_runs/{best_snapshots[DATASET]["file"]}'
            parent_dir = str(Path(f'{PROJ_DIR}/training_runs/{best_snapshots[DATASET]["file"]}').parents[1])
        except KeyError as e:
            if verbose:                      
                print(e)
            del d[DATASET]
            continue

        with open(file) as f:
            lines = f.readlines()

        for n, line in enumerate(lines):
            if best_snapshots[DATASET]['snapshot'] in line:
                best_line_idx = n
                break


        def find_log_train_time(line):
            def calc_time(t, unit):
                s = t.partition(unit)[0][-2:].replace(' ', '')
                if t.partition(unit)[0] != t:
                    return int(s)
                return 0

            findWholeWord = lambda w, s: re.compile(rf'\b({w})\b', flags=re.IGNORECASE
                                                ).search(s)
            sp = findWholeWord('time', line).span()
            t = line[sp[1] + 1:sp[1] + 12]
            last = t.partition('s')[-1]
            t = t.replace(last, '')
            T = (calc_time(t, 'd') * 24) + calc_time(t, 'h') + (
                calc_time(t, 'm') / 60) + (calc_time(t, 's') / 3600)
            return T


        stop_T = find_log_train_time(lines[n - 2])


        d[DATASET]['times'].append(stop_T)


        training_runs = sorted(glob(f'{parent_dir}/*'))
        other_training_runs = [x for x in training_runs if
                               int(str(Path(x).name)[:5]) < int(str(Path(file).parent.name)[:5])]

        other_TR_logs = [x + '/log.txt' for x in other_training_runs]

        for x in other_TR_logs:
            with open(x) as log_f:
                lines = log_f.readlines()
            if 'Exiting...' in lines[-1]:
                line = lines[-5]
            elif 'Evaluating metrics...' in lines[-1]:
                line = lines[-2]
            elif 'network-snapshot' in lines[-1]:
                line = lines[-3]
            elif 'Exporting sample images...' in lines[-1]:
                continue
            else:
                line = lines[-1]
            try:
                T = find_log_train_time(line)
                d[DATASET]['times'].append(T)
            except AttributeError as e:
                if verbose:
                    print(e)

        d[DATASET]['total_time'] = sum(d[DATASET]['times'])

        d[DATASET]['FID'] = best_snapshots[DATASET]['score']

    days = [round(d[x]['total_time'] / 24, 1) for x in d.keys()]
                                  
    table = tabulate([[x, round(d[x]['total_time'], 2), z, round(d[x]['FID'], 2)]
                  for (x, y), z in zip(d.items(), days)],
                 headers=[
                     'Dataset', 'Training time (in hrs)', 'Training time (in days)', 'FID'
                 ],
                 tablefmt='github')
    return table
