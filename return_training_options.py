import os
from glob import glob
import shutil
from pathlib import Path


def return_training_options(PROJ_DIR=f'{os.environ["WORK"]}/ADA_Project'):

    os.chdir(PROJ_DIR)
    trainOpDir = f'{PROJ_DIR}/training_options'
    shutil.rmtree(trainOpDir)
    Path(trainOpDir).mkdir(exist_ok=False)

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
        files = [
            x for x in files
            if 'training_options' in x or 'submit_config.txt' in x
        ]
        if files == []:
            continue
        d[dataset] = {}
        d[dataset]['file'] = files[-1]

    for _, v in d.items():
        file = v['file']
        file_dst = f'{PROJ_DIR}/training_options/{Path(file).parent.name.replace("AFHQ_custom", "AFHQ-CAT_custom")}__{Path(file).name}'
        shutil.copyfile(file, file_dst)
        print(f' Copying: {file} to ==> {file_dst}\n')


return_training_options()
