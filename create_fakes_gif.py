from glob import glob
from PIL import Image
import os
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
# from IPython import display


def create_fakes_gif(DATASET_NAME: string):

    WORK = os.environ["WORK"]
    PROJ_DIR = f'{WORK}/ADA_Project'
    os.chdir(PROJ_DIR)

    TRfolders = f'{PROJ_DIR}/training_runs/'
    TRfolders_ = glob(f'{PROJ_DIR}/training_runs/*')
    datasets = [x.replace(TRfolders, '').replace('_training-runs', '') for x in TRfolders_]
    datasets = ['AFHQ-CAT' if x == 'AFHQ' else x for x in datasets]

    d = {}

    for folder, dataset in zip(TRfolders_, datasets):
        files = sorted(glob(folder + "/**/*"))
        files = [x for x in files if 'fakes' in x]
        if files == []:
            continue
        else:
            d[dataset] = {}
            d[dataset]['files'] = files


    history = f'{PROJ_DIR}/datasets/{DATASET_NAME}_history'
    Path(history).mkdir(exist_ok=True)

    def process(i):
        im = Image.open(i)
        left, top, right, bottom = 0, 0, 1020, 1020
        im_cropped = im.crop((left, top, right, bottom))
        im_cropped.save(f'{history}/{Path(i).stem}.jpg')

    n_jobs = multiprocessing.cpu_count() - 1

    images = Parallel(n_jobs=n_jobs)(delayed(process)(i) for i in tqdm(d[DATASET_NAME]['files']))

    #     display(im_cropped)

    fp_in = glob(f'{DATASET_NAME}/*')
    fp_out = f'{DATASET_NAME}.gif'

    img, *imgs = [Image.open(f) for f in tqdm(fp_in)]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=100, loop=0)
