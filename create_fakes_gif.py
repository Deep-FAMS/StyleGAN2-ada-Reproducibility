from glob import glob
from PIL import Image
import os
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
import imageio
from pygifsicle import optimize
from IPython.display import Markdown, display


def create_fakes_gif(DATASET_NAME, output_dir=None, subset=None, display_gif=False, verbose=False):
    
    def process(i):
        im = Image.open(i)
        if DATASET_NAME == 'metfaces':
            left, top, right, bottom = 0, 0, 1020 * 2, 1020 * 2
        else:
            left, top, right, bottom = 0, 0, 1020, 1020
        im_cropped = im.crop((left, top, right, bottom))
        return im_cropped.save(f'{history}/{Path(i).stem}.jpg')

    WORK = os.environ["WORK"]
    PROJ_DIR = f'{WORK}/ADA_Project'

    TRfolders = f'{PROJ_DIR}/training_runs/'
    TRfolders_ = glob(f'{PROJ_DIR}/training_runs/*')
    datasets = [
        x.replace(TRfolders, '').replace('_training-runs', '')
        for x in TRfolders_
    ]
    ds_rename = lambda before, after: [after if x == before else x for x in datasets]
    datasets = ds_rename('AFHQ', 'AFHQ-CAT')
    datasets = ds_rename('FFHQ', 'FFHQ_FULL')
    
    if verbose:
        print(f'Available datasets:\n {datasets}')

    d = {}

    for folder, dataset in zip(TRfolders_, datasets):
        files = sorted(glob(folder + "/**/*"))
        fakes = [x for x in files if 'fakes' in x]
        if fakes == []:
            continue
        d[dataset] = {}
        d[dataset]['files'] = fakes[::subset]
    
    history = f'{PROJ_DIR}/datasets/{DATASET_NAME}_history'
    try:
        shutil.rmtree(history)
    except:
        pass
    Path(history).mkdir(exist_ok=True)
    
    if output_dir is None:
        output_dir = Path.cwd()

    n_jobs = multiprocessing.cpu_count() - 1

    _ = Parallel(n_jobs=n_jobs)(delayed(process)(i)
                                     for i in tqdm(d[DATASET_NAME]['files']))

    history_imgs = sorted([x for x in glob(f'{history}/*.jpg')])
    history_imgs = [history_imgs[-1]] + [x for x in history_imgs if 'init' not in x]
    
    if subset is not None and verbose:
        print(f'Subset size: {len(history_imgs)} image')
    
    anim_file = f'{DATASET_NAME}.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        for filename in history_imgs:
            image = imageio.imread(filename)
            if filename == history_imgs[-1]:
                for _ in range(20):
                    writer.append_data(image)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    
    file_size = lambda file: Path(file).stat().st_size / 1e+6
    if verbose:
        print(f'gif size before optimization: {file_size(anim_file):.2f} MB')
    optimize(source=anim_file, destination=anim_file)
    if verbose:
        print(f'         after optimization: {file_size(anim_file):.2f} MB')
    
    if display_gif is True:
        print('Loading...')
        return display(Markdown(f'<img src="{anim_file}" width="600">'))


# create_fakes_gif(DATASET_NAME='FFHQ_5K', display_gif=True, verbose=True)
