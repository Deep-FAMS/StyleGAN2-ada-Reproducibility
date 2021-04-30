from glob import glob
from PIL import Image as Img
import os
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
import imageio
from pygifsicle import optimize
from IPython.display import Markdown, display, Image
import dotenv
import base64
import requests
import json
import shutil


def create_fakes_gif(
    DATASET_NAME,
    subset=None,
    output_dir=None,
    ftype='gif',
    display_output=False,
    verbose=False,
    shift={
        'shift_r': 0,
        'shift_b': 0
    }
):
    
    def process(i):
        im = Img.open(i)
        if DATASET_NAME == 'metfaces':
            left, top, right, bottom = 0, 0, (256 * 4) * 2, (256 * 4) * 2
        else:
            left, top, right, bottom = 1 * shift['shift_r'], 1 * shift[
                'shift_b'], (256 * 4) + shift['shift_r'], (256 * 4) + shift['shift_b']
        im_cropped = im.crop((left, top, right, bottom))
        return im_cropped.save(f'{history}/{Path(i).stem}.png')
    
    def upload_img(image, token):
        with open(image, "rb") as file:
            url = "https://api.imgbb.com/1/upload"
            parameters = {
                "key": token,
                "image": base64.b64encode(file.read()),
            }
            res = requests.post(url, parameters)
            link = res.json()
            url = link['data']['url']
            return url

    WORK = os.environ["WORK"]
    PROJ_DIR = f'{WORK}/ADA_Project'
    
    dotenv.load_dotenv(f'{PROJ_DIR}/.env')
    token = os.getenv('TOKEN')
    
    history = f'{PROJ_DIR}/datasets/{DATASET_NAME}_history'
    try:
        shutil.rmtree(history)
    except:
        pass
    
    TRfolders = f'{PROJ_DIR}/training_runs/'
    TRfolders_ = glob(f'{PROJ_DIR}/training_runs/*')
    datasets = [
        x.replace(TRfolders, '').replace('_training-runs', '')
        for x in TRfolders_
    ]
    ds_rename = lambda before, after: [after if x == before else x for x in datasets]
    datasets = ds_rename('AFHQ', 'AFHQ-CAT')
    
    if verbose:
        print(f'Available datasets:\n {datasets}')

    d = {}
    
    with open(f'{PROJ_DIR}/FID_of_best_snapshots.json') as jf:
        jd = json.load(jf)
        best = 'fakes' + jd[DATASET_NAME]['snapshot'].replace('network-snapshot-', '')

    for folder, dataset in zip(TRfolders_, datasets):
        files = sorted(glob(folder + "/**/*"))
        fakes = [x for x in files if 'fakes' in x]
        if fakes == []:
            continue
        d[dataset] = {}
        d[dataset]['files'] = fakes[::subset] + [
            f'{Path(fakes[-1]).parent}/{best}{Path(fakes[-1]).suffix}']
    
    Path(history).mkdir(exist_ok=True)
    
    n_jobs = multiprocessing.cpu_count() - 1

    _ = Parallel(n_jobs=n_jobs)(delayed(process)(i)
                                     for i in tqdm(d[DATASET_NAME]['files']))

    history_imgs = sorted([x for x in glob(f'{history}/*.png')])
    history_imgs = [history_imgs[-1]] + [x for x in history_imgs if 'init' not in x]
    
    
    if subset is not None and verbose:
        print(f'Subset size: {len(history_imgs)} image')

    if output_dir is None:
        output_dir = Path.cwd()
        anim_file = f'{output_dir}/{DATASET_NAME}.{ftype}'
    anim_file = f'{DATASET_NAME}.{ftype}'

    with imageio.get_writer(anim_file, mode='I') as writer:
        for filename in tqdm(history_imgs):
            image = imageio.imread(filename)
            writer.append_data(image)
            if str(Path(filename).stem) == best:
                for _ in range(20):
                    writer.append_data(image)
                break
    
    file_size = lambda file: Path(file).stat().st_size / 1e+6
    if verbose:
        if ftype == 'mp4':
            print(f'file size: {file_size(anim_file):.2f} MB')
        else:
            print(f'file size before optimization: {file_size(anim_file):.2f} MB')
    
    if ftype == 'gif':
        optimize(source=anim_file, destination=anim_file)
            
        if verbose:
            print(f'          after optimization: {file_size(anim_file):.2f} MB')
    
    if display_output is True:
        if ftype == 'gif':
            print('Loading...')
            img_url = upload_img(anim_file, token)
            print(f'{ftype} url ==> {img_url}')
            display(Markdown(f'![]({img_url})'))


# create_fakes_gif(
#     DATASET_NAME='AFHQ-DOG',
#     display_output=False,
#     verbose=True,
#     subset=10,
#     ftype='mp4',
#     shift={
#         'shift_r': 256 * 5,
#         'shift_b': 256 * 3
#     }
# )
