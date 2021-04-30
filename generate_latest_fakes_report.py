import os
import re
import json
import base64
import requests
import dotenv
import io
from glob import glob
from pathlib import Path
from pprint import pprint
from PIL import Image
from datetime import datetime
from IPython.display import Markdown, display
from training_time import training_time


def generate_latest_fakes_report(PROJ_DIR, exclude:list=None, verbose=False, export=False, display_output=False):

    tr = f'{WORK}/ADA_Project/training_runs'

    fid_logs = {}

    tf_folders = glob(f'{tr}/*')
    if exclude is not None:
        exclude = [x + '_training-runs' for x in exclude]
        tf_folders = [x for x in tf_folders if Path(x).name not in exclude]
    
    for f in tf_folders:
        folder = sorted(glob(f'{f}/*'))[-1]
        fid_file = glob(f'{folder}/metric-*.txt')
        if fid_file != []:
            fid_file = fid_file[0]
            dataset = Path(fid_file).parents[1].name.replace(
                '_training-runs', '')
            fid_logs[dataset] = fid_file

#     fid_logs['AFHQ-CAT'] = fid_logs.pop('AFHQ')

#     fid_logs = {k.replace('FFHQ', 'FFHQ_custom'): v for k, v in fid_logs.items()}

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
        DI = snapshots[ds]['scores']
        keys = [list(x.keys()) for x in DI]
        vals = [list(x.values()) for x in DI]
        best_snapshots[ds] = {
            'snapshot': keys[vals.index(min(vals))][0],
            'score': min(vals)[0]
        }

    files = [v.replace('metric-fid50k_full.txt', 'log.txt').replace(
        'metric-fid50k.txt', 'log.txt')
             for k, v in fid_logs.items()]

    for (k, v), f in zip(best_snapshots.items(), files):
        best_snapshots[k]['file'] = f.replace(f'{tr}/' , '')

    if export is True:
        with open(f'{PROJ_DIR}/FID_of_best_snapshots.json', 'w') as out_file:
            json.dump(best_snapshots, out_file, indent=4)

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

    dotenv.load_dotenv(f'{PROJ_DIR}/.env')
    token = os.getenv('TOKEN')

    mb_size = lambda x: Path(x).stat().st_size / (1024 * 1024)
    dir_up = lambda x, y: "/".join(Path(x).parts[y:])

    TRfolders_ = f'{PROJ_DIR}/training_runs'
    TRfolders = glob(f'{TRfolders_}/*')
    backups_dir = f'{PROJ_DIR}/.tmp_imgs'
    Path(backups_dir).mkdir(exist_ok=True)

    md_content = []
    latest_fakes = [str(Path(tr + '/' + best_snapshots[k]["file"]).parent) +
                    f'/{best_snapshots[k]["snapshot"]}.png'.replace('network-snapshot-', 'fakes')
                    for k, v in best_snapshots.items()]

    now = datetime.now()
    date_time = now.strftime('%m/%d/%Y, %H:%M:%S')
    md_content.append('# Latest fakes\n')
    md_content.append(f'## Date and time: {date_time}\n')


    if verbose:
        print('=' * 90, '\n\nLatest fakes:\n')
        pprint([x.replace(str(TRfolders_), '') for x in latest_fakes])
        print('\n', '=' * 90, '\n')

    for img in latest_fakes:
        image = Image.open(img)
        compressed_path = f'{backups_dir}/{Path(img).stem}' + '.jpg'
        
        if 'StyleGAN2_WILD-AFHQ' in img:
            left, top, right, bottom = 0, 0, 256 * 15, 256 * 8
            image = image.crop((left, top, right, bottom))
            temp = io.BytesIO()
            
        image.save(compressed_path)
            
        if verbose:
            print(
                Path(img).name,
                f'compressed from ({mb_size(img):.2f}MB) to ==> '
                f'({mb_size(compressed_path):.2f}MB)')

        url = upload_img(compressed_path, token)
        if verbose == 1:
            print(f'Link ==> {url}\n')
        img_subdir = dir_up(img, -3)

        md_content.append(
            f'### {img_subdir}\n'
            f'![{Path(compressed_path).name}]({url} "{img_subdir}")'
            '\n\n')

#     Tstamp = datetime.now().strftime('%m_%d_%Y__%H_%M')
    report_path = f'{PROJ_DIR}/latest_fakes_report.md'
    table = training_time(PROJ_DIR, best_snapshots, False)
    
    with open(report_path, 'w') as f:
        f.write(''.join(md_content))
        f.write(table)

    if verbose:
        print(f'Generated a report at ==> {report_path}')

    if display_output:
        display(Markdown(report_path))

        
WORK = os.environ["WORK"]
PROJ_DIR = f'{WORK}/ADA_Project'


generate_latest_fakes_report(PROJ_DIR=PROJ_DIR,
                             verbose=True,
                             export=True,
                             display_output=False)

# exclude=['POKEMON', 'ANIME-FACES']

