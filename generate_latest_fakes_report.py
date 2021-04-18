import os
WORK = os.environ["WORK"]
PROJ_DIR = f'{WORK}/ADA_Project'
os.chdir(PROJ_DIR)

from glob import glob
from pathlib import Path
from pprint import pprint
# import PIL
from PIL import Image
import base64
import requests
import dotenv
from datetime import datetime
# from IPython.display import Markdown, display

import training_time
import FID_results


def generate_latest_fakes_report(PROJ_DIR, verbose=1):
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

    WORK = Path(PROJ_DIR).parent
    TRfolders_ = f'{PROJ_DIR}/training_runs'
    TRfolders = glob(f'{TRfolders_}/*')
    backups_dir = f'{PROJ_DIR}/.tmp_imgs'
    Path(backups_dir).mkdir(exist_ok=True)

    md_content = []
    latest_fakes = []

    now = datetime.now()
    date_time = now.strftime('%m/%d/%Y, %H:%M:%S')
    md_content.append('# Latest fakes\n')
    md_content.append(f'## Date and time: {date_time}\n')

    for folder in TRfolders:
        for num in range(-1, -10, -1):
            files = sorted(glob(folder + "/**/*"))
            files = [x for x in files if 'fakes0' in x]
            if files == []:
                continue
            else:
                latest_fake = sorted(files)[-1]
                latest_fakes.append(latest_fake)
                break

    if verbose == 1:
        print('=' * 90, '\n\nLatest fakes:\n')
        pprint([x.replace(str(TRfolders_), '') for x in latest_fakes])
        print('\n', '=' * 90, '\n')

    for img in latest_fakes:
        image = Image.open(img)
        compressed_path = f'{backups_dir}/{Path(img).stem}' + '.jpg'
        #         output_dim = tuple(x // 2 for x in image.size)
        #         compressed = image.resize(output_dim)
        compressed = image
        compressed.save(compressed_path)
        if verbose == 1:
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

    Tstamp = datetime.now().strftime('%m_%d_%Y__%H_%M')
    report_path = f'{PROJ_DIR}/latest_fakes_reports/{Tstamp}.md'

    FID_results.FID_results()

    with open(report_path, 'w') as f:
        f.write(''.join(md_content))
        f.write(training_time.training_time())

    if verbose == 1:
        print(f'Generated a report at ==> {report_path}')

    return report_path


report_path = generate_latest_fakes_report(PROJ_DIR, verbose=1)

# display(Markdown(report_path))
