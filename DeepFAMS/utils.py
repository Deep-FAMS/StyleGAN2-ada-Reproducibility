from glob import glob
import os
import subprocess
import shlex
import tarfile
import urllib.request


def last_snap(num, training_runs_dir):
    files = glob(f'{sorted(glob(training_runs_dir + "/*"))[num]}/*')
    files = [x for x in files if 'network-snapshot' in x]  
    return files

             
def execute(command: str):
    command = shlex.split(command)
    stdout = subprocess.run(command, capture_output=True, text=True).stdout
    lines = "\n".join([line for line in stdout.strip().splitlines()])
    print(lines)


def executePopen(cmd: str):
    with open('tmp_script.sh', 'w') as f:
        f.write(cmd)

    p = subprocess.Popen("bash tmp_script.sh",
                     shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    while p.poll() is None:
        line = p.stdout.readline()
        print(line)
    
    os.remove('tmp_script.sh')


def return_dirs(PROJ_DIR, dataset_name):
    RAW_IMGS_DIR = f'{PROJ_DIR}/datasets/{dataset_name}_images_raw'
    RESIZED_IMGS_DIR = f'{PROJ_DIR}/datasets/{dataset_name}_resized'
    DATA_CUSTOM_DIR = f'{PROJ_DIR}/datasets/{dataset_name}_custom'
    TRAIN_RUNS_DIR = f'{PROJ_DIR}/training_runs/{dataset_name}_training-runs'

    return RAW_IMGS_DIR, RESIZED_IMGS_DIR, DATA_CUSTOM_DIR, TRAIN_RUNS_DIR
 

def Get_Raw_Data(url, datasets_dir, RAW_IMGS_DIR, file_name):
    file_name = url.split('/')[-1]
    urllib.request.urlretrieve(url, f'{datasets_dir}/{file_name}')
    tarf = tarfile.open(f'{datasets_dir}/{file_name}')
    tarf.extractall(path=RAW_IMGS_DIR)

    print(f'Downloaded and extracted to:\n{RAW_IMGS_DIR}')
