from glob import glob
import os
import subprocess
import shlex


def last_snap(num, training_runs_dir):
    files = glob(f'{sorted(glob(training_runs_dir + "/*"))[num]}/*')
    files = [x for x in files if 'network-snapshot' in x]  
    return files

             
def execute(command: str):
    command = shlex.split(command)
    stdout = subprocess.run(command, capture_output=True, text=True).stdout
    lines = "\n".join([line for line in stdout.strip().splitlines()])
    print(lines)
