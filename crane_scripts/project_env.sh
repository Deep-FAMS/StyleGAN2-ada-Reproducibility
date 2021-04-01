#!/bin/bash

module load anaconda

conda create -p $WORK/.conda/envs/ada-env -f ../environment.yml
conda activate $WORK/.conda/envs/ada-env
python -m ipykernel install --user --name "$CONDA_DEFAULT_ENV" --display-name "Python ($CONDA_DEFAULT_ENV)"

# run the next line only if you don't already have a .jupyter/kernels folder
# mkdir -p $WORK/.jupyter/kernels
mv ~/.local/share/jupyter/kernels/ada-env/ $WORK/.jupyter/kernels/

conda config --append envs_dirs $WORK/.conda/envs/ada-env
conda config --set env_prompt '({name})'

# restart your shell, then run
module load cuda
module load compiler/gcc/6.1
conda activate ada-env
