#!/bin/bash

# StyleGAN2-ada env

module load anaconda

conda create -p $WORK/.conda/envs/ada-env -f ./envs/environment.yml
conda activate $WORK/.conda/envs/ada-env

python -m ipykernel install --user --name "$CONDA_DEFAULT_ENV" --display-name "Python ($CONDA_DEFAULT_ENV)"

# run the next line only if you don't already have a .jupyter/kernels folder
# mkdir -p $WORK/.jupyter/kernels
mv ~/.local/share/jupyter/kernels/$CONDA_DEFAULT_ENV/ $WORK/.jupyter/kernels/

conda config --append envs_dirs $WORK/.conda/envs/$CONDA_DEFAULT_ENV
conda config --set env_prompt '({name})'


#------------------------------------------------------------

# Baseline StyleGAN2 env

conda create -p $WORK/.conda/envs/stylegan2 -f ./envs/StyleGAN2_environment.yml
conda activate $WORK/.conda/envs/stylegan2

python -m ipykernel install --user --name "$CONDA_DEFAULT_ENV" --display-name "Python ($CONDA_DEFAULT_ENV)"

# run the next line only if you don't already have a .jupyter/kernels folder
# mkdir -p $WORK/.jupyter/kernels
mv ~/.local/share/jupyter/kernels/$CONDA_DEFAULT_ENV/ $WORK/.jupyter/kernels/

conda config --append envs_dirs $WORK/.conda/envs/$CONDA_DEFAULT_ENV
conda config --set env_prompt '({name})'



# restart your shell, then run
# module load cuda compiler/gcc/6.1
# conda activate ada-env  # or stylegan2
