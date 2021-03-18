#!/bin/bash

module load tensorflow-gpu/py37/1.14
module load anaconda
conda create -p $WORK/.conda/envs/ada-env --clone $CONDA_DEFAULT_ENV
module unload tensorflow-gpu/py37/1.14

conda activate $WORK/.conda/envs/ada-env

python -m ipykernel install --user --name "$CONDA_DEFAULT_ENV" --display-name "Python ($CONDA_DEFAULT_ENV)"

mv ~/.local/share/jupyter/kernels/ada-env/ $WORK/.jupyter/kernels/

conda install -y scipy==1.3.3 requests==2.22.0 Pillow==6.2.1 h5py==2.9.0 imageio==2.9.0 imageio-ffmpeg==0.4.2 tqdm==4.49.0 install

conda install -y cudatoolkit==10.0.130

conda config --append envs_dirs $WORK/.conda/envs/ada-env

git clone git@github.com:Deep-FAMS/ADA_Project.git $WORK/ADA_Project

