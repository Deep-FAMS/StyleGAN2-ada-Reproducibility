# ADA Project

In this project, I attempt to evaluate the reproducibility and generalizability of the results published in *[Training Generative Adversarial Networks with Limited Data](https://arxiv.org/abs/2006.06676)* with some of the datasets that were used in the paper and some other small datasets.

**StyleGAN2-ada official repository:** https://github.com/NVlabs/stylegan2-ada


![example.jpg](https://i.ibb.co/vD5Z74q/ddb309dc0571.jpg "AFHQ-WILD_training-runs/00004-AFHQ-WILD_custom-auto2-resumecustom/fakes004382.png")


## Getting started
### Clone this repository
```shell
$ git clone git@github.com:Deep-FAMS/ADA_Project.git
```
### Create a conda environment with all dependencies and requirements
```shell
$ cd ADA_Project
$ conda env create -f environment.yml
$ conda activate ada-env
```

### Create your .env file
```shell
# in ./ADA_Project
$ mv default_env .env
$ nano .env    # or any other text editor
```
Edit the file to append your working directory to the first line (e.g., `WORK=/home/my_projects`). The working directory should be the same directory you cloned this repository to. **THIS IS VERY IMPORTANT!**


### Create subdirectories tree
```shell
$ mkdir datasets training_runs .tmp .tmp_imgs jobs_log 
```

From here, you can start using the notebooks after you [add the new environment to your jupyter kernels](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments)! :tada:

If you want to submit jobs on Crane (or any other HPC that uses Slurm), see the next section.


## Working on a cluster

On Crane, to avoid permission errors, it's advisable that you install your environment on $WORK (replace $WORK with your working directory if it's different). Hence, you may have to specify the installation directory of your conda environment.

```shell
$ module load anaconda

$ conda env create -p $WORK/.conda/envs/ada-env -f environment.yml
$ conda activate $WORK/.conda/envs/ada-env
$ conda install -c conda-forge ipykernel
$ python -m ipykernel install --user --name "$CONDA_DEFAULT_ENV" --display-name "Python ($CONDA_DEFAULT_ENV)"

# run the next line only if you don't already have a .jupyter/kernels folder
# $ mkdir -p $WORK/.jupyter/kernels
$ mv ~/.local/share/jupyter/kernels/"$CONDA_DEFAULT_ENV" $WORK/.jupyter/kernels/

$ conda config --append envs_dirs $WORK/.conda/envs/"$CONDA_DEFAULT_ENV"
$ conda config --set env_prompt '({name})'

# restart your shell, then run
$ module load cuda
$ module load compiler/gcc/6.1
$ conda activate ada-env
```

If you want to run any of the scripts in [py_scripts](./py_scripts) interactively in the terminal or submit them as a batch job, then you will have to load few modules before you start.
```shell
$ module load cuda
$ module load compiler/gcc/6.1
```

<br></br>
... `README.md` is under construction 👷‍! I will add more details soon!
