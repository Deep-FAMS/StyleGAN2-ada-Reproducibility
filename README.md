# ADA Project

In this project, I attempt to evaluate the reproducibility and generalizability of the results published in *[Training Generative Adversarial Networks with Limited Data](arxiv.org/abs/2006.06676)* with some of the datasets that were used in the paper and some other small datasets.

**StyleGAN2-ada official repository:** https://github.com/NVlabs/stylegan2-ada


## Getting started
### Clone this repository
```shell
$ git clone git@github.com:Deep-FAMS/ADA_Project.git
```
### Create a conda environment with all the requirements
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
Edit the file to append your working directory to the first line (e.g., `WORK=/home/my_projects`). The working directory should be the same directory you cloned this reposotory to. **THIS IS VERY IMPORTANT!**


### Create subdirectories tree
```shell
$ mkdir datasets training_runs .tmp .tmp_imgs jobs_log 
```

From here, you can start using the notebooks after you [add the new environment to your jupyter kernerls](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments)! :tada:

If you want to submit jobs on Crane (or any other HPC that uses Slurm), see the next section.


## Submitting jobs to the cluster
If you want to run any of the script in [py_scripts](./py_scripts) interactively, then you will have to load few modules before you start.
```shell
module load cuda
module load compiler/gcc/6.1
```

<br></br>
... `README.md` is under construction 👷‍! I will add more details soon!
