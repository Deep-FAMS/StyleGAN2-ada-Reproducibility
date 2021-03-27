# ADA Project

## Getting started
### Clone this repository
```shell
$ git clone git@github.com:Deep-FAMS/ADA_Project.git
```
### Create a conda environment with all the requirements
```shell
$ cd ADA_Project
$ conda env create -f environment.yml
```

### Create your .env file
```shell
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


... `README.md` is under construction 👷‍! I will add more details soon!
