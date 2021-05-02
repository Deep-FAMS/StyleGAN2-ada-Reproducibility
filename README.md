# ADA Project

In this project, I attempt to evaluate the reproducibility and generalizability of the results published in *[Training Generative Adversarial Networks with Limited Data](https://arxiv.org/abs/2006.06676)* with some of the datasets that were used in the paper and some other small datasets.

**StyleGAN2-ada official repository:** https://github.com/NVlabs/stylegan2-ada


![example.jpg](https://i.ibb.co/vD5Z74q/ddb309dc0571.jpg "AFHQ-WILD_training-runs/00004-AFHQ-WILD_custom-auto2-resumecustom/fakes004382.png")


## Getting started
### Clone this repository
```shell
$ git clone git@github.com:Deep-FAMS/ADA_Project.git
```
### Create a conda environments with all dependencies and requirements
```shell
$ cd ADA_Project
$ bash create_project_envs.sh
$ module load cuda compiler/gcc/6.1
```

### Clone source code of StyleGAN2-ada and StyleGAN2
```shell
$ git clone https://github.com/NVlabs/stylegan2-ada.git StyleGAN2-ada  # Replace instances of $HOME with $WORK, or just export $HOME as $WORK
$ git clone https://github.com/NVlabs/stylegan2.git
```

### Create your .env file
```shell
$ mv default_env .env
$ nano .env    # or any other text editor
```
Edit the file to append your working directory to the first line (e.g., `WORK=/work/my_projects`). The working directory should be the parent directory of this repository. **THIS IS VERY IMPORTANT!**


### Create subdirectories tree
```shell
$ mkdir datasets training_runs .tmp .tmp_imgs jobs_log 
```

Now everything is ready! :tada:
