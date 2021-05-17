# ADA Project

In this project, I attempt to evaluate the reproducibility and generalizability of the results published in *[Training Generative Adversarial Networks with Limited Data](https://arxiv.org/abs/2006.06676)* with some of the datasets that were used in the paper and some other small datasets.

**StyleGAN2-ada official repository:** https://github.com/NVlabs/stylegan2-ada


![example.jpg](https://i.ibb.co/vD5Z74q/ddb309dc0571.jpg "AFHQ-WILD_training-runs/00004-AFHQ-WILD_custom-auto2-resumecustom/fakes004382.png")


## Getting started
### Clone this repository
```shell
$ git clone git@github.com:Deep-FAMS/ADA_Project.git
```
### Create conda environments with all dependencies and requirements
```shell
$ cd ADA_Project
$ bash create_project_envs.sh
$ module load cuda compiler/gcc/6.1    # on Crane
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

Now you're ready to go! :tada:

---

## Pretrained Weights*

| Dataset             |   Training time (in hrs) |   FID | Pickle file |
|---------------------|--------------------------|-------|-------------|
| AFHQ-WILD           |                   116.86 |  2.04 | [Download](https://drive.google.com/uc?id=1p-M_PICnek3hLwT4LPsyhh3LnvpaUm19) | 
| metfaces            |                   181.01 | 18.26 | [Download](https://drive.google.com/uc?id=1tQTh5sTMg_VaU98wmPAZ9VxKNIe2wu5D) |
| cars196             |                   139.06 |  8.07 | [Download](https://drive.google.com/uc?id=16eH9cZ--1onDLzZC3m4xDuZxYrLQED0_) |
| AFHQ-DOG            |                    65.98 |  8.68 | [Download](https://drive.google.com/uc?id=1uUYLWP0-A3tZrVKkcJTMnGiFHHWtvChc) |
| 102flowers          |                   119.1  |  6.85 | [Download](https://drive.google.com/uc?id=1tOW9eJnoWvjF-YKveT9Mnkbaft3VBEL1) |
| FFHQ                |                    71.45 |  6.16 | [Download](https://drive.google.com/uc?id=1Yt0H31FVXRGh5opi7XU7zkhd2tCCfHQV) |
| ANIME-FACES         |              92.51   | 19.53 | [Download](https://drive.google.com/uc?id=1o8z9mgoR7JqZnhAmh3C7sZKw6YmVOt2n) |
| StanfordDogs | 182.31 | 31.56 | [Download](https://drive.google.com/uc?id=1CovWMe3vFbglxBOftgfTkr50crMn67sr) |
| Best-Artworks-of-All-Time | 72 | 19.87 | [Download](https://drive.google.com/uc?id=1Ezn7TT6yeZzjCUw0XiDqnoxrsPfYTQNk) |

*All models are trained on 2 Tesla V100 GPUs.
