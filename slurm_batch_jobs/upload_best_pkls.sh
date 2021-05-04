#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --partition=tmp_anvil 
#SBATCH --ntasks-per-node=4 
#SBATCH --error=/work/chaselab/malyetama/ADA_Project/jobs_log/%J.err 
#SBATCH --output=/work/chaselab/malyetama/ADA_Project/jobs_log/%J.out 
#SBATCH --time=4:00:00 

module load rclone 


rclone mkdir GoogleDrive:/best_pkls
rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/AFHQ-WILD_training-runs/00005-AFHQ-WILD_custom-auto2-resumecustom/network-snapshot-007127.pkl GoogleDrive:/best_pkls/AFHQ-WILD_network-snapshot-007127.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/StyleGAN2_FFHQ_training-runs_30K/00001-stylegan2-FFHQ_custom_30K-2gpu-config-f/network-snapshot-003225.pkl GoogleDrive:/best_pkls/StyleGAN2_FFHQ_30K_network-snapshot-003225.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/metfaces_training-runs/00006-metfaces_custom-auto2-resumecustom/network-snapshot-004112.pkl GoogleDrive:/best_pkls/metfaces_network-snapshot-004112.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/StyleGAN2_AFHQ-DOG_training-runs/00003-stylegan2-AFHQ-DOG_custom-2gpu-config-f/network-snapshot-001209.pkl GoogleDrive:/best_pkls/StyleGAN2_AFHQ-DOG_network-snapshot-001209.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/cars196_training-runs/00014-cars196_custom-auto2-resumecustom/network-snapshot-001843.pkl GoogleDrive:/best_pkls/cars196_network-snapshot-001843.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/AFHQ-DOG_training-runs/00001-AFHQ-DOG_custom-auto2-resumecustom/network-snapshot-010895.pkl GoogleDrive:/best_pkls/AFHQ-DOG_network-snapshot-010895.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/FFHQ_training-runs_5K/00002-FFHQ_custom_5K-auto2-resumecustom/network-snapshot-013639.pkl GoogleDrive:/best_pkls/FFHQ_5K_network-snapshot-013639.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/FFHQ_training-runs_2K/00001-FFHQ_custom_2K-auto2-resumecustom/network-snapshot-011796.pkl GoogleDrive:/best_pkls/FFHQ_2K_network-snapshot-011796.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/StyleGAN2_FFHQ_training-runs_5K/00001-stylegan2-FFHQ_custom_5K-2gpu-config-f/network-snapshot-001371.pkl GoogleDrive:/best_pkls/StyleGAN2_FFHQ_5K_network-snapshot-001371.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/StyleGAN2_WILD-AFHQ_training-runs/00006-stylegan2-AFHQ-WILD_custom-2gpu-config-f/network-snapshot-001532.pkl GoogleDrive:/best_pkls/StyleGAN2_WILD-AFHQ_network-snapshot-001532.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/102flowers_training-runs/00027-102flowers_custom-auto2-resumecustom/network-snapshot-000245.pkl GoogleDrive:/best_pkls/102flowers_network-snapshot-000245.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/StyleGAN2_FFHQ_training-runs/00002-stylegan2-FFHQ_custom-2gpu-config-f/network-snapshot-003306.pkl GoogleDrive:/best_pkls/StyleGAN2_FFHQ_network-snapshot-003306.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/FFHQ_training-runs_30K/00001-FFHQ_custom_30K-auto2-resumecustom/network-snapshot-012902.pkl GoogleDrive:/best_pkls/FFHQ_30K_network-snapshot-012902.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/FFHQ_training-runs/00002-FFHQ_custom-auto2-resumecustom/network-snapshot-013516.pkl GoogleDrive:/best_pkls/FFHQ_network-snapshot-013516.pkl -P

rclone copyto /work/chaselab/malyetamaADA_Project/training_runs/StyleGAN2_FFHQ_training-runs_2K/00001-stylegan2-FFHQ_custom_2K-2gpu-config-f/network-snapshot-000403.pkl GoogleDrive:/best_pkls/StyleGAN2_FFHQ_2K_network-snapshot-000403.pkl -P

