#!/bin/bash
#SBATCH --account=s4754411
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/job_%j.out
#SBATCH --error=slurm_logs/job_%j.error


conda activate comp3710 

python training.py