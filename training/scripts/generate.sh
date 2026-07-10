#!/bin/bash

#SBATCH --partition=nvgpu
#SBATCH --constraint="GPU_MEM:96GB"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=23:59:59
#SBATCH --job-name=ARProtein_test

cd ${SLURM_SUBMIT_DIR}

module load cuda/13.0.2
conda deactivate

cd /gpfs1/home/j/j/jjung2/scratch/AMPGANv3/
source .venv/bin/activate

cd /gpfs1/home/j/j/jjung2/scratch/AMPGANv3/training/

RUN=$1
uv run generate_samples.py $RUN
