#!/bin/bash
#SBATCH --job-name=dd
#SBATCH --open-mode=append
#SBATCH --output=./monitor/dd.out
#SBATCH --error=./monitor/dd.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH -c 4


singularity exec --nv --overlay /scratch/sz4651/Projects/overlay-share2.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source ~/.bashrc \
&& source activate /scratch/sz4651/Projects/ext3/miniconda3/envs/diffusion_embedder \
&& cd /scratch/sz4651/Projects/diffusion_embedder \
&& ./run2.sh"
