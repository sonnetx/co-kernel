#!/bin/bash -l
#SBATCH --job-name=cokernel_day1
#SBATCH --partition=roxanad,gpu
#SBATCH --constraint=GPU_MEM:80GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ml python/3.12.1 cuda/12.4.0 gcc/12.4.0

CK=/scratch/groups/roxanad/sonnet/co-kernel
REPO=/home/groups/roxanad/sonnet/co-kernel

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ETAG_TIMEOUT=10
export HF_HUB_DOWNLOAD_TIMEOUT=60
export TORCH_CUDA_ARCH_LIST=9.0
export HF_HOME=$CK/hf
export TORCH_HOME=$CK/torch
export TMPDIR=$CK/tmp
export COKERNEL_BUILD=$CK/build/$SLURM_JOB_ID
mkdir -p "$TMPDIR" "$COKERNEL_BUILD"

source $CK/venv/bin/activate

cd "$REPO"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,persistence_mode --format=csv
python scripts/day_one.py \
  --corpus corpus \
  --out "results/day_one_${SLURM_JOB_ID}.json" \
  --n-iter 100
