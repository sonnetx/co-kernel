#!/bin/bash -l
# Run the install validation on a GPU node. Batch submission means the check
# survives a dropped login session, unlike an interactive compile.
#
#   sbatch --partition=<yours> --constraint=<yours> scripts/slurm/validate.sbatch
#
#SBATCH --job-name=cokernel_validate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

: "${COKERNEL_MODULES:=python/3.12.1 cuda/12.4.0 gcc/12.4.0}"
: "${COKERNEL_VENV:=/scratch/groups/roxanad/sonnet/co-kernel/venv}"
: "${COKERNEL_SCRATCH:=/scratch/groups/roxanad/sonnet/co-kernel}"
: "${COKERNEL_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

ml ${COKERNEL_MODULES}

export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=9.0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$COKERNEL_SCRATCH/tmp"
export COKERNEL_BUILD="$COKERNEL_SCRATCH/build/validate_${SLURM_JOB_ID:-manual}"
export COKERNEL_LOG="$COKERNEL_REPO/logs/validate_${SLURM_JOB_ID:-manual}.log"
mkdir -p "$TMPDIR" "$COKERNEL_BUILD"

source "$COKERNEL_VENV/bin/activate"
cd "$COKERNEL_REPO"

nvidia-smi --query-gpu=name,compute_cap,clocks.sm,persistence_mode --format=csv
exec ./scripts/validate.sh "$@"
