#!/bin/bash -l
# Site-specific example: Stanford Sherlock, H100 80GB.
#
# Treat this as a template rather than a portable script. The module names,
# partitions, and paths are particular to one cluster. On any single-GPU box,
# use scripts/run_local.sh instead. Everything site-specific here is an
# environment variable with a default, so adapting it means editing the block
# below rather than hunting through the file.
#
#   sbatch --partition=<yours> --constraint=<yours> scripts/slurm/sherlock.sbatch
#
#SBATCH --job-name=cokernel_day1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# --- site configuration ------------------------------------------------------
: "${COKERNEL_MODULES:=python/3.12.1 cuda/12.4.0 gcc/12.4.0}"
: "${COKERNEL_VENV:=/scratch/groups/roxanad/sonnet/co-kernel/venv}"
: "${COKERNEL_SCRATCH:=/scratch/groups/roxanad/sonnet/co-kernel}"
# SLURM copies this script to a spool directory, so BASH_SOURCE does not point
# into the repo. SLURM_SUBMIT_DIR is where sbatch was invoked, which is the
# repo root by convention here; the fallback covers running the file directly.
: "${COKERNEL_REPO:=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
: "${TORCH_CUDA_ARCH_LIST:=9.0}"
# -----------------------------------------------------------------------------

ml ${COKERNEL_MODULES}

export TORCH_CUDA_ARCH_LIST
export PYTHONUNBUFFERED=1
# Fail fast and loudly instead of hanging on a HuggingFace metadata request
# that never returns while holding a GPU.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ETAG_TIMEOUT=10
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HOME="$COKERNEL_SCRATCH/hf"
export TORCH_HOME="$COKERNEL_SCRATCH/torch"
export TMPDIR="$COKERNEL_SCRATCH/tmp"
export COKERNEL_BUILD="$COKERNEL_SCRATCH/build/${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR" "$COKERNEL_BUILD"

source "$COKERNEL_VENV/bin/activate"
cd "$COKERNEL_REPO"

nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,persistence_mode --format=csv

python scripts/day_one.py \
  --corpus corpus \
  --out "results/day_one_${SLURM_JOB_ID:-manual}.json" \
  "$@"
