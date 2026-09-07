#!/bin/bash -l
# Capability probe on a GPU node. Fast, so it backfills easily.
#
#SBATCH --job-name=cokernel_probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

: "${COKERNEL_MODULES:=python/3.12.1 cuda/12.4.0 gcc/12.4.0}"
: "${COKERNEL_VENV:=/scratch/groups/roxanad/sonnet/co-kernel/venv}"
: "${COKERNEL_REPO:=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"

ml ${COKERNEL_MODULES}
source "$COKERNEL_VENV/bin/activate"
cd "$COKERNEL_REPO"
exec ./scripts/probe_capabilities.sh
