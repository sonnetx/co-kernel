#!/bin/bash -l
# The mechanism comparison: one node, three sharing mechanisms, same corpus.
#
#   two-stream      one process, co-execution on the same SMs (the day-one arm)
#   multi-process   two CUDA contexts, no MPS: time-sliced, so SM-level
#                   contention should largely vanish and only L2/HBM and
#                   context-switch effects remain
#   MPS             two processes sharing one context via the MPS daemon:
#                   genuine co-execution, the deployment-representative arm
#
# If the effect is SM-level resource contention, fragility should be large
# under two-stream and MPS and small under plain multi-process. Skips the MPS
# arm with a clear note when the daemon cannot start (see the probe job).
#
#SBATCH --job-name=cokernel_mech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail

: "${COKERNEL_MODULES:=python/3.12.1 cuda/12.4.0 gcc/12.4.0}"
: "${COKERNEL_VENV:=/scratch/groups/roxanad/sonnet/co-kernel/venv}"
: "${COKERNEL_SCRATCH:=/scratch/groups/roxanad/sonnet/co-kernel}"
: "${COKERNEL_REPO:=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"

ml ${COKERNEL_MODULES}

export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=9.0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$COKERNEL_SCRATCH/tmp"
export COKERNEL_BUILD="$COKERNEL_SCRATCH/build/mech_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR" "$COKERNEL_BUILD"

source "$COKERNEL_VENV/bin/activate"
cd "$COKERNEL_REPO"
J="${SLURM_JOB_ID:-manual}"

nvidia-smi --query-gpu=name,compute_cap,compute_mode,clocks.sm,persistence_mode --format=csv

echo "=== arm 1/3: two-stream (in-process) ==="
python scripts/day_one.py --corunner synthetic \
  --out "results/mech_twostream_${J}.json" "$@"

echo "=== arm 2/3: multi-process, no MPS ==="
python scripts/day_one.py --corunner proc-pressure \
  --out "results/mech_multiproc_${J}.json" "$@"

echo "=== arm 3/3: multi-process under MPS ==="
if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "SKIPPED: nvidia-cuda-mps-control not on PATH"
  exit 0
fi
export CUDA_MPS_PIPE_DIRECTORY="$TMPDIR/mps_pipe_$J"
export CUDA_MPS_LOG_DIRECTORY="$TMPDIR/mps_log_$J"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if ! nvidia-cuda-mps-control -d; then
  echo "SKIPPED: MPS daemon would not start (see probe job for diagnosis)"
  exit 0
fi
sleep 2
trap 'echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true' EXIT
python scripts/day_one.py --corunner proc-pressure \
  --out "results/mech_mps_${J}.json" "$@"
echo "=== all arms complete ==="
