#!/usr/bin/env bash
# Confirm an install works: compile every extension, then measure two kernels.
#
#   ./scripts/validate.sh              # compile check, plus smoke test if a GPU is visible
#   ./scripts/validate.sh --build-only # compile check only, no GPU needed
#
# The compile check needs a CUDA toolkit but no GPU, so it runs on a login node
# or a CPU-only box. The smoke test needs a GPU.
#
# Output goes to a log file as well as the terminal, so a dropped connection
# does not lose the result. Run detached to survive one entirely:
#   nohup ./scripts/validate.sh > /dev/null 2>&1 &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

: "${COKERNEL_BUILD:=$REPO/build/validate}"
: "${COKERNEL_CORPUS:=$REPO/corpus}"
: "${COKERNEL_LOG:=$REPO/logs/validate.log}"
mkdir -p "$COKERNEL_BUILD" "$(dirname "$COKERNEL_LOG")"
export COKERNEL_BUILD

exec > >(tee "$COKERNEL_LOG") 2>&1
echo "=== co-kernel validate, $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "build dir: $COKERNEL_BUILD"

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  # No GPU visible means no driver query; a compile check still needs a target.
  export TORCH_CUDA_ARCH_LIST="${cc:-9.0}"
  echo "compile target: $TORCH_CUDA_ARCH_LIST${cc:+ (from driver)}"
fi

echo
echo "--- step 1: compile every extension ---"
if ! python scripts/prebuild.py --build-dir "$COKERNEL_BUILD" --corpus "$COKERNEL_CORPUS"; then
  echo "RESULT: FAIL (compilation)"
  exit 1
fi

if [ "${1:-}" = "--build-only" ]; then
  echo "RESULT: PASS (compile only, smoke test skipped)"
  exit 0
fi

if ! python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo
  echo "RESULT: PASS (compile only); no GPU visible, so the smoke test was skipped."
  echo "Run this on a GPU node to exercise the co-runner."
  exit 0
fi

echo
echo "--- step 2: smoke test on GPU ---"
if ! python scripts/day_one.py \
      --corpus "$COKERNEL_CORPUS" \
      --out "$REPO/results/validate.json" \
      --quick; then
  echo "RESULT: FAIL (smoke test)"
  exit 1
fi

echo
echo "RESULT: PASS"
