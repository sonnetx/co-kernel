#!/usr/bin/env bash
# Portable single-GPU runner. No cluster or site assumptions.
#
#   ./scripts/run_local.sh              # full sweep
#   ./scripts/run_local.sh --quick      # ~2 minute smoke test
#
# Override any of these from the environment:
#   COKERNEL_BUILD   where compiled extensions land   (default ./build)
#   COKERNEL_OUT     results file                     (default results/day_one_local.json)
#   COKERNEL_CORPUS  corpus directory                 (default ./corpus)
#   TORCH_CUDA_ARCH_LIST  target arch, e.g. 9.0 H100, 8.0 A100, 10.0 B200
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

: "${COKERNEL_BUILD:=$REPO/build}"
: "${COKERNEL_OUT:=$REPO/results/day_one_local.json}"
: "${COKERNEL_CORPUS:=$REPO/corpus}"
export COKERNEL_BUILD

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  # Ask the driver rather than guessing, so this works on any card.
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$cc" ]; then
    export TORCH_CUDA_ARCH_LIST="$cc"
    echo "detected compute capability $cc"
  else
    echo "could not detect compute capability; set TORCH_CUDA_ARCH_LIST yourself" >&2
  fi
fi

mkdir -p "$COKERNEL_BUILD" "$(dirname "$COKERNEL_OUT")"

if [ ! -f "$COKERNEL_CORPUS/manifest.json" ]; then
  echo "no corpus at $COKERNEL_CORPUS" >&2
  echo "build one with: python scripts/prepare_corpus.py --out $COKERNEL_CORPUS" >&2
  echo "or point COKERNEL_CORPUS at your own (format: docs/CORPUS.md)" >&2
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,persistence_mode \
  --format=csv 2>/dev/null || true

exec python scripts/day_one.py \
  --corpus "$COKERNEL_CORPUS" \
  --out "$COKERNEL_OUT" \
  "$@"
