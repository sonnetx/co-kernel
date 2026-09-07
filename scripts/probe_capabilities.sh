#!/usr/bin/env bash
# Report which co-location mechanisms and profiling tools this node permits.
#
# Three of the planned experiments need capabilities a shared cluster may
# withhold, and each fails in a different, confusing way when absent. Probing
# first is cheaper than debugging a job that died for a permissions reason.
#
#   ./scripts/probe_capabilities.sh          # on a GPU node
#   sbatch scripts/slurm/probe.sbatch        # via the queue
set -uo pipefail

say() { printf '\n=== %s ===\n' "$1"; }
verdict() { printf '  VERDICT %-22s %s\n' "$1" "$2"; }

say "device"
nvidia-smi --query-gpu=name,compute_cap,compute_mode,mig.mode.current,persistence_mode,driver_version \
  --format=csv 2>&1 || echo "  nvidia-smi unavailable"

say "ncu (hardware counters, needed for the predictor regression)"
if command -v ncu >/dev/null 2>&1; then
  echo "  found: $(command -v ncu)"
  ncu --version 2>&1 | head -2
  # Profiling counters require either root or the driver module loaded with
  # NVreg_RestrictProfilingToAdminUsers=0. Test with a trivial kernel rather
  # than trusting the flag, since the failure is a runtime permission error.
  cat > /tmp/ncu_probe_$$.py <<'PY'
import torch
a = torch.randn(1024, 1024, device="cuda")
for _ in range(3):
    (a @ a).sum().item()
PY
  if ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
         --target-processes all --launch-count 1 \
         python /tmp/ncu_probe_$$.py > /tmp/ncu_out_$$.txt 2>&1; then
    verdict "ncu-counters" "AVAILABLE"
  else
    echo "  --- ncu output (tail) ---"
    tail -12 /tmp/ncu_out_$$.txt | sed 's/^/  /'
    if grep -qi "permission\|ERR_NVGPUCTRPERM\|administrator" /tmp/ncu_out_$$.txt; then
      verdict "ncu-counters" "BLOCKED (needs NVreg_RestrictProfilingToAdminUsers=0; ask admins)"
    else
      verdict "ncu-counters" "FAILED (see output above)"
    fi
  fi
  rm -f /tmp/ncu_probe_$$.py /tmp/ncu_out_$$.txt
else
  verdict "ncu-counters" "ABSENT (try: ml cuda; or module load nsight-compute)"
fi

say "MPS (shared CUDA context, the deployment-representative arm)"
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "  found: $(command -v nvidia-cuda-mps-control)"
  export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/mps_pipe_$$"
  export CUDA_MPS_LOG_DIRECTORY="${TMPDIR:-/tmp}/mps_log_$$"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if nvidia-cuda-mps-control -d > /tmp/mps_start_$$.txt 2>&1; then
    sleep 2
    if echo get_default_active_thread_percentage | nvidia-cuda-mps-control 2>&1 | grep -qE '^[0-9]'; then
      verdict "mps" "AVAILABLE (daemon started as user)"
    else
      verdict "mps" "PARTIAL (daemon started, control pipe unresponsive)"
    fi
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  else
    tail -6 /tmp/mps_start_$$.txt | sed 's/^/  /'
    verdict "mps" "BLOCKED (daemon would not start; GPU may need EXCLUSIVE_PROCESS or admin help)"
  fi
  rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" /tmp/mps_start_$$.txt
else
  verdict "mps" "ABSENT (nvidia-cuda-mps-control not on PATH)"
fi

say "MIG (hard partitioning, the falsification control)"
mig_mode=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
echo "  mig.mode.current = ${mig_mode:-unknown}"
case "$mig_mode" in
  Enabled)
    nvidia-smi mig -lgi 2>&1 | sed 's/^/  /' | head -12
    verdict "mig" "ENABLED (enumerate instances above; usable if any are allocated to this job)"
    ;;
  Disabled)
    verdict "mig" "DISABLED (enabling needs root and drains the GPU; requires admin request)"
    ;;
  *)
    verdict "mig" "UNSUPPORTED-OR-UNKNOWN on this device"
    ;;
esac

say "multi-process (no MPS): always available as a comparison arm"
verdict "multi-process" "AVAILABLE (time-sliced contexts; expect L2/HBM interference only)"

say "summary"
echo "  Arms that need admin action are reported BLOCKED above. The two-stream"
echo "  arm and plain multi-process need nothing and always work."
