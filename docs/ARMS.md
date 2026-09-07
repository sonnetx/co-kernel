# Sharing-mechanism arms

The question "does the ranking survive co-location" has a different answer per
sharing mechanism, and the mechanisms disagree in exactly the ways that
identify the cause. The harness runs the same corpus under each.

## two-stream (in-process)

One process, candidate on the legacy default stream, neighbor on a non-blocking
stream, co-executing on the same multiprocessors. The cleanest mechanism probe,
with full control of launch timing. Not how deployments share GPUs.

    python scripts/day_one.py --corunner synthetic

## multi-process, no MPS

Neighbor in its own process, so its own CUDA context. Without MPS the driver
time-slices contexts, so kernels from the two processes do not co-reside on
SMs. Expect L2/HBM disturbance and context-switch cost only. Serves as a
negative control: SM-level effects (register starvation, the inversion) should
largely vanish here, and a fragility that survives time-slicing is a memory
system effect, not an occupancy effect.

    python scripts/day_one.py --corunner proc-pressure

## multi-process under MPS

Same two processes, but with the MPS daemon up and both inheriting
`CUDA_MPS_PIPE_DIRECTORY`, they share one context and genuinely co-execute.
The deployment-representative arm, and the number to quote. Starting the daemon
needs no privileges on nodes that allow it; the probe job checks.

`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` caps the SM fraction an MPS client may use,
which is the mitigation axis: sweeping it down measures how much of the effect
an operator can buy back with stock tooling.

All three of the above run in one job:

    sbatch scripts/slurm/mechanism_arms.sbatch

## MIG (requires admin action)

MIG hard-partitions multiprocessors and L2 slices. It is the falsification
control: an SM-contention account predicts the effect largely disappears when
candidate and neighbor sit in separate MIG instances, while a memory-bandwidth
account predicts it persists. Pre-registering that prediction is the point of
the arm.

Creating instances needs root (`nvidia-smi -mig 1`, then `nvidia-smi mig -cgi`)
and draining the GPU, so on a shared cluster it is an administrator request.
What to ask for, concretely: one H100 with MIG mode enabled and two `3g.40gb`
instances, allocatable to one job, for a few hours. Inside the job, place the
candidate in one instance and the neighbor in the other by setting
`CUDA_VISIBLE_DEVICES` to each instance's MIG UUID (from `nvidia-smi -L`) in
the respective process. `ProcessCoRunner` already runs the neighbor in its own
process, so the only change is the two `CUDA_VISIBLE_DEVICES` values.

## Neighbor kinds

Independent of the mechanism, `--corunner` selects what the neighbor computes:

| kind | shape | level unit |
|---|---|---|
| `synthetic` / `proc-pressure` | register-resident FMA spin | live accumulators per thread |
| `proc-gemm` | back-to-back large bf16 matmuls, compute-bound, prefill-shaped | matrix dimension |
| `proc-stream` | large elementwise multiplies, bandwidth-bound, decode-shaped | buffer MiB |

The gemm and stream kinds are torch-level stand-ins for real workloads. A
genuine serving engine slots in through the same `CoRunner` protocol (see
EXTENDING.md), with `start` returning once the server reports steady state and
`progress` reading its request counter; a containerized engine works the same
way since the protocol only touches files and process control.

## Aggression

Every arm reports fragility (how much the candidate suffers). Arms whose
neighbor reports progress also record aggression: the neighbor's solo work rate
divided by its rate while the candidate runs. Fragility and aggression together
give the standard co-scheduling metrics (system throughput, normalized
turnaround) without further measurement. Nothing in any isolated leaderboard
penalizes a kernel for aggression, which is why agent-generated kernels are
worth measuring on it.
