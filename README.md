# co-kernel

Finished note. The short public version is [The faster kernel was slower](https://sonnetx.github.io/co-kernel/). This repository is the measurement record from one H100 on Sherlock. It is not being extended.

Do GPU kernel rankings survive co-location?

Kernel-generation benchmarks rank candidate kernels by latency on an idle
device, and autotuners select kernels the same way. Production accelerators are
shared. This harness measures what happens to a ranking when something else is
using the GPU.

On one H100 it found an inversion. Two verified agent-generated kernels for the
same matrix-vector problem swap places under co-location: the one that is 23.8%
faster in isolation ends up 9.6% slower when a neighbor holds half the register
file, because the two degrade by different factors (2.15x against 1.58x). The
writeup is in [results/day_one_report.md](results/day_one_report.md).

## What it does

For each candidate kernel, the harness measures latency on an idle device and
then under a co-runner it controls, re-verifying correctness in the measuring
process first. It reports per-candidate *fragility*, the ratio of co-located to
isolated latency, and per-task *inversions*, meaning cases where the isolated
winner is not the co-located winner.

The default co-runner is synthetic, one persistent block per multiprocessor
spinning on register-resident arithmetic, with register and shared-memory
knobs. Its purpose is isolating a resource axis rather than representing a
workload, and the interface accepts any neighbor you want to supply instead.

## Requirements

An NVIDIA GPU, a CUDA toolkit, and a PyTorch build matching it. Install torch
for your CUDA version first, since a mismatch surfaces as a compile error when
the harness builds its extensions rather than at install time.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your toolkit
pip install -e '.[corpus,clocks]'
```

Tested on H100 (sm_90) with CUDA 12.4 and torch 2.5.1. Nothing is
architecture-specific beyond the compile target, which `run_local.sh` reads
from the driver.

Check an install with `./scripts/validate.sh`, which compiles every extension
and then measures two kernels. The compile step needs a CUDA toolkit but no
GPU, so `--build-only` works on a login node or a CPU-only machine.

## Quickstart

```bash
python scripts/prepare_corpus.py --out corpus   # needs network, one time
./scripts/run_local.sh --quick                  # ~2 min smoke test
./scripts/run_local.sh                          # full sweep, ~30 min
```

`--quick` runs few iterations over two kernels at one pressure level. It
confirms the install works and its timings are not measurements.

Results land in a JSON file carrying, for every candidate and condition, the
latency distribution summary, clock samples, verification outcome, and the
co-runner's own description. The file is rewritten after each candidate, so an
interrupted run keeps everything completed so far.

To run on a cluster, adapt [scripts/slurm/sherlock.sbatch](scripts/slurm/sherlock.sbatch),
which is a site-specific template rather than a portable script.

## Reading the output

```
1_task4_k0: ok in 61s | isolated 66.6 us | r32 1.16x | r96 2.15x | r240 no-progress
...
=== top-1 flips (isolated winner vs co-located winner) ===
1_task4    4_Matrix_vector_multiplication_   r32:same  r96:FLIP  r240:no-rank
```

`r96` names a co-runner level, here 96 live accumulators per thread, which
compiles to 127 registers per thread. Read achieved register counts from the
`corunner.reg_info` field of the results rather than the level number, since
`__launch_bounds__` caps them. `no-progress` means the candidate completed no
iterations inside the watchdog budget, either through genuine starvation or
extreme slowdown. `no-rank` means fewer than two candidates for that task
produced a comparable measurement.

## When a ranking inverts

Writing F(k) for the fragility of candidate k, and taking a faster than b in
isolation, a loses its lead exactly when

    F(a) / F(b)  >  T_iso(b) / T_iso(a)

The fragility ratio has to exceed the isolated margin. Inversion therefore
requires the isolated-faster candidate to be the more fragile one, and it gets
easier as the isolated margin narrows. A large uniform slowdown never inverts
anything.

The condition has a practical consequence for corpus construction. Selecting
the two fastest candidates per task, which is the obvious thing to do, produces
near-duplicate kernels whose fragilities barely differ, and an early run built
that way found no inversions at all. Selecting the fastest candidate plus the
most source-dissimilar candidate within a runtime band of it is what surfaced
the inversion.

## Hacking

| Goal | Where |
|---|---|
| Change the register sweep | one list in `cokernel/csrc/pressure.cu` |
| Sweep shared memory | `--pressure-smem-bytes` |
| Supply your own neighbor | implement `CoRunner` in `cokernel/corunner.py` |
| Supply your own kernels | write a corpus directory, [docs/CORPUS.md](docs/CORPUS.md) |
| Change kernel selection | `cokernel/corpus.py` |
| Add a metric | read the results JSON, do not modify the measurement path |

[docs/EXTENDING.md](docs/EXTENDING.md) covers each of these.
[docs/PITFALLS.md](docs/PITFALLS.md) collects the measurement traps this project
fell into, all of which produced confident wrong numbers, and is worth reading
before building anything similar.

## Layout

```
cokernel/
  csrc/pressure.cu   synthetic co-runner, register and shared-memory knobs
  corunner.py        CoRunner interface, synthetic and null implementations
  harness.py         build, verify, time; watchdog and abort handling
  corpus.py          candidate selection from the public kernel archive
scripts/
  run_local.sh       portable single-GPU entry point
  day_one.py         sweep driver, subprocess isolation, inversion analysis
  measure_one.py     measures one candidate, run as a subprocess
  prebuild.py        compile every extension without a GPU, fails fast
  prepare_corpus.py  build a corpus from the archive
  slurm/             site-specific job templates
docs/                corpus format, extension guide, pitfalls
results/             writeup and archived measurements
```

Each candidate is measured in its own subprocess under a hard timeout, because
a kernel that hangs the CUDA context cannot be recovered from inside the
measuring process, and one such kernel should not cost a whole run.

## Status and limitations

The inversion is an existence proof from a single run, not a rate. The neighbor
is synthetic. There is no Multi-Process Service configuration, so candidates do
not yet share a CUDA context the way a real multi-tenant deployment has them
do, and no MIG configuration, which is the control that would separate
multiprocessor contention from memory-bandwidth contention. Only fragility is
measured, never how much a candidate costs its neighbor. Section 6 of the
[report](results/day_one_report.md) states these in full.

## Data and licensing

Code is Apache-2.0, see [LICENSE](LICENSE). The corpus generator downloads the
AI CUDA Engineer archive, released separately under CC-BY-4.0, and the
`corpus/` directory in this repository holds kernels derived from it. That
archive's correctness screening has documented gaps, which is why this harness
re-verifies every candidate before timing and why two candidates carrying an
upstream correct flag fail verification here.
