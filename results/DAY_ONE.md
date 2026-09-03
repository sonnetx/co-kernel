# Day-one results

The falsification check: do agent-generated kernels keep their performance
rankings under co-location? Ten to twelve kernels from the Sakana AI CUDA
Engineer archive (six KernelBench tasks, source-diverse correct kernels per
task), timed isolated and under a synthetic register-pressure co-runner at
three levels (40, 127, 255 registers per thread), two-stream single-process
co-execution on one H100 80GB.

## Verdict

The effect is real: co-located register pressure inverted a task's top-1
kernel. In the diverse-corpus run (`results/run4_41757499/`), the
matrix-vector task's isolated winner lost its lead under medium pressure. One
flip in a six-task sample justifies the full study.

The mechanism behind it is graded, register-dependent, and kernel-specific
slowdown, with outright starvation at the highest pressure. A flip happens when
the isolated-faster kernel is also the more fragile one and the fragility gap
exceeds the isolated margin.

### The flip (task 4, matrix-vector multiply, r96 = 127 regs/thread)

| kernel | isolated | under r96 | slowdown |
|--------|----------|-----------|----------|
| k0 | **66.6 us** (winner) | 143.0 us | 2.15x |
| k1 | 82.4 us | **130.5 us** (winner) | 1.58x |

Isolated, k0 leads by 24%. Under pressure k0 degrades 2.15x but k1 only 1.58x,
so k1 finishes 10% ahead. The distributions do not overlap (k0 r96 p10-p90
142.1 to 143.5 us; k1 129.7 to 131.3 us), 100 iterations each, no partial
windows. An autotuner or leaderboard selecting k0 in isolation would ship the
slower kernel for any co-located deployment at this pressure.

## The rest of the diverse-corpus run

Run 4 (`results/run4_41757499/`, 12 kernels, subprocess-isolated) held the flip
alongside these:

| task | op | iso k0 | iso k1 | r40 | r127 | r255 |
|------|----|--------|--------|-----|------|------|
| 1 | Square matmul | 1090 us | 1141 us | both starved | both starved | starved |
| 3 | Batched matmul | 540 us | (hung) | 4.1x | 9.8x | starved |
| 23 | Softmax | 12.5 us | 15.4 us | 1.9x / 2.1x | 3.9x / 4.4x | starved |
| 4 | Matrix-vector | 66.6 us | 82.4 us | 1.2x / 1.1x | **flip** | starved |
| 16 | Matmul (A^T) | 2472 us | 2982 us | 2.6x / starved | 6.5x / starved | starved |

Square matmul (task 1) makes no progress even at the lowest pressure, which a
simple residency calculation does not predict for its 256-thread blocks, so
something beyond register headroom is operating there and we have not
identified it. In task 16 the two kernels diverge sharply, k0 survives r40 and
r127 while k1 makes no progress at any level, which is differential fragility
extreme enough that k1 is unrankable rather than merely slower.

A `starved` cell means zero iterations completed inside the 60 s watchdog
budget. That covers both true starvation (blocks never resident) and any
slowdown past roughly 5000x; the harness cannot currently distinguish them. Both KLDiv kernels (task 98) failed
in-harness verification (candidate returns shape [1], reference []); the archive
marked them correct, so the harness shape check is stricter than the archive's,
a corpus-hygiene note for the full study rather than a co-location finding. One
task-3 kernel hung the context and was killed by its subprocess budget without
taking the run down, which is the isolation working as intended.

## What the earlier runs showed

Run 3 (fixed co-runner, near-tied corpus, `results/run3_41734728/`) gave clean,
interpretable numbers but no flip, because top-2-by-runtime kernels differ under
1% isolated and their faster member was also the more robust one:

| task | op | iso k0 | iso k1 | r40 slowdown | r127 slowdown | r255 |
|------|----|--------|--------|--------------|---------------|------|
| 23 | Softmax | 13 us | 12 us | 1.9x | 3.9x | starved |
| 3 | Batched matmul | 539 us | 543 us | 4.1x | 9.8x | starved |
| 4 | Matrix-vector | 67 us | 66 us | 1.2x | 2.1x | starved |
| 16 | Matmul (A^T) | 2473 us | (hung) | 2.6x, 6.5x | | starved |

Slowdown under co-location depends strongly on the kernel's own character: the
compute-bound batched matmul suffers 4 to 10x from register contention, while
the bandwidth-bound matrix-vector kernel, already register-light, loses only 1.2
to 2.1x. In run 3's near-tied pairs the isolated-faster kernel was also the less
fragile one, so co-location widened the gap rather than closing it. Selecting
source-diverse kernels within a runtime band (run 4) rather than near-identical
top-2 exposed the case where fragility and isolated speed disagree, which is the
flip above.

## Harness lessons (documented so the full study inherits them)

The co-runner must be profiled as carefully as the candidates. The first
co-runner polled its stop flag in mapped host memory; 33k threads reading it
flooded the PCIe path and slowed every co-located kernel a flat ~500 to 770x
regardless of type or pressure, a uniform artifact with none of the
kernel-specific structure above (`results/run2_41669671/`). Moving the flag to
device memory, raised by the copy engine, removed it.

`torch.cuda.synchronize()` is device-wide and deadlocks against a persistent
co-runner; timing syncs on a candidate-stream event instead. Register
starvation at 255 regs/thread is genuine (blocks never become resident), handled
by a watchdog that stops the co-runner and keeps whatever iterations completed.
Kernels that launch internal streams and host-block inside `forward()` (task 16
k1) can hang the context uninterruptibly, so each kernel now runs in its own
subprocess under a hard wall timeout and a hang costs one kernel, not the run.

Clocks stayed pinned at 1830 MHz (persistence mode on) across all passes, so
DVFS was not a confound at these durations, and isolated medians reproduced the
archive's runtimes within a few percent.

## Not yet done (full-study scope)

No MPS (deployment number) or MIG (falsification control) arm. Aggression is not
measured (the co-runner has no progress counter, so only fragility). Clocks not
hardware-locked. Single node, one run per condition. Tasks restricted to
parameter-free references.
