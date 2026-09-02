# Day-one results

The falsification check: do agent-generated kernels keep their performance
rankings under co-location? Ten to twelve kernels from the Sakana AI CUDA
Engineer archive (six KernelBench tasks, source-diverse correct kernels per
task), timed isolated and under a synthetic register-pressure co-runner at
three levels (40, 127, 255 registers per thread), two-stream single-process
co-execution on one H100 80GB.

## Verdict

The mechanism is real and measurable: co-location produces graded,
register-dependent, kernel-specific slowdown, and at the highest pressure it
starves candidates outright. Across the kernel pairs that could be ranked, no
task's isolated top-1 kernel lost its lead under pressure (zero top-1 flips).
The full run that motivates a larger study is the diverse-corpus run tracked in
`results/day_one_*.json`; the runs below record how the harness got to a
trustworthy measurement.

## What the runs showed

Run 3 (fixed co-runner, near-tied corpus, `results/run3_41734728/`) gave clean,
interpretable numbers:

| task | op | iso k0 | iso k1 | r40 slowdown | r127 slowdown | r255 |
|------|----|--------|--------|--------------|---------------|------|
| 23 | Softmax | 13 us | 12 us | 1.9x | 3.9x | starved |
| 3 | Batched matmul | 539 us | 543 us | 4.1x | 9.8x | starved |
| 4 | Matrix-vector | 67 us | 66 us | 1.2x | 2.1x | starved |
| 16 | Matmul (A^T) | 2473 us | (hung) | 2.6x, 6.5x | | starved |

Two observations. Slowdown under co-location depends strongly on the kernel's
own character: the compute-bound batched matmul suffers 4 to 10x from register
contention, while the bandwidth-bound matrix-vector kernel, already
register-light, loses only 1.2 to 2.1x. And within each task the isolated-faster
kernel was also the less fragile one, so co-location widened the gap rather than
closing it, which is the opposite of a flip. The near-tied corpus (top-2 by
runtime differ under 1%) is the wrong instrument for finding flips, motivating
the diverse-corpus selection.

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
