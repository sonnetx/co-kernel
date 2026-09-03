# Co-location can invert GPU kernel rankings

Two agent-generated CUDA kernels solve the same matrix-vector problem. Run alone on an H100, the first finishes in 66.6 microseconds and the second in 82.4, so the first wins by 24 percent. Give the GPU a second tenant that consumes half the register file, and the first takes 143.0 microseconds while the second takes 130.5. The loser is now the winner by 10 percent.

Nothing about either kernel changed. The only thing that changed is what else was running.

That matters because every kernel leaderboard, and every autotuner that picks a kernel by measuring it, measures in isolation. Production GPUs are shared. If the ranking depends on the neighbor, then isolated selection optimizes a quantity that does not survive deployment.

## The measurement

The candidates come from the AI CUDA Engineer archive, roughly 30,000 CUDA kernels that an agent generated for KernelBench tasks. The level-1 split holds 12,157 kernels, 7,222 of them flagged correct. Picking the top two by recorded runtime turns out to be useless for this question, because agents converge on one strategy and the top two are usually near-identical source differing by under one percent in runtime. Instead we take, for each task, the fastest correct kernel plus the most source-dissimilar correct kernel within 1.5x of its runtime. Across six tasks that yields pairs with source similarity between 0.16 and 0.41, close in speed but genuinely different in approach.

The neighbor is synthetic, and this is the most important caveat in the whole experiment. It is not an LLM serving a request. It is one persistent block per SM, 256 threads each, spinning on a chain of fused multiply-adds over a configurable number of live accumulators until the host tells it to stop. The template parameter sets how much of the register file it holds, and `cudaFuncGetAttributes` reports the result rather than us assuming it, at 40, 127, and 255 registers per thread against the H100's budget of 65,536 registers per SM. Thread count stays at 256 across all three levels, so instruction issue pressure and thread-slot occupancy hold constant while the register footprint sweeps.

Candidate and neighbor share one process and one device. The candidate runs on the legacy default stream, where the archive's plain triple-angle-bracket launches land anyway, and the neighbor runs on a stream created with `cudaStreamNonBlocking`. The legacy default stream does not synchronize against non-blocking streams, so the two genuinely co-execute on the same SMs rather than time-slicing.

Timing is 100 iterations after 10 warmups, CUDA events recorded on the candidate stream, a 256 MB buffer zeroed before every timed iteration to flush L2 the way the standard kernel benchmarks do. The neighbor starts, the harness waits 250 milliseconds, and only then does measurement begin, so the whole window sits inside the neighbor's steady state. Every kernel gets re-verified in-process against the PyTorch reference before timing, three trials with fresh seeds, because the archive's own correctness screening has known gaps.

SM clocks sat pinned at 1830 MHz throughout with persistence mode on, so thermal and DVFS effects are not doing the work here.

## What happened

| task | isolated (k0 / k1) | at 40 regs | at 127 regs | at 255 regs |
|---|---|---|---|---|
| matrix-vector | 66.6 / 82.4 us | 1.16x / 1.13x | 2.15x / **1.58x** | no progress |
| softmax | 12.5 / 15.4 us | 1.93x / 2.05x | 3.95x / 4.44x | no progress |
| batched matmul | 540 us | 4.11x | 9.83x | no progress |
| matmul, transposed A | 2472 / 2982 us | 2.58x / none | 6.51x / none | no progress |
| square matmul | 1090 / 1141 us | no progress | no progress | no progress |

Slowdowns span 1.13x to 9.83x, and they track what the kernel is doing. The compute-bound batched matmul loses a factor of ten to a neighbor holding registers. The bandwidth-bound matrix-vector kernel, already register-light, loses under a factor of two at the same pressure.

The matrix-vector row is the result. At 40 registers of neighbor pressure the isolated ranking holds, with k0 at 77 microseconds against k1 at 93. At 127 registers it inverts. The distributions do not overlap, k0 spanning 142.1 to 143.5 microseconds between its 10th and 90th percentiles against k1's 129.7 to 131.3, across a full 100 iterations each with no truncated windows. Both kernels passed correctness verification in the same process minutes earlier.

## The condition for a flip

A flip is not the same as a slowdown, and most slowdowns preserve rankings. Write the fragility of kernel A as F(A), its co-located time divided by its isolated time. Kernel A loses its lead to B exactly when

    F(A) / F(B)  >  T_iso(B) / T_iso(A)

The fragility ratio has to beat the isolated margin. For the matrix-vector pair at 127 registers, fragility gives 2.15 / 1.58 = 1.36 and the isolated margin gives 82.4 / 66.6 = 1.24, so the inequality holds and the order reverses.

This explains why an earlier pass over near-identical top-two pairs found no flips at all. In those pairs the faster kernel was also the more robust one, so co-location widened every gap instead of closing any. A flip needs the isolated winner to be the more fragile kernel, and it needs that fragility gap to exceed the margin it won by. Both conditions are properties of a pair, not of a kernel.

## Why, tentatively

The two matrix-vector kernels differ in a way that suggests an explanation. The fragile winner runs 512 threads per block and accumulates with scalar loads unrolled four times. The robust loser runs 256 threads per block and accumulates with 128-bit `float4` loads through `__ldg`, reaching the same bandwidth with half the threads.

Per-block register demand scales with threads per block. A neighbor at 127 registers per thread over 256 threads holds 32,512 of the SM's 65,536 registers, leaving about 33,000. A 512-thread block fits only if it needs under 64 registers per thread. A 256-thread block fits at up to 129. The smaller block has roughly twice the headroom, which would explain both why it degrades less and why the crossover appears between 40 and 127 registers of pressure rather than at the bottom of the sweep.

We have not confirmed that. The arithmetic is consistent with the timings but we ran no occupancy counters, and the same reasoning mispredicts elsewhere in the table. Square matmul makes no progress even at the lowest pressure, where the register budget should comfortably admit its blocks, so something beyond a simple residency calculation is operating. A neighbor that holds registers also consumes issue slots and L2, and separating those requires profiling counters and a hard-partitioned control, neither of which we have run. The register footprint is the axis we varied and the axis the flip tracks. Naming the microarchitectural pathway takes more work.

One instrumentation limit belongs here too. The cells reading "no progress" mean zero iterations completed inside a 60 second budget. That covers true starvation, where blocks never become resident, and it equally covers a slowdown past roughly 5000x. Our harness cannot currently tell those apart.

## An expensive mistake worth publishing

The first version of this experiment reported a flat slowdown between 500x and 770x for every kernel at every pressure level, with no dependence on kernel type. Uniformity that clean is not a finding, it is a bug.

The neighbor polled its stop flag in mapped host memory. Some 33,000 threads reading a zero-copy flag every few microseconds saturated the PCIe path, and the resulting interference swamped the register contention we meant to measure by two orders of magnitude. Moving the flag into device memory, and having the host raise it with an async copy on a dedicated copy stream, removed the artifact and revealed the kernel-specific structure in the table above.

The general lesson is that a co-runner needs the same scrutiny as the kernels under test, because a measurement apparatus that perturbs the shared resource is indistinguishable from the effect it is trying to observe. Two smaller versions of the same trap sit nearby. Calling `torch.cuda.synchronize()` deadlocks against a persistent neighbor, since it waits on every stream including the one that only stops when told, so timing has to synchronize on an event recorded on the candidate stream. And a kernel that launches internal streams and blocks on them inside its own forward call can hang the context in a way no watchdog inside the process can reach, which is why each kernel now runs in a separate subprocess under a hard timeout.

## What this shows and what it does not

We have an existence proof. Co-located register pressure inverted the ranking of two correct, independently verified, agent-generated kernels for the same problem, at a pressure level a real tenant could plausibly create, with clean non-overlapping distributions. One inversion in six tasks is enough to say that isolated ranking is not a property of a kernel alone.

We do not have a rate. One flip across five rankable tasks measures almost nothing about how often this happens, and we ran the flip once, so it needs a repeat before anyone calls it reproducible. We do not know the mechanism at the level of hardware counters. We tested a synthetic neighbor rather than a real workload, which is the right instrument for isolating a variable and the wrong one for quoting a deployment number. We ran no MPS configuration, so we have not measured the case where processes share a context the way a real multi-tenant deployment does, and no MIG partition, which is the control that would separate SM-level contention from memory-bandwidth contention by hard-partitioning both. We measured only fragility, never aggression, because the neighbor carries no progress counter and cannot report how much the candidate cost it.

Those are the next experiments. The reason to run them is that the cheap version already produced the inversion it was designed to falsify.
