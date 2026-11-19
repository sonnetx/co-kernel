# Co-location inverts the ranking of agent-generated GPU kernels: a preliminary report

Status. The experiment in this note was run once and is not being extended. The harness is meant to be reused: bring your own kernels, or your own neighbor. The short public version is at https://sonnetx.github.io/co-kernel/.

## Abstract

Kernel-generation benchmarks rank candidate GPU kernels by latency measured on an otherwise idle device, and autotuners select kernels the same way. Production accelerators are shared. We ask whether an isolated ranking survives co-location, and we measure it directly by running agent-generated CUDA kernels alongside a synthetic co-runner whose register footprint we control. On one NVIDIA H100, two independently verified kernels for the same matrix-vector problem reverse order under co-location. The kernel that is 23.8% faster in isolation becomes 9.6% slower when a neighbor holds half the register file, because the two kernels degrade by different factors (2.15x against 1.58x). We derive the condition under which a pair inverts, showing that inversion requires the isolated-faster kernel to be the more fragile one by a margin exceeding its isolated advantage, and we use the condition to explain why an earlier corpus of near-identical kernel pairs produced no inversions. We report one inversion across five rankable tasks, which establishes that isolated ranking is not a property of a kernel alone, and we state explicitly what a single synthetic-neighbor experiment does not establish.

## 1. Introduction

Agent-generated GPU kernels are evaluated by measuring the latency of one kernel on one idle device, then ranking candidates by that measurement. Every public kernel-generation benchmark we examined adopts the procedure, and the practice extends to autotuners, which select among candidate implementations by timing each in isolation.

The procedure matters because kernel selection is consequential and increasingly automated. A benchmark that ranks kernels wrongly propagates the error into every system that trusts the ranking, and a kernel chosen by an autotuner is deployed without a human reconsidering the choice. Deployed inference accelerators are rarely dedicated to a single kernel. Operators co-locate work to raise utilization, whether by multiplexing tenants onto one device, by overlapping prefill and decode phases, or by scheduling best-effort batch work beside latency-sensitive serving.

Predicting the effect of co-location from an isolated measurement is difficult because the quantity that changes is not the kernel's instruction stream but the resources available to execute it. A streaming multiprocessor admits a block only when its register, shared-memory, and thread-slot requirements all fit in what remains after resident work is accounted for. A neighbor that consumes registers therefore changes how many of a kernel's blocks can be resident, and the resulting slowdown depends on the kernel's own resource profile rather than on any property visible in its isolated latency. Two kernels with almost identical isolated latency can differ substantially in how much they lose.

Prior work on GPU spatial sharing establishes that co-located kernels interfere and develops mechanisms and metrics for scheduling them, and the vendor tooling for concurrent execution and hard partitioning is mature. We are not aware of work that asks whether the interference is strong enough to invert the ranking of candidate kernels produced for the same problem, which is the question a benchmark or an autotuner implicitly answers. We owe this section a full citation pass and have not yet completed one, so we position the contribution in general terms rather than against specific results.

We measure the effect directly on one H100 by co-executing agent-generated kernels with a synthetic co-runner that sweeps register consumption while holding thread count fixed. Section 2 defines the quantities we report and derives the inversion condition. Section 3 describes the corpus, the co-runner, and the timing methodology, including two harness defects that produced confidently wrong numbers before we found them. Section 4 reports the measurements and the inversion. Section 5 offers a mechanism hypothesis and shows where the hypothesis fails to predict our own data. Section 6 states the limitations, of which the most consequential is that our neighbor is synthetic rather than a real workload.

Summary of contributions:

1. A measurement harness that co-executes a candidate kernel with a resource-parameterized co-runner in one process, re-verifies candidate correctness under co-location, and distinguishes co-located slowdown from starvation (Section 3).
2. The inversion condition relating fragility ratio to isolated margin, which predicts which candidate pairs can invert and explains a null result on near-identical pairs (Section 2.2, Section 4.3).
3. An observed top-1 inversion on agent-generated kernels for a matrix-vector task, with non-overlapping latency distributions (Section 4.2).
4. Two harness defects worth publishing because each produced plausible, self-consistent, and wrong measurements (Section 3.4).

## 2. Definitions

### 2.1 Quantities

A *task* is a problem specification with a reference implementation in PyTorch. A *candidate* k is a CUDA kernel that computes the task's output and passes correctness verification against the reference. A *co-runner configuration* c specifies the workload sharing the device, and c = 0 denotes an otherwise idle device.

We write T_iso(k) for the median latency of candidate k measured on an idle device, and T_co(k, c) for the median latency of the same candidate measured while configuration c occupies the device. The *fragility* of candidate k under configuration c is

    F(k, c) = T_co(k, c) / T_iso(k)

so that F = 1 denotes a candidate unaffected by the neighbor and larger values denote greater degradation.

A *top-1 inversion* occurs for a task when the candidate minimizing T_iso differs from the candidate minimizing T_co under a given configuration. Inversion is the quantity of interest because selecting the single fastest candidate is what a leaderboard reports and what an autotuner does.

### 2.2 When a pair inverts

Consider two candidates a and b for one task, with a faster in isolation, so T_iso(a) < T_iso(b). Candidate a loses its lead under configuration c when T_co(a, c) > T_co(b, c). Substituting T_co(k, c) = F(k, c) T_iso(k) and rearranging gives

    F(a, c) / F(b, c)  >  T_iso(b) / T_iso(a)

The left side is the ratio of fragilities and the right side is the isolated margin, which is greater than one by construction. Three consequences follow from the inequality.

Inversion requires the isolated-faster candidate to be the more fragile one, since otherwise the left side does not exceed one while the right side does. Inversion becomes easier as the isolated margin narrows, because a near-tie needs only a small fragility difference to overturn it. And a large uniform slowdown, however dramatic, never inverts anything, because equal fragilities leave the left side at one.

The third consequence is why we report fragility per candidate rather than a single aggregate slowdown. A measurement showing that co-location halves throughput says nothing about whether rankings hold.

## 3. Method

### 3.1 Corpus

We draw candidates from the AI CUDA Engineer archive, a public release of roughly 30,000 CUDA kernels that an agent generated for KernelBench tasks, each row carrying kernel source, the PyTorch reference module, a correctness flag, and a recorded runtime. The level-1 split contains 12,157 kernels, of which 7,222 carry a correct flag.

Selecting the two fastest correct kernels per task, which is the obvious construction, produces a corpus unsuited to the question. Agents converge on a single strategy, so the fastest candidates are near-duplicates in source and differ by under one percent in recorded runtime. By the inversion condition such pairs can invert, since their margin is small, but only if their fragilities differ, and near-identical source gives little reason to expect that.

We therefore select, for each task, the fastest correct kernel together with the correct kernel that is most dissimilar in source among those within 1.5x of the fastest runtime. Dissimilarity is one minus the ratio reported by a line-level sequence matcher, and candidates above 0.95 similarity to an already-selected kernel are rejected as duplicates. The construction keeps the isolated margin narrow while admitting genuinely different implementations. Across six tasks the resulting pairs have source similarity between 0.16 and 0.41, against 0.23 to 0.99 for top-two-by-runtime selection on the same tasks.

We restrict tasks to those whose reference takes no constructor arguments, which makes the candidate calling convention unambiguous and excludes convolutions.

### 3.2 Co-runner

Our co-runner is synthetic, and Section 6 treats the choice as the experiment's principal limitation. It launches one persistent block per streaming multiprocessor, 132 blocks on our device, each of 256 threads, and each thread spins on a chain of fused multiply-add operations over NREG live accumulators until the host raises a stop flag. The loop carries a dependence on every accumulator, which keeps all NREG values live and makes register consumption track NREG.

We instantiate the kernel at three settings and read the achieved register count from `cudaFuncGetAttributes` rather than assuming it, obtaining 40, 127, and 255 registers per thread. The device provides 65,536 registers per multiprocessor. Thread count stays at 256 across all three settings, so instruction issue pressure and thread-slot occupancy hold constant and comparisons between settings isolate the register axis. Comparisons between an idle device and any co-runner setting do not isolate that axis, because the presence of the neighbor changes issue pressure and cache occupancy as well.

The stop flag lives in device memory. The host raises it with an asynchronous copy from pinned memory on a dedicated non-blocking stream, so the copy engine delivers the flag even when every multiprocessor is occupied. Section 3.4 explains why the design matters.

### 3.3 Co-execution and timing

Candidate and co-runner share one process and one device. The candidate executes on the legacy default stream, where the archive's kernels launch by default, and the co-runner executes on a stream created with the non-blocking flag. The legacy default stream does not synchronize against non-blocking streams, so the two co-execute on the same multiprocessors rather than time-slicing.

Each candidate is compiled ahead of time. Before timing, we verify correctness in the measuring process over three trials with seeds the archive did not use, comparing against the reference with absolute and relative tolerance 1e-2. In-process verification is necessary because the archive's screening has documented gaps and because our co-execution setup relaxes assumptions the archive's harness made.

We measure 100 iterations after 10 warmup iterations, recording CUDA events on the candidate stream and zeroing a 256 MB buffer before each timed iteration to flush the last-level cache, following the convention of the established kernel benchmarks. After starting the co-runner we wait 250 ms before the first warmup iteration, so the entire measurement window falls inside the co-runner's steady state. We record wall-clock time for each pass beside the event sum, which exposes any candidate whose work escapes the timed stream.

A candidate whose blocks cannot become resident makes no progress at all, so a 60-second watchdog stops the co-runner and abandons the pass. Iterations that completed before the watchdog fired are valid co-located samples and we retain them with their count. We report a pass with zero completed iterations as *no progress*, and Section 6 notes that the label does not distinguish true starvation from extreme slowdown.

Each candidate runs in a separate subprocess under a hard wall-clock budget, because a candidate that hangs the CUDA context cannot be recovered from inside the measuring process.

### 3.4 Harness validation

Two defects produced self-consistent and wrong measurements, and we report them because each would be easy to repeat.

Our first co-runner polled its stop flag in mapped host memory. Approximately 33,000 threads reading a zero-copy flag every few microseconds saturated the host interconnect, and the resulting interference slowed every candidate by a factor between 500 and 770 regardless of candidate type or register setting. The uniformity is the diagnostic signature, since genuine register contention should depend on both. Moving the flag to device memory removed the artifact and revealed the candidate-specific structure that Section 4 reports. A co-runner requires the same scrutiny as the kernels under test, because an apparatus that perturbs the shared resource cannot be distinguished from the effect it measures.

Our first timing loop synchronized with `torch.cuda.synchronize()`, which waits on every stream including the one carrying a co-runner that stops only when instructed, and therefore deadlocks. Timing must synchronize on an event recorded on the candidate stream.

We validate the corrected harness three ways. Isolated medians reproduce the archive's recorded runtimes within a few percent across all candidates. After a later refactor that moved co-runner control behind an interface, isolated medians for the square-matmul candidates reproduced to 1088.4 us and 1140.5 us against 1090.0 us and 1140.7 us measured before it, agreeing to 0.15% on a different day, which bounds run-to-run variation in the isolated measurement even though it does not speak to the co-located one. Multiprocessor clocks, sampled through NVML before and after every pass, held at 1830 MHz throughout with persistence mode enabled, against a maximum of 1980 MHz, so we can exclude thermal and frequency effects at these durations without locking clocks, which we lack the privileges to do.

## 4. Results

### 4.1 Fragility across tasks

Table 1 reports isolated latency and fragility for every candidate that built and verified. We name the two matrix-vector candidates by their launch strategy, since the distinction carries the result.

Table 1. Isolated latency and fragility by register setting. Entries reading *np* completed no iterations within the watchdog budget.

| Task | Candidate | T_iso (us) | F at 40 | F at 127 | F at 255 |
|---|---|---|---|---|---|
| Matrix-vector | 512-thread scalar | 66.6 | 1.16 | 2.15 | np |
| Matrix-vector | 256-thread vectorized | 82.4 | 1.13 | 1.58 | np |
| Softmax | fastest | 12.5 | 1.93 | 3.95 | np |
| Softmax | diverse | 15.4 | 2.05 | 4.44 | np |
| Batched matmul | fastest | 539.6 | 4.11 | 9.83 | np |
| Transposed matmul | fastest | 2472.3 | 2.58 | 6.51 | np |
| Transposed matmul | diverse | 2981.7 | np | np | np |
| Square matmul | fastest | 1090.0 | np | np | np |
| Square matmul | diverse | 1140.7 | np | np | np |

Fragility ranges from 1.13 to 9.83 and varies with what the candidate computes. The compute-bound batched matmul loses nearly an order of magnitude to a neighbor that holds registers, while the bandwidth-bound matrix-vector candidates, which are already register-light, lose under a factor of two at the same setting. Every candidate makes no progress at 255 registers per thread, where the co-runner consumes 65,280 of the device's 65,536 registers per multiprocessor and leaves nothing for a candidate block.

The second batched-matmul candidate hung its CUDA context and its subprocess budget expired, so we exclude the task from ranking. Both candidates for a Kullback-Leibler divergence task failed in-process verification, returning shape [1] where the reference returns a scalar, despite carrying the archive's correct flag. We treat the discrepancy as a corpus-hygiene finding rather than a co-location result.

### 4.2 An observed inversion

The matrix-vector task inverts at 127 registers per thread. Table 2 gives the latencies and the percentile spread.

Table 2. Matrix-vector candidates, isolated and at 127 registers per thread. Percentiles are the 10th and 90th over 100 iterations.

| Candidate | T_iso (us) | p10 to p90 | T_co (us) | p10 to p90 |
|---|---|---|---|---|
| 512-thread scalar | 66.56 | 65.44 to 67.52 | 143.02 | 142.14 to 143.49 |
| 256-thread vectorized | 82.43 | 81.44 to 83.17 | 130.50 | 129.73 to 131.33 |

The scalar candidate leads by 23.8% in isolation and trails by 9.6% under co-location. The co-located distributions do not overlap between their 10th and 90th percentiles, both passes completed all 100 iterations without watchdog intervention, and both candidates passed verification in the same process minutes earlier. At 40 registers per thread the isolated order still holds, with the scalar candidate at 77 us against 93 us, so the inversion appears as register pressure rises between the two settings.

### 4.3 The condition, applied

For the matrix-vector pair at 127 registers, the fragility ratio is 2.15 / 1.58 = 1.36 and the isolated margin is 82.43 / 66.56 = 1.24. The inequality of Section 2.2 holds, and the order reverses.

An earlier pass using top-two-by-runtime selection produced no inversion on any task. The condition explains the null. In those pairs the isolated-faster candidate was also the less fragile one at every setting, so the left side of the inequality fell below one while the right side exceeded it, and co-location widened every margin instead of closing any. Selecting for source diversity within a runtime band, rather than for runtime alone, is what produced a pair whose fragility ordering disagrees with its isolated ordering.

## 5. Mechanism

The launch configurations of the two matrix-vector candidates suggest an explanation. The fragile candidate runs 512 threads per block and accumulates with scalar loads unrolled four times. The robust candidate runs 256 threads per block and accumulates with 128-bit vector loads through the read-only data path, reaching comparable bandwidth with half the threads.

Register demand per block scales with threads per block. A co-runner at 127 registers per thread over 256 threads holds 32,512 registers, leaving roughly 33,000 of the multiprocessor's 65,536. A 512-thread block becomes resident only if it needs fewer than 64 registers per thread, while a 256-thread block tolerates up to 129. The smaller block carries roughly twice the per-thread headroom, which would account both for its lower fragility and for the inversion appearing between the 40-register and 127-register settings rather than at the lowest setting.

We have not confirmed the account, and our own data contradicts the simple version of it. The square-matmul candidates make no progress even at 40 registers per thread, where the co-runner leaves roughly 55,000 registers and the candidates launch 256-thread blocks, a configuration the residency arithmetic predicts should fit comfortably. Something beyond per-block register headroom governs that case and we have not identified it. A co-runner that holds registers also consumes issue slots and last-level cache, and separating those contributions requires hardware counters, which we did not collect, together with a hard-partitioned control condition, which we did not run. What we can state from measurement is that the inversion tracks the co-runner's register footprint while thread count and instruction mix hold constant.

## 6. Limitations

Our neighbor is synthetic. A kernel that spins on register-resident arithmetic isolates one variable cleanly, which is why we chose it, and it is not a workload any operator runs. We do not measure what a real tenant does, and we do not report a deployment number.

We observe one inversion across five rankable tasks, which measures the existence of the effect and not its rate. We ran the inverting configuration once, so we do not claim reproducibility and a repeat is required before anyone should.

We do not identify the mechanism at the level of hardware counters, and Section 5 documents a case our mechanism hypothesis fails to predict.

We do not test the Multi-Process Service, so we have not measured candidates sharing a single CUDA context in the manner of a genuine multi-tenant deployment. We do not test MIG partitioning, which would separate multiprocessor-level contention from memory-bandwidth contention by hard-partitioning both and which is the control that would falsify a register-contention account.

We measure fragility only. Our co-runner carries no progress counter, so we do not measure how much a candidate costs its neighbor, which is the quantity an operator co-scheduling a latency-sensitive tenant actually cares about.

Our *no progress* label does not distinguish a candidate whose blocks never become resident from one slowed past roughly 5000x, because neither completes an iteration inside the watchdog budget.

We do not lock multiprocessor clocks, lacking the privileges, and instead sample them to show they did not vary.

We restrict the corpus to tasks with argument-free references and to one kernel archive, on one device, on one node.

## 7. Conclusion and future work

Co-located register pressure reversed the top-1 ranking of two verified agent-generated kernels for the same matrix-vector problem, turning a 23.8% isolated advantage into a 9.6% deficit through a fragility difference of 2.15x against 1.58x. Isolated ranking is therefore not a property of a candidate kernel alone, and a benchmark or autotuner that selects on isolated latency can select the slower kernel for a shared device. The inversion condition of Section 2.2 identifies which candidate pairs are exposed, namely those whose isolated margin is narrower than the ratio of their fragilities, and it accounts for our own earlier null result on near-identical pairs.

We are extending the experiment in four directions. We are adding a Multi-Process Service configuration to obtain a deployment-representative number and a MIG configuration as a falsification control, since a register-contention account predicts the effect largely vanishes under hard partitioning while a bandwidth account predicts it persists. We are instrumenting the co-runner with a progress counter to measure aggression alongside fragility, and adopting the system-throughput and normalized-turnaround metrics of the co-scheduling literature so our numbers compare against prior work. We are replacing the synthetic neighbor with real workloads, specifically decode, prefill, and best-effort batch inference. And we are profiling every candidate with hardware counters to test whether register footprint and wave quantization predict fragility across a corpus large enough to regress on.
