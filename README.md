# co-kernel

Do agent-generated GPU kernels keep their performance rankings under co-location? Kernels are selected and leaderboarded in isolation, but deployed GPUs are shared. If the isolated winner loses when a neighbor occupies registers, shared memory, or bandwidth, then every leaderboard and autotuner that picks kernels in isolation is optimizing the wrong objective.

## Day-one experiment

A falsification check before the full build. Ten agent-generated kernels (5 KernelBench tasks, top-2 correct kernels per task from the [Sakana AI CUDA Engineer archive](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive)), each timed isolated and under a synthetic co-runner at three register-pressure levels, on one H100. The binary question is whether any task's top-1 kernel flips under pressure. One flip justifies the full study. Thirty null results within noise mean the effect is absent at this occupancy regime, and the next check is whether the isolated kernels were occupancy-saturated to begin with.

### Sharing mechanism

Two CUDA streams in one process. The candidate runs on the legacy default stream (archive kernels launch with plain `<<<>>>`, which lands there). The co-runner runs on a `cudaStreamNonBlocking` stream, which the legacy default stream does not synchronize with, so the two co-execute on the same SMs. This is the mechanism experiment; MPS (deployment number) and MIG (falsification control) come later.

### Synthetic co-runner

`cokernel/csrc/pressure.cu`. One persistent block per SM, 256 threads, spinning on an FMA chain over NREG live accumulators until the host raises a zero-copy stop flag. NREG in {32, 96, 168, 240} sweeps the register-file footprint from ~12% to ~94% of the SM's 64K registers while thread count, and with it FMA issue pressure and thread-slot usage, stays fixed. Actual registers per thread are read back with `cudaFuncGetAttributes` and recorded, not assumed.

A pressure level that leaves fewer free registers than a candidate block needs starves the candidate completely, since its blocks queue behind pressure blocks that never retire. A watchdog stops the co-runner after a timeout and the pass is recorded as starved rather than timed. Starvation is data (the kernel cannot make progress next to that neighbor at all), and a starved pass ranks last in the flip analysis.

### Timing

CUDA events around each forward call, 100 iterations, one 256 MB L2 flush before every timed iteration (matching SOL-ExecBench and KernelBench methodology), device synchronize per iteration, warmup excluded. The co-runner launches, then the harness sleeps 250 ms before the first timed iteration so the whole measurement window sits inside the co-runner's steady state. Wall time per pass is recorded next to the event sum, so a kernel whose work escapes the timed stream shows up as a discrepancy.

### Correctness

Every kernel is re-verified in-process before timing (three trials, fresh seeds, `allclose` at the KernelBench tolerance of 1e-2), because the archive's own screening is known to have missed reward hacks and because our streams setup relaxes assumptions the archive's harness made. Kernels that fail verification or fail to build are dropped and reported.

### Known compromises (day-one only)

- Clocks are not locked (no root on the cluster). SM clock is sampled before and after each pass via NVML and recorded, so DVFS-confounded passes are at least visible. The full study needs locked/unlocked runs reported separately.
- Tasks are restricted to parameter-free references (empty `get_init_inputs`), which keeps the candidate calling convention unambiguous but excludes convs and most fused L2 ops.
- Single GPU, single run. No MPS, no MIG control yet.

## Corpus note

Neither SOL-ExecBench nor FastKernels ships usable candidate kernels. SOL-ExecBench (B200-targeted) releases problem definitions only; its leaderboard API exposes rankings and latencies but source downloads require auth. FastKernels releases baselines and a runner with an empty candidate slot. The Sakana archive is the one public corpus of real agent-generated kernels with correctness metadata, and it targets exactly the isolated-benchmark regime this project interrogates. Archive runtimes are used only as a pre-ranking; everything is re-measured here.

## Running

```bash
# anywhere with network (login node): select kernels, write corpus/
python scripts/prepare_corpus.py --out corpus

# on Sherlock
sbatch scripts/slurm_day_one.sh
```

Results land in `results/day_one_<jobid>.json` with per-kernel per-condition timing summaries and a per-task flip analysis.
