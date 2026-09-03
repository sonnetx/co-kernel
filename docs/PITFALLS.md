# Measurement pitfalls

Every item here produced confident, self-consistent, wrong numbers during this
project. They are recorded because anyone building a co-location harness will
meet them.

## A co-runner that polls host memory destroys the experiment

The first synthetic co-runner polled its stop flag in mapped host memory
(`cudaHostAllocMapped`). Roughly 33,000 threads reading a zero-copy flag every
few microseconds saturated the host interconnect. Every candidate slowed by a
factor between 500 and 770, independent of candidate type and independent of
register level.

The uniformity is the tell. Genuine register contention depends on both the
candidate's resource profile and the pressure level, so a flat multiplier
across a heterogeneous corpus indicates the apparatus, not the effect. Keep the
flag in device memory and raise it with an asynchronous copy on a dedicated
stream.

The general form of the mistake is that a measurement apparatus which perturbs
the shared resource cannot be distinguished from the effect it measures. A
co-runner needs the same scrutiny as the kernels under test.

## Device-wide synchronization deadlocks against a persistent neighbor

`torch.cuda.synchronize()` is `cudaDeviceSynchronize`, which waits on every
stream, including the one carrying a co-runner that stops only when instructed.
Timing must synchronize on an event recorded on the candidate stream:

```python
end = torch.cuda.Event(enable_timing=True)
end.record()
end.synchronize()          # candidate stream only
```

## Stopping the neighbor must not need the main thread

When a candidate is starved, the main thread blocks inside a CUDA synchronize
that returns only after the neighbor releases the device. A watchdog that stops
the neighbor by queueing work behind the main thread therefore deadlocks. The
copy engine executes independently of multiprocessor occupancy, which is why
the stop flag travels by `cudaMemcpyAsync` on its own stream.

## Starvation is a real regime, not an error

A neighbor holding more registers than the multiprocessor can spare denies
residency to a candidate's blocks entirely, and the candidate makes no progress
for as long as the neighbor runs. The watchdog stops the neighbor and records
the pass. Iterations that completed before the watchdog fired are valid
co-located samples and are retained with their count.

A pass with zero completed iterations is reported as no progress. The label
does not distinguish true starvation, where blocks never become resident, from
a slowdown past roughly the watchdog budget divided by the isolated latency.
Separating the two needs occupancy counters.

## Some candidates hang the context beyond any in-process rescue

A candidate that creates its own CUDA streams and blocks on them inside its
`forward` call can hang in a way no in-process watchdog reaches, because
control never returns to the timing loop to check an abort flag. Measuring each
candidate in a subprocess under a hard wall-clock budget bounds the damage to
one candidate.

## Correctness flags from upstream corpora are not sufficient

Two candidates carrying a correct flag from their source archive failed
in-process verification, returning shape `[1]` where the reference returns a
scalar. Re-verify in the measuring process, with seeds the source did not use,
particularly when your harness relaxes an assumption the source made. This
harness permits multiple streams, which some benchmark harnesses forbid as an
anti-gaming measure, so their verification does not transfer.

## Clocks

Co-location raises power draw and can trigger frequency throttling, which is a
real deployment effect and a different one from resource contention. Conflating
them is avoidable. Lock clocks where you have the privileges. Where you do not,
sample them through NVML before and after every pass and report the samples, so
a frequency-confounded pass is at least visible. Our measurements held at
1830 MHz throughout with persistence mode enabled.

## Cache state

Flush the last-level cache before every timed iteration, by zeroing a buffer
larger than it, so isolated numbers stay comparable against benchmarks that do
the same. Under co-location the neighbor is already disturbing the cache, which
makes the flush partly redundant, but dropping it would break comparability
with the isolated baseline.

## Steady state

Start the neighbor, wait, and only then measure. A window that straddles the
neighbor's ramp-up averages two different regimes and reports neither.

## Extension names must be C identifiers

`TORCH_EXTENSION_NAME` is substituted into a macro, so a slug containing a dot
produces a syntax error deep in generated pybind code rather than a clear
failure. A float-valued identifier column read from a parquet file is an easy
way to acquire one, since `1` becomes `1.0`.
