# Extending the harness

Four things people most often want to change, in rough order of how easy they
are.

## Change the register sweep

Edit one list in `cokernel/csrc/pressure.cu`:

```cpp
#define COKERNEL_PRESSURE_LEVELS(X) \
  X(32)                             \
  X(96)                             \
  X(168)                            \
  X(240)
```

The launch dispatch and the attribute table are generated from that list, and
Python reads the compiled levels back through `levels()`, so no other file needs
editing. Each level costs one template instantiation of compile time.

Achieved registers per thread are not the same as the level number, because
`__launch_bounds__(256)` caps the count at 255. On H100 the levels above
compile to 40, 127, 255, and 255, which makes the last two redundant. Always
read the achieved counts from `reg_info()`, reported in every results file
under `corunner.reg_info` as `(level, regs/thread, static smem, local bytes)`,
rather than assuming the level number is the register count.

## Sweep shared memory instead of registers

The co-runner takes `smem_bytes`, requested as dynamic shared memory per block,
which constrains occupancy independently of registers.

```bash
python scripts/day_one.py --levels 32 --pressure-smem-bytes 65536
```

Holding the register level fixed while sweeping shared memory separates the two
axes. Requests above 48 KB per block are opted in automatically through
`cudaFuncSetAttribute`. The device limit is per-multiprocessor, so a request
near the maximum leaves nothing for a candidate block and produces uniform
no-progress results, which is a real measurement rather than a bug.

## Bring your own co-runner

The harness knows nothing about what shares the device. Implement the
`CoRunner` protocol in `cokernel/corunner.py` and pass an instance to
`harness.run_conditions`. The protocol is five methods: `levels`, `start`,
`stop`, `describe`, and `label`.

```python
class MyWorkload:
    def levels(self):        return ["batch1", "batch8"]
    def label(self, level):  return f"llama_{level}"
    def describe(self):      return {"kind": "llama_decode", "model": "..."}

    def start(self, level):
        # Begin the workload and return only once it is in steady state.
        self._driver = launch_decode_loop(batch=int(level.lstrip("batch")))
        self._driver.wait_until_steady()

    def stop(self):
        # Called from a watchdog thread while the main thread may be blocked
        # inside a CUDA synchronize. Must not depend on main-thread progress,
        # and must be safe to call twice.
        if getattr(self, "_driver", None):
            self._driver.stop()
            self._driver = None
```

Two requirements are easy to miss and both cost real measurements when ignored.

`start` must return only when the neighbor is genuinely resident and doing
steady-state work, because measurement begins shortly after. The harness waits
250 ms as a courtesy, which suffices for a kernel launch and does not suffice
for a model that still has weights to load.

`stop` must work from another thread without the main thread progressing. When
a candidate is starved, the main thread sits inside a CUDA synchronize that
only returns once the neighbor releases the device, so a `stop` that queues
work behind the main thread deadlocks. The synthetic co-runner satisfies the
requirement by raising a device-memory flag through the copy engine, which
executes regardless of multiprocessor occupancy.

`NullCoRunner` in the same module is a working minimal implementation and a
useful control for checking that starting and stopping a neighbor is not itself
producing an effect.

## Bring your own kernels

Write a corpus directory and point `--corpus` at it. The format is plain files
and is documented in [CORPUS.md](CORPUS.md). Nothing downstream depends on the
kernel archive the default generator pulls from.

To change which archive kernels get selected, `cokernel/corpus.py` holds the
task categories in `DEFAULT_CATEGORIES` and the selection policy in
`_select_diverse`. The policy matters more than it looks. Selecting the two
fastest candidates per task yields near-duplicate source differing by under one
percent in runtime, and such pairs cannot show an inversion, because inversion
requires the isolated-faster candidate to be the more fragile one by a margin
exceeding its lead. The default policy instead takes the fastest candidate plus
the most source-dissimilar candidate within `runtime_band` of it.

## Where the analysis lives

`analyze_flips` in `scripts/day_one.py` computes fragility and inversions from
the results dictionary. It discovers condition keys from the data through
`condition_keys` rather than reconstructing them, so a custom co-runner's
labels flow through without changes. To add a metric, read the checkpointed
results file, which carries the full per-iteration summary for every candidate
and condition, rather than modifying the measurement path.
