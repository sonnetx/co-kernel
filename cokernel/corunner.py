"""Co-runner interface and implementations.

A co-runner is whatever shares the GPU with the candidate kernel under test.
The harness only needs four things from one, so supplying your own neighbor
means implementing the `CoRunner` protocol below and passing an instance to
`harness.run_conditions`. Nothing else in the harness knows what the neighbor
is.

Shipped implementations:

  SyntheticPressure  one persistent block per SM spinning on register-resident
                     arithmetic, with register and shared-memory knobs. Isolates
                     a resource axis; not a workload anyone deploys.
  NullCoRunner       a no-op neighbor, useful as a control to confirm that the
                     act of starting and stopping a neighbor is not itself
                     responsible for an observed effect.

To co-locate a real workload (an LLM decode loop, a batch job), implement the
protocol over your own driver. Two requirements are easy to miss. Your `start`
must return only once the workload is actually resident and doing steady-state
work, since the harness begins measuring shortly after. And your `stop` must
be callable from a watchdog thread while the main thread is blocked inside a
CUDA synchronize, so it cannot rely on the main thread making progress. See
docs/EXTENDING.md for a worked sketch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CoRunner(Protocol):
    """The device neighbor a candidate kernel is measured against."""

    def levels(self) -> list[Any]:
        """Configurations to sweep. Each is passed back to `start`."""
        ...

    def start(self, level: Any) -> None:
        """Begin occupying the device and return once in steady state."""
        ...

    def stop(self) -> None:
        """Stop occupying the device. Must be safe to call from another thread
        and safe to call when already stopped."""
        ...

    def describe(self) -> dict:
        """Provenance recorded into the results file."""
        ...

    def label(self, level: Any) -> str:
        """Short key for a level, used in result keys and log lines."""
        ...


class NullCoRunner:
    """A neighbor that does nothing. Control condition."""

    def levels(self) -> list[Any]:
        return ["null"]

    def start(self, level: Any) -> None:
        return None

    def stop(self) -> None:
        return None

    def describe(self) -> dict:
        return {"kind": "null"}

    def label(self, level: Any) -> str:
        return "null"


class SyntheticPressure:
    """Persistent register/shared-memory hog, one block per SM.

    `levels` defaults to the register levels compiled into the extension, so
    changing the sweep means editing COKERNEL_PRESSURE_LEVELS in pressure.cu
    and rebuilding, with no Python change. Pass `levels=` to use a subset.
    """

    def __init__(
        self,
        csrc_dir: str | Path,
        build_dir: str | Path,
        blocks_per_sm: int = 1,
        threads: int = 256,
        smem_bytes: int = 0,
        levels: list[int] | None = None,
    ):
        from torch.utils.cpp_extension import load

        build_dir = Path(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        self._ext = load(
            name="cokernel_pressure",
            sources=[str(Path(csrc_dir) / "pressure.cu")],
            build_directory=str(build_dir),
            verbose=False,
        )
        compiled = [int(x) for x in self._ext.levels()]
        if levels is None:
            self._levels = compiled
        else:
            missing = sorted(set(levels) - set(compiled))
            if missing:
                raise ValueError(
                    f"levels {missing} are not compiled into the extension; "
                    f"compiled levels are {compiled}. Add them to "
                    f"COKERNEL_PRESSURE_LEVELS in pressure.cu and rebuild."
                )
            self._levels = list(levels)
        self.blocks_per_sm = blocks_per_sm
        self.threads = threads
        self.smem_bytes = smem_bytes

    def levels(self) -> list[Any]:
        return list(self._levels)

    def start(self, level: Any) -> None:
        # The kernel is resident as soon as the launch is accepted; the caller
        # adds a steady-state wait before measuring.
        self._ext.start(int(level), self.blocks_per_sm, self.threads, self.smem_bytes)

    def stop(self) -> None:
        self._ext.stop()

    def describe(self) -> dict:
        return {
            "kind": "synthetic_pressure",
            "num_sms": self._ext.num_sms(),
            "blocks_per_sm": self.blocks_per_sm,
            "threads": self.threads,
            "smem_bytes": self.smem_bytes,
            "levels": list(self._levels),
            # (nreg, achieved regs/thread, static smem, local bytes)
            "reg_info": [list(map(int, row)) for row in self._ext.reg_info()],
        }

    def label(self, level: Any) -> str:
        return f"r{int(level)}"
