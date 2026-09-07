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

import subprocess
import sys
import tempfile
import time
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

    def progress(self) -> int | None:
        """Cumulative work units completed since `start`, or None when the
        neighbor cannot report progress. Units are arbitrary but must be
        consistent within one start/stop cycle; the harness only ever uses
        the ratio of two rates. Must be callable while the neighbor runs and
        while the main thread is busy with candidate work."""
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

    def progress(self) -> int | None:
        return None


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

    def progress(self) -> int | None:
        return int(self._ext.progress())


class ProcessCoRunner:
    """A neighbor in its own process, so its own CUDA context.

    With no MPS daemon the contexts time-slice, which is the plain
    multi-process arm. With an MPS daemon up and both processes inheriting
    CUDA_MPS_PIPE_DIRECTORY, kernels co-execute in a shared context, which is
    the deployment-representative arm. The harness code is identical either
    way; the difference is environment set up by the launcher.

    `kind` selects a built-in neighbor from cokernel.neighbor (pressure, gemm,
    stream). Coordination happens through ready/stop/progress files, so `stop`
    works from a watchdog thread regardless of what either process is doing.
    """

    def __init__(
        self,
        kind: str,
        levels: list,
        build_dir: str | Path | None = None,
        ready_timeout_s: float = 120.0,
        workdir: str | Path | None = None,
    ):
        self.kind = kind
        self._levels = list(levels)
        self.build_dir = Path(build_dir) if build_dir else None
        self.ready_timeout_s = ready_timeout_s
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="cokernel_nbr_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen | None = None
        self._files: dict[str, Path] = {}

    def levels(self) -> list[Any]:
        return list(self._levels)

    def start(self, level: Any) -> None:
        if self._proc is not None:
            raise RuntimeError("neighbor already running")
        tag = f"{self.kind}_{level}_{time.time_ns()}"
        self._files = {
            "ready": self.workdir / f"{tag}.ready",
            "stop": self.workdir / f"{tag}.stop",
            "progress": self.workdir / f"{tag}.progress",
        }
        cmd = [
            sys.executable, "-m", "cokernel.neighbor",
            "--kind", self.kind, "--level", str(level),
            "--ready-file", str(self._files["ready"]),
            "--stop-file", str(self._files["stop"]),
            "--progress-file", str(self._files["progress"]),
        ]
        if self.build_dir:
            cmd += ["--build-dir", str(self.build_dir)]
        self._proc = subprocess.Popen(cmd)
        deadline = time.perf_counter() + self.ready_timeout_s
        while not self._files["ready"].exists():
            if self._proc.poll() is not None:
                self._proc = None
                raise RuntimeError(f"neighbor exited before ready (kind={self.kind})")
            if time.perf_counter() > deadline:
                self.stop()
                raise TimeoutError(f"neighbor not ready in {self.ready_timeout_s}s")
            time.sleep(0.05)

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if self._files:
            self._files["stop"].touch()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def progress(self) -> int | None:
        f = self._files.get("progress")
        if f is None or not f.exists():
            return None
        try:
            return int(f.read_text().strip())
        except (ValueError, OSError):
            return None

    def describe(self) -> dict:
        return {
            "kind": f"process_{self.kind}",
            "levels": [str(x) for x in self._levels],
            "mps_pipe": __import__("os").environ.get("CUDA_MPS_PIPE_DIRECTORY"),
        }

    def label(self, level: Any) -> str:
        return f"{self.kind}{level}"


def make_corunner(spec: str, levels: list, build_dir: Path, **synth_kwargs):
    """Build a co-runner from a CLI spec.

    "synthetic" gives the in-process two-stream arm. "proc-pressure",
    "proc-gemm", and "proc-stream" give a neighbor process (multi-process arm,
    or MPS when the daemon environment is present).
    """
    if spec == "synthetic":
        repo = Path(__file__).resolve().parent
        return SyntheticPressure(
            csrc_dir=repo / "csrc",
            build_dir=build_dir / "pressure",
            levels=[int(x) for x in levels],
            **synth_kwargs,
        )
    if spec.startswith("proc-"):
        return ProcessCoRunner(
            kind=spec.removeprefix("proc-"),
            levels=levels,
            build_dir=build_dir / "neighbor_pressure",
        )
    raise ValueError(f"unknown co-runner spec {spec!r}")
