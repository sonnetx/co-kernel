"""Two-stream co-location timing harness.

The candidate kernel runs on PyTorch's current (legacy default) stream. The
pressure kernel runs on a cudaStreamNonBlocking stream created inside the
pressure extension. The legacy default stream does not synchronize with
non-blocking streams, so the two genuinely co-execute; archive kernels launch
with plain `<<<>>>` (legacy default stream), which keeps candidate work and our
timing events on one stream.

Timing: CUDA events around the forward call, one L2 flush (zeroing a 256 MB
buffer) before every timed iteration to match SOL-ExecBench/KernelBench
methodology, a device synchronize after every iteration. Wall time for each
pass is recorded alongside the event times so that a kernel whose work escapes
the default stream (a known reward-hack shape) shows up as event-sum <<
wall-clock and can be flagged.

Correctness: fresh randn inputs (three trials, different seeds than the
archive used), compared against the reference module with the KernelBench
tolerance (atol=rtol=1e-2), inside this harness process, since our streams
setup relaxes assumptions the archive's screening made.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

L2_FLUSH_BYTES = 256 * 1024 * 1024


def _load_module_from_source(py_source: str, name: str):
    import types

    mod = types.ModuleType(name)
    exec(compile(py_source, name, "exec"), mod.__dict__)
    return mod


def build_candidate(kernel_cu: Path, build_dir: Path, name: str):
    """AOT-compile one archive kernel as a torch extension; returns module."""
    import re as _re

    from torch.utils.cpp_extension import load

    # The name becomes TORCH_EXTENSION_NAME and must be a valid C identifier.
    name = _re.sub(r"\W", "_", name)
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=name,
        sources=[str(kernel_cu)],
        build_directory=str(build_dir),
        verbose=False,
    )


def build_pressure(csrc_dir: Path, build_dir: Path):
    """Compile the synthetic pressure extension.

    Retained for the prebuild smoke test. Measurement code should construct a
    cokernel.corunner.SyntheticPressure instead, which owns the same extension
    and exposes it through the CoRunner interface.
    """
    from torch.utils.cpp_extension import load

    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="cokernel_pressure",
        sources=[str(csrc_dir / "pressure.cu")],
        build_directory=str(build_dir),
        verbose=False,
    )


@dataclass
class VerifyResult:
    ok: bool
    max_abs_diff: float = float("nan")
    error: str = ""


def verify_candidate(candidate, ref_mod, trials: int = 3) -> VerifyResult:
    """Compare candidate.forward(*get_inputs()) against reference Model."""
    try:
        ref_model = ref_mod.Model(*ref_mod.get_init_inputs()).cuda().eval()
    except Exception as e:  # noqa: BLE001
        return VerifyResult(False, error=f"ref init failed: {e!r}")
    worst = 0.0
    for trial in range(trials):
        torch.manual_seed(1234 + trial)
        try:
            inputs = [
                t.cuda() if isinstance(t, torch.Tensor) else t for t in ref_mod.get_inputs()
            ]
            with torch.no_grad():
                ref_out = ref_model(*inputs)
                cand_out = candidate.forward(*inputs)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            return VerifyResult(False, error=f"trial {trial} failed: {e!r}")
        if not isinstance(cand_out, torch.Tensor) or not isinstance(ref_out, torch.Tensor):
            return VerifyResult(False, error="non-tensor output")
        if cand_out.shape != ref_out.shape:
            return VerifyResult(False, error=f"shape {cand_out.shape} vs {ref_out.shape}")
        diff = (cand_out.float() - ref_out.float()).abs().max().item()
        worst = max(worst, diff)
        if not torch.allclose(cand_out.float(), ref_out.float(), atol=1e-2, rtol=1e-2):
            return VerifyResult(False, max_abs_diff=worst, error="allclose(1e-2) failed")
    return VerifyResult(True, max_abs_diff=worst)


@dataclass
class TimingResult:
    times_ms: list[float]
    wall_s: float
    clocks_mhz: list[int] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms)

    def summary(self) -> dict:
        ts = sorted(self.times_ms)
        n = len(ts)
        return {
            "n": n,
            "median_ms": statistics.median(ts),
            "mean_ms": statistics.fmean(ts),
            "p10_ms": ts[max(0, int(0.10 * n) - 1)],
            "p90_ms": ts[min(n - 1, int(0.90 * n))],
            "min_ms": ts[0],
            "max_ms": ts[-1],
            "event_total_ms": sum(ts),
            "wall_s": self.wall_s,
            "clocks_mhz": self.clocks_mhz,
        }


class ClockMonitor:
    def __init__(self):
        self.handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001
            self.pynvml = None

    def sample(self) -> int:
        if self.handle is None:
            return -1
        try:
            return self.pynvml.nvmlDeviceGetClockInfo(
                self.handle, self.pynvml.NVML_CLOCK_SM
            )
        except Exception:  # noqa: BLE001
            return -1


def time_forward(
    candidate,
    inputs: list,
    flush_buf: torch.Tensor,
    n_iter: int = 100,
    warmup: int = 10,
    clock_monitor: ClockMonitor | None = None,
    abort: threading.Event | None = None,
) -> TimingResult:
    # NB: never torch.cuda.synchronize() here — that is cudaDeviceSynchronize,
    # which waits on ALL streams, including the pressure kernel that only
    # terminates when told to stop. Event.synchronize waits on the candidate
    # stream alone.
    #
    # `abort` is the watchdog's rescue signal: once it fires the co-runner is
    # gone, so any further iterations would time the wrong regime. Stop
    # immediately and keep only the iterations completed under pressure.
    def aborted() -> bool:
        return abort is not None and abort.is_set()

    with torch.no_grad():
        for _ in range(warmup):
            if aborted():
                break
            candidate.forward(*inputs)
        fence = torch.cuda.Event()
        fence.record()
        fence.synchronize()
        clocks = [clock_monitor.sample()] if clock_monitor else []
        times: list[float] = []
        t0 = time.perf_counter()
        for _ in range(n_iter):
            if aborted():
                break
            flush_buf.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            candidate.forward(*inputs)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        wall = time.perf_counter() - t0
        if clock_monitor:
            clocks.append(clock_monitor.sample())
    return TimingResult(times, wall, clocks)


def run_conditions(
    candidate,
    ref_mod,
    corunner,
    flush_buf: torch.Tensor,
    levels: list | None = None,
    n_iter: int = 100,
    warmup: int = 10,
    steady_state_s: float = 0.25,
    clock_monitor: ClockMonitor | None = None,
    min_timeout_s: float = 60.0,
) -> dict[str, dict]:
    """Time one candidate on an idle device and under each co-runner level.

    `corunner` is any object satisfying cokernel.corunner.CoRunner. Result keys
    are "isolated" plus one per level, named by `corunner.label(level)`.
    """
    torch.manual_seed(7)
    inputs = [t.cuda() if isinstance(t, torch.Tensor) else t for t in ref_mod.get_inputs()]
    out: dict[str, dict] = {}
    if levels is None:
        levels = corunner.levels()

    print("  condition: isolated", flush=True)
    res = time_forward(candidate, inputs, flush_buf, n_iter, warmup, clock_monitor)
    out["isolated"] = res.summary()

    # A co-runner that leaves too few resources for a candidate block starves
    # the candidate outright: its blocks queue behind neighbor blocks that
    # never retire. The watchdog stops the co-runner after a timeout, which
    # unblocks the device, and the pass is recorded rather than hanging.
    timeout_s = max(min_timeout_s, 20.0 * res.wall_s)

    supports_progress = getattr(corunner, "progress", None) is not None

    for level in levels:
        key = corunner.label(level)
        print(f"  condition: {key} (watchdog {timeout_s:.0f}s)", flush=True)
        corunner.start(level)
        fired = threading.Event()

        def _rescue():
            fired.set()
            corunner.stop()

        watchdog = threading.Timer(timeout_s, _rescue)
        aggression: dict | None = None
        try:
            time.sleep(steady_state_s)  # let the co-runner reach steady state

            # Aggression baseline: the neighbor's own rate with the device
            # otherwise idle, sampled just before the candidate starts. The
            # co-located rate is sampled across the candidate's timing pass,
            # and aggression is the ratio: how much the candidate slows its
            # neighbor. Units cancel, so the neighbor's notion of a work unit
            # never matters.
            alone_rate = None
            if supports_progress and (p := corunner.progress()) is not None:
                t_a0, p_a0 = time.perf_counter(), p
                time.sleep(0.5)
                p_a1 = corunner.progress()
                if p_a1 is not None and p_a1 > p_a0:
                    alone_rate = (p_a1 - p_a0) / (time.perf_counter() - t_a0)

            watchdog.start()
            t_c0 = time.perf_counter()
            p_c0 = corunner.progress() if alone_rate is not None else None
            res = time_forward(
                candidate, inputs, flush_buf, n_iter, warmup, clock_monitor, abort=fired
            )
            # Sample co-located progress before stop(); after a watchdog
            # rescue the neighbor is already gone and the window is polluted,
            # so aggression is only reported for unrescued passes.
            if alone_rate is not None and not fired.is_set():
                p_c1 = corunner.progress()
                if p_c0 is not None and p_c1 is not None and p_c1 > p_c0:
                    co_rate = (p_c1 - p_c0) / (time.perf_counter() - t_c0)
                    aggression = {
                        "corunner_rate_alone": alone_rate,
                        "corunner_rate_colocated": co_rate,
                        "aggression": alone_rate / co_rate,
                    }
        finally:
            watchdog.cancel()
            corunner.stop()
        if fired.is_set():
            # Iterations completed before the rescue ran fully co-located and
            # are valid samples; zero completions means no progress at all.
            if res.times_ms:
                cond = res.summary()
                cond.update({"aborted": True, "timeout_s": timeout_s})
            else:
                cond = {"starved": True, "timeout_s": timeout_s}
        else:
            cond = res.summary()
        if aggression is not None:
            cond.update(aggression)
        out[key] = cond
        time.sleep(0.05)
    return out


def load_ref(corpus_kernel_dir: Path):
    return _load_module_from_source(
        (corpus_kernel_dir / "ref.py").read_text(), f"ref_{corpus_kernel_dir.name}"
    )


def load_manifest(corpus_dir: Path) -> list[dict]:
    return json.loads((corpus_dir / "manifest.json").read_text())
