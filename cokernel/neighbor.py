"""Standalone neighbor process for cross-process co-location arms.

Run as `python -m cokernel.neighbor --kind gemm ...` by ProcessCoRunner, or by
hand. Where the two-stream arm shares one CUDA context, a neighbor process has
its own, so the pair exercises what plain multi-process time-slicing or MPS
(when its daemon is up and both processes inherit its pipe directory) actually
deploys.

Protocol with the parent, all through files so it works across any process
boundary and from watchdog threads:

  --ready-file  created by the neighbor once it is in steady state
  --stop-file   created by the parent; the neighbor exits when it appears
  --progress-file  rewritten atomically by the neighbor as "<count>\n"

Kinds:
  pressure  the synthetic register hog, via the same extension the in-process
            arm uses. Level is NREG.
  gemm      back-to-back large bf16 matmuls. Compute-bound aggressor in the
            shape of prefill. Level is the square matrix dimension.
  stream    large elementwise multiplies. Bandwidth-bound neighbor in the
            shape of decode. Level is the buffer size in MiB.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def _write_progress(path: Path, count: int) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(f"{count}\n")
    os.replace(tmp, path)  # atomic on POSIX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["pressure", "gemm", "stream"])
    ap.add_argument("--level", required=True)
    ap.add_argument("--ready-file", type=Path, required=True)
    ap.add_argument("--stop-file", type=Path, required=True)
    ap.add_argument("--progress-file", type=Path, required=True)
    ap.add_argument("--build-dir", type=Path, help="pressure kind only")
    ap.add_argument("--warmup-iters", type=int, default=5)
    args = ap.parse_args()

    import torch

    assert torch.cuda.is_available(), "neighbor needs a CUDA device"

    def stopped() -> bool:
        return args.stop_file.exists()

    if args.kind == "pressure":
        from cokernel.corunner import SyntheticPressure

        repo = Path(__file__).resolve().parent
        corunner = SyntheticPressure(
            csrc_dir=repo / "csrc",
            build_dir=args.build_dir or Path("build/neighbor_pressure"),
        )
        corunner.start(int(args.level))
        args.ready_file.touch()
        try:
            while not stopped():
                p = corunner.progress()
                if p is not None:
                    _write_progress(args.progress_file, p)
                time.sleep(0.05)
        finally:
            corunner.stop()
        return 0

    # Torch-driven kinds share one loop shape: warm up, signal ready, then do
    # one unit of work per iteration, publishing progress and polling the stop
    # file. The device synchronize per iteration keeps the published count an
    # honest measure of completed work rather than queued work.
    if args.kind == "gemm":
        n = int(args.level)
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)

        def unit() -> None:
            torch.mm(a, b)

    else:  # stream
        mib = int(args.level)
        buf = torch.randn(mib * 1024 * 1024 // 4, device="cuda")

        def unit() -> None:
            buf.mul_(1.0000001)

    with torch.no_grad():
        for _ in range(args.warmup_iters):
            unit()
        torch.cuda.synchronize()
        args.ready_file.touch()
        count = 0
        last_pub = 0.0
        while not stopped():
            unit()
            torch.cuda.synchronize()
            count += 1
            now = time.perf_counter()
            if now - last_pub > 0.05:
                _write_progress(args.progress_file, count)
                last_pub = now
        _write_progress(args.progress_file, count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
