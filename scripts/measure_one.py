#!/usr/bin/env python3
"""Measure a single kernel (build, verify, time all conditions) and write its
record to a JSON file. Run as a subprocess by day_one.py so that a kernel which
hangs the CUDA context (internal streams, host-side sync, register starvation
with no watchdog-reachable return) is killed and abandoned without taking the
whole sweep down. Mirrors SOL-ExecBench's isolated-subprocess execution.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

print = functools.partial(print, flush=True)  # noqa: A001

import torch  # noqa: E402

from cokernel import harness  # noqa: E402
from cokernel.corunner import SyntheticPressure  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--build-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--levels", type=int, nargs="+", default=[32, 96, 240])
    ap.add_argument("--blocks-per-sm", type=int, default=1)
    ap.add_argument("--pressure-threads", type=int, default=256)
    ap.add_argument("--pressure-smem-bytes", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    manifest = {e["slug"]: e for e in harness.load_manifest(args.corpus)}
    entry = manifest[args.slug]
    rec: dict = {"meta": entry}

    def dump(status: str) -> None:
        rec["status"] = status
        args.out.write_text(json.dumps(rec) + "\n")

    dump("started")  # so a hard kill leaves evidence of where we died

    corunner = SyntheticPressure(
        csrc_dir=REPO / "cokernel" / "csrc",
        build_dir=args.build_dir / "pressure",
        blocks_per_sm=args.blocks_per_sm,
        threads=args.pressure_threads,
        smem_bytes=args.pressure_smem_bytes,
        levels=args.levels,
    )
    rec["corunner"] = corunner.describe()
    kdir = args.corpus / args.slug
    try:
        cand = harness.build_candidate(
            kdir / "kernel.cu", args.build_dir / args.slug, f"cand_{args.slug}_{entry['code_hash']}"
        )
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
        dump("build_failed")
        return 0

    ref_mod = harness.load_ref(kdir)
    v = harness.verify_candidate(cand, ref_mod)
    rec["verify"] = {"ok": v.ok, "max_abs_diff": v.max_abs_diff, "error": v.error}
    if not v.ok:
        dump("verify_failed")
        return 0

    clock_monitor = harness.ClockMonitor()
    flush_buf = torch.empty(harness.L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    rec["conditions"] = harness.run_conditions(
        cand, ref_mod, corunner, flush_buf,
        n_iter=args.n_iter,
        warmup=args.warmup,
        clock_monitor=clock_monitor,
    )
    dump("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
