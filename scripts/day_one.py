#!/usr/bin/env python3
"""Day-one falsification experiment.

Agent-generated kernels (a handful of KernelBench tasks, several source-diverse
correct kernels each within a runtime band of the fastest, from the Sakana
archive), timed isolated and under a synthetic co-runner at three
register-pressure levels, two-stream single-process co-execution on one H100.
The question is binary: does any task's top-1 kernel flip under pressure?

Each kernel is measured in its own subprocess (measure_one.py) under a hard
wall timeout, so a kernel that hangs the CUDA context is abandoned rather than
consuming the whole allocation.

Usage (on a GPU node):
    python scripts/day_one.py --corpus corpus --out results/day_one.json
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001 - logs must survive a hang

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from cokernel import harness  # noqa: E402


def measure_kernel(slug: str, entry: dict, args, build_root: Path, rec_path: Path) -> dict:
    """Run measure_one.py for one kernel with a hard wall timeout.

    Isolating each kernel in a subprocess means a kernel that hangs the CUDA
    context (internal streams + host-side sync, or register starvation with no
    return the watchdog can reach) is killed and abandoned; the sweep goes on.
    The per-kernel budget covers build + verify + (1 + n_levels) timed windows,
    each capped by the in-process watchdog, plus generous slack.
    """
    n_windows = 1 + len(args.levels)
    budget_s = 180 + n_windows * 90  # matches the harness watchdog ceiling
    cmd = [
        sys.executable, str(REPO / "scripts" / "measure_one.py"),
        "--corpus", str(args.corpus), "--slug", slug,
        "--build-dir", str(build_root), "--out", str(rec_path),
        "--n-iter", str(args.n_iter), "--warmup", str(args.warmup),
        "--levels", *[str(x) for x in args.levels],
        "--corunner", args.corunner,
        "--blocks-per-sm", str(args.blocks_per_sm),
        "--pressure-threads", str(args.pressure_threads),
        "--pressure-smem-bytes", str(args.pressure_smem_bytes),
    ]
    print(f"{slug} ({entry['op_name']}, archive {entry['archive_runtime_ms']:.3f} ms) "
          f"-> subprocess (budget {budget_s}s)")
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, timeout=budget_s, check=False)
    except subprocess.TimeoutExpired:
        print(f"  {slug}: HUNG, killed after {budget_s}s")
        return {"meta": entry, "status": "hung", "budget_s": budget_s}
    dt = time.perf_counter() - t0
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        status = rec.get("status", "?")
        if status == "ok":
            iso = rec["conditions"]["isolated"]["median_ms"]
            line = f"  {slug}: ok in {dt:.0f}s | isolated {iso*1000:.1f} us"
            for key, cond in rec["conditions"].items():
                if key == "isolated":
                    continue
                if cond.get("starved"):
                    line += f" | {key} no-progress"
                else:
                    partial = f" n={cond['n']}" if cond.get("aborted") else ""
                    line += f" | {key} {cond['median_ms']/iso:.2f}x{partial}"
            print(line)
        else:
            print(f"  {slug}: {status} ({rec.get('verify',{}).get('error') or rec.get('error','')})")
        return rec
    print(f"  {slug}: no record written (crashed before first dump)")
    return {"meta": entry, "status": "crashed"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "corpus")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "day_one.json")
    ap.add_argument("--build-dir", type=Path, default=None,
                    help="extension build root (default: $COKERNEL_BUILD or ./build)")
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    # Measured regs/thread on H100 (cudaFuncGetAttributes): 32 -> 40,
    # 96 -> 127, 240 -> 255 (the launch-bounds cap; NREG=168 also compiles
    # to 255, so it would duplicate the top level).
    ap.add_argument("--levels", type=int, nargs="+", default=None,
                    help="co-runner levels; default depends on --corunner")
    ap.add_argument("--corunner", default="synthetic",
                    choices=["synthetic", "proc-pressure", "proc-gemm", "proc-stream"],
                    help="synthetic = in-process two-stream arm; proc-* = a "
                         "neighbor process (plain multi-process, or MPS when "
                         "the daemon environment is present)")
    ap.add_argument("--blocks-per-sm", type=int, default=1)
    ap.add_argument("--pressure-threads", type=int, default=256)
    ap.add_argument("--pressure-smem-bytes", type=int, default=0,
                    help="dynamic shared memory the co-runner holds per block")
    ap.add_argument("--limit", type=int, default=None,
                    help="measure only the first N kernels in the manifest")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: few iterations, one pressure level, "
                         "first 2 kernels. Verifies an install in ~2 minutes; "
                         "the timings it produces are not publishable.")
    args = ap.parse_args()

    if args.levels is None:
        # Level units are per kind: registers for the pressure kinds, matrix
        # dimension for gemm, buffer MiB for stream.
        args.levels = {
            "synthetic": [32, 96, 240],
            "proc-pressure": [32, 96, 240],
            "proc-gemm": [8192],
            "proc-stream": [2048],
        }[args.corunner]

    if args.quick:
        args.n_iter, args.warmup = 10, 3
        args.levels = args.levels[:1]
        args.limit = args.limit or 2
        print("QUICK MODE: results are a smoke test, not a measurement")

    build_root = args.build_dir or Path(os.environ.get("COKERNEL_BUILD", REPO / "build"))
    env = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hostname": os.uname().nodename,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_iter": args.n_iter,
        "levels": args.levels,
        "blocks_per_sm": args.blocks_per_sm,
        "pressure_threads": args.pressure_threads,
    }
    print(f"env: {json.dumps(env)}")

    manifest = harness.load_manifest(args.corpus)
    if args.limit:
        manifest = manifest[: args.limit]
    results = {"env": env, "kernels": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rec_dir = args.out.parent / f"{args.out.stem}_records"
    rec_dir.mkdir(parents=True, exist_ok=True)

    for entry in manifest:
        slug = entry["slug"]
        rec = measure_kernel(slug, entry, args, build_root, rec_dir / f"{slug}.json")
        results["kernels"][slug] = rec
        results["flips"] = analyze_flips(results)
        args.out.write_text(json.dumps(results, indent=2) + "\n")  # checkpoint every kernel

    print(f"wrote {args.out}")
    print_flip_table(results["flips"], condition_keys(results))
    return 0


def condition_keys(results: dict) -> list[str]:
    """Co-located condition keys present in the results, in first-seen order.

    Keys come from the co-runner's own labelling, so the analysis works for any
    CoRunner implementation without knowing its naming scheme.
    """
    keys: list[str] = []
    for rec in results["kernels"].values():
        for k in rec.get("conditions", {}):
            if k != "isolated" and k not in keys:
                keys.append(k)
    return keys


def analyze_flips(results: dict, keys: list[str] | None = None) -> dict:
    """Per task, compare the isolated ranking against each co-located ranking.

    Reports fragility per candidate (co-located over isolated latency) and
    whether the isolated top-1 candidate is still top-1 co-located. A candidate
    that completed no iterations ranks last and is excluded from fragility.
    """
    if keys is None:
        keys = condition_keys(results)
    by_task: dict[str, list[tuple[str, dict]]] = {}
    for slug, rec in results["kernels"].items():
        if rec.get("status") != "ok":
            continue
        task = f"{rec['meta']['level']}_task{rec['meta']['task_id']}"
        by_task.setdefault(task, []).append((slug, rec["conditions"]))

    def med(cond: dict) -> float:
        return cond.get("median_ms", float("inf"))  # no progress ranks last

    flips: dict = {}
    for task, kernels in by_task.items():
        if len(kernels) < 2:
            flips[task] = {"error": f"only {len(kernels)} usable kernel(s)"}
            continue
        iso_rank = sorted(kernels, key=lambda kv: kv[1]["isolated"]["median_ms"])
        entry: dict = {
            "op_name": results["kernels"][iso_rank[0][0]]["meta"]["op_name"],
            "isolated_ranking": [(s, c["isolated"]["median_ms"]) for s, c in iso_rank],
            "conditions": {},
        }
        for key in keys:
            present = [(s, c) for s, c in kernels if key in c]
            if len(present) < 2:
                entry["conditions"][key] = {"error": "condition missing for pair"}
                continue
            co_rank = sorted(present, key=lambda kv: med(kv[1][key]))
            rankable = [(s, c) for s, c in present if "median_ms" in c[key]]
            entry["conditions"][key] = {
                "ranking": [
                    (s, c[key].get("median_ms"), bool(c[key].get("starved")))
                    for s, c in co_rank
                ],
                # A flip needs at least two candidates that actually ran.
                "top1_flip": len(rankable) >= 2 and co_rank[0][0] != iso_rank[0][0],
                "rankable": len(rankable),
                "fragility": {
                    s: (
                        c[key]["median_ms"] / c["isolated"]["median_ms"]
                        if "median_ms" in c[key]
                        else None
                    )
                    for s, c in present
                },
            }
        flips[task] = entry
    return flips


def print_flip_table(flips: dict, keys: list[str]) -> None:
    print("\n=== top-1 flips (isolated winner vs co-located winner) ===")
    any_flip = False
    for task, entry in flips.items():
        if "error" in entry:
            print(f"{task}: {entry['error']}")
            continue
        row = [f"{task:14s}", f"{entry['op_name'][:34]:34s}"]
        for key in keys:
            c = entry["conditions"].get(key, {})
            if "error" in c:
                row.append(f"{key}:n/a")
            elif c["rankable"] < 2:
                row.append(f"{key}:no-rank")
            else:
                row.append(f"{key}:{'FLIP' if c['top1_flip'] else 'same'}")
                any_flip = any_flip or c["top1_flip"]
        print("  ".join(row))
    print(f"\nany top-1 flip: {any_flip}")


if __name__ == "__main__":
    raise SystemExit(main())
