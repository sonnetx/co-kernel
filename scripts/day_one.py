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
        "--blocks-per-sm", str(args.blocks_per_sm),
        "--pressure-threads", str(args.pressure_threads),
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
            for lvl in args.levels:
                cond = rec["conditions"][f"pressure_r{lvl}"]
                if cond.get("starved"):
                    line += f" | r{lvl} STARVED"
                else:
                    partial = f" n={cond['n']}" if cond.get("aborted") else ""
                    line += f" | r{lvl} {cond['median_ms']/iso:.2f}x{partial}"
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
    ap.add_argument("--levels", type=int, nargs="+", default=[32, 96, 240])
    ap.add_argument("--blocks-per-sm", type=int, default=1)
    ap.add_argument("--pressure-threads", type=int, default=256)
    args = ap.parse_args()

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
    results = {"env": env, "kernels": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rec_dir = args.out.parent / f"{args.out.stem}_records"
    rec_dir.mkdir(parents=True, exist_ok=True)

    for entry in manifest:
        slug = entry["slug"]
        rec = measure_kernel(slug, entry, args, build_root, rec_dir / f"{slug}.json")
        results["kernels"][slug] = rec
        results["flips"] = analyze_flips(results, args.levels)
        args.out.write_text(json.dumps(results, indent=2) + "\n")  # checkpoint every kernel

    print(f"wrote {args.out}")
    print_flip_table(results["flips"], args.levels)
    return 0


def analyze_flips(results: dict, levels: list[int]) -> dict:
    by_task: dict[str, list[tuple[str, dict]]] = {}
    for slug, rec in results["kernels"].items():
        if rec.get("status") != "ok":
            continue
        task = f"{rec['meta']['level']}_task{rec['meta']['task_id']}"
        by_task.setdefault(task, []).append((slug, rec["conditions"]))

    flips: dict = {}
    for task, kernels in by_task.items():
        if len(kernels) < 2:
            flips[task] = {"error": f"only {len(kernels)} usable kernel(s)"}
            continue
        iso_rank = sorted(kernels, key=lambda kv: kv[1]["isolated"]["median_ms"])
        entry: dict = {
            "op_name": results["kernels"][iso_rank[0][0]]["meta"]["op_name"],
            "isolated_ranking": [
                (s, c["isolated"]["median_ms"]) for s, c in iso_rank
            ],
            "conditions": {},
        }
        def med(cond: dict) -> float:
            return cond.get("median_ms", float("inf"))  # starved ranks last

        for lvl in levels:
            key = f"pressure_r{lvl}"
            co_rank = sorted(kernels, key=lambda kv: med(kv[1][key]))
            all_starved = all("median_ms" not in c[key] for _, c in kernels)
            entry["conditions"][key] = {
                "ranking": [
                    (s, c[key].get("median_ms"), bool(c[key].get("starved")))
                    for s, c in co_rank
                ],
                "top1_flip": (not all_starved) and co_rank[0][0] != iso_rank[0][0],
                "fragility": {
                    s: (
                        c[key]["median_ms"] / c["isolated"]["median_ms"]
                        if "median_ms" in c[key]
                        else None
                    )
                    for s, c in kernels
                },
            }
        flips[task] = entry
    return flips


def print_flip_table(flips: dict, levels: list[int]) -> None:
    print("\n=== top-1 flips (isolated winner vs co-located winner) ===")
    any_flip = False
    for task, entry in flips.items():
        if "error" in entry:
            print(f"{task}: {entry['error']}")
            continue
        row = [task, entry["op_name"][:40]]
        for lvl in levels:
            c = entry["conditions"][f"pressure_r{lvl}"]
            row.append(f"r{lvl}:{'FLIP' if c['top1_flip'] else 'same'}")
            any_flip = any_flip or c["top1_flip"]
        print("  ".join(row))
    print(f"\nany top-1 flip: {any_flip}")


if __name__ == "__main__":
    raise SystemExit(main())
