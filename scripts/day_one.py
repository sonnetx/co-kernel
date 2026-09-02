#!/usr/bin/env python3
"""Day-one falsification experiment.

Ten agent-generated kernels (5 tasks x top-2 from the Sakana archive), timed
isolated and under a synthetic co-runner at three register-pressure levels,
two-stream single-process co-execution on one H100. The question is binary:
does any task's top-1 kernel flip under pressure?

Usage (on a GPU node):
    python scripts/day_one.py --corpus corpus --out results/day_one.json
"""

from __future__ import annotations

import argparse
import faulthandler
import functools
import json
import os
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001 - logs must survive a hang

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from cokernel import harness  # noqa: E402


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

    # If anything hangs (a starved launch, a stuck sync), dump all thread
    # stacks to stderr every 10 minutes so the walltime kill is diagnosable.
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, repeat=True)

    assert torch.cuda.is_available(), "no CUDA device"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
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

    print("building pressure extension ...")
    pressure = harness.build_pressure(REPO / "cokernel" / "csrc", build_root / "pressure")
    reg_info = pressure.reg_info()
    env["num_sms"] = pressure.num_sms()
    env["pressure_reg_info"] = reg_info
    print(f"SMs: {env['num_sms']}; pressure levels (nreg, regs/thread, smem, local): {reg_info}")

    manifest = harness.load_manifest(args.corpus)
    clock_monitor = harness.ClockMonitor()
    flush_buf = torch.empty(harness.L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")

    # Build everything AOT before any timing.
    built = {}
    for entry in manifest:
        slug = entry["slug"]
        kdir = args.corpus / slug
        print(f"building {slug} ({entry['op_name']}, archive {entry['archive_runtime_ms']:.3f} ms) ...")
        t_build = time.perf_counter()
        try:
            cand = harness.build_candidate(
                kdir / "kernel.cu", build_root / slug, f"cand_{slug}_{entry['code_hash']}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  BUILD FAILED: {e}")
            built[slug] = None
            continue
        print(f"  built in {time.perf_counter() - t_build:.1f}s")
        built[slug] = cand

    results = {"env": env, "kernels": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        args.out.write_text(json.dumps(results, indent=2) + "\n")

    for entry in manifest:
        slug = entry["slug"]
        cand = built.get(slug)
        rec: dict = {"meta": entry}
        if cand is None:
            rec["status"] = "build_failed"
            results["kernels"][slug] = rec
            checkpoint()
            continue
        ref_mod = harness.load_ref(args.corpus / slug)
        print(f"verifying {slug} ...")
        v = harness.verify_candidate(cand, ref_mod)
        rec["verify"] = {"ok": v.ok, "max_abs_diff": v.max_abs_diff, "error": v.error}
        if not v.ok:
            print(f"{slug}: VERIFY FAILED ({v.error})")
            rec["status"] = "verify_failed"
            results["kernels"][slug] = rec
            checkpoint()
            continue
        print(f"{slug}: verified (max abs diff {v.max_abs_diff:.2e}); timing ...")
        rec["conditions"] = harness.run_conditions(
            cand,
            ref_mod,
            pressure,
            flush_buf,
            pressure_levels=args.levels,
            blocks_per_sm=args.blocks_per_sm,
            pressure_threads=args.pressure_threads,
            n_iter=args.n_iter,
            warmup=args.warmup,
            clock_monitor=clock_monitor,
        )
        rec["status"] = "ok"
        iso = rec["conditions"]["isolated"]["median_ms"]
        line = f"  isolated {iso*1000:.1f} us"
        for lvl in args.levels:
            cond = rec["conditions"][f"pressure_r{lvl}"]
            if cond.get("starved"):
                line += f" | r{lvl} STARVED"
            else:
                m = cond["median_ms"]
                partial = f" partial n={cond['n']}" if cond.get("aborted") else ""
                line += f" | r{lvl} {m*1000:.1f} us ({m/iso:.2f}x{partial})"
        print(line)
        results["kernels"][slug] = rec
        checkpoint()

    results["flips"] = analyze_flips(results, args.levels)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
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
