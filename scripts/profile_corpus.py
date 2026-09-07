#!/usr/bin/env python3
"""Profile every corpus candidate with ncu, isolated, for the predictor study.

Collects the static and dynamic properties the fragility regression needs:
registers per thread, shared memory per block, grid and block size (wave
quantization), achieved and theoretical occupancy, DRAM throughput, L2 hit
rate, and kernel duration. One row per kernel launch, since a candidate may
launch several kernels.

Needs ncu with counter permissions; the probe job reports whether this node
grants them. Run on a GPU node:

    python scripts/profile_corpus.py --corpus corpus --build-dir build/profile \
        --out results/profiles.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

METRICS = [
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__grid_size",
    "launch__block_size",
    "launch__occupancy_limit_registers",
    "sm__warps_active.avg.pct_of_peak_sustained_active",   # achieved occupancy
    "sm__maximum_warps_per_active_cycle_pct",              # theoretical occupancy
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",                          # L2 hit rate
    "gpu__time_duration.sum",
]


def profile_slug(slug: str, args) -> dict:
    runner = [
        sys.executable, str(REPO / "scripts" / "run_candidate_once.py"),
        "--corpus", str(args.corpus), "--slug", slug,
        "--build-dir", str(args.build_dir),
    ]
    # Pass 1, no ncu: warms the extension build cache so compilation stays out
    # of the profile, and proves the candidate runs at all before ncu (whose
    # failure modes are harder to read) gets involved.
    warm = subprocess.run(runner, capture_output=True, text=True, timeout=600)
    if warm.returncode != 0:
        return {"status": "run_failed", "stderr": warm.stderr[-1500:]}

    cmd = [
        "ncu", "--csv", "--target-processes", "all",
        "--profile-from-start", "off",
        "--metrics", ",".join(METRICS),
        *runner,
    ]
    prof = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if prof.returncode != 0:
        return {"status": "ncu_failed", "stderr": prof.stderr[-1500:]}

    # ncu --csv emits banner lines before the header; the header row starts
    # with "ID". One row per (launch, metric).
    lines = prof.stdout.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"ID"')), None)
    if start is None:
        return {"status": "parse_failed", "stdout_tail": prof.stdout[-1500:]}
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    launches: dict[str, dict] = {}
    for r in rows:
        key = f"{r.get('ID', '?')}:{r.get('Kernel Name', '?')[:80]}"
        launch = launches.setdefault(key, {"kernel": r.get("Kernel Name", "?")})
        launch[r.get("Metric Name", "?")] = r.get("Metric Value")
    return {"status": "ok", "launches": list(launches.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "corpus")
    ap.add_argument("--build-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "profiles.json")
    args = ap.parse_args()

    manifest = json.loads((args.corpus / "manifest.json").read_text())
    out: dict = {"metrics_requested": METRICS, "profiles": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        slug = entry["slug"]
        print(f"profiling {slug} ...", flush=True)
        rec = profile_slug(slug, args)
        rec["meta"] = entry
        print(f"  {rec['status']}"
              + (f", {len(rec.get('launches', []))} launch(es)" if rec["status"] == "ok" else ""),
              flush=True)
        out["profiles"][slug] = rec
        args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
