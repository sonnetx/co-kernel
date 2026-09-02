#!/usr/bin/env python3
"""Compile the pressure extension and every corpus kernel without a GPU.

Catches build breakage before the SLURM job burns its queue slot. Run on a
login node with the same modules the job loads (python/3.12.1 cuda/12.4.0
gcc/12.4.0).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cokernel import harness  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=REPO / "corpus")
    ap.add_argument("--build-dir", type=Path, required=True)
    args = ap.parse_args()

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
    failures = []

    print("building pressure ...", flush=True)
    try:
        harness.build_pressure(REPO / "cokernel" / "csrc", args.build_dir / "pressure")
    except Exception:  # noqa: BLE001
        failures.append(("pressure", traceback.format_exc(limit=4)))

    for entry in harness.load_manifest(args.corpus):
        slug = entry["slug"]
        print(f"building {slug} ...", flush=True)
        try:
            harness.build_candidate(
                args.corpus / slug / "kernel.cu",
                args.build_dir / slug,
                f"cand_{slug}_{entry['code_hash']}",
            )
        except Exception:  # noqa: BLE001
            failures.append((slug, traceback.format_exc(limit=4)))

    if failures:
        for name, tb in failures:
            print(f"\n=== BUILD FAILED: {name} ===\n{tb}")
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("ALL BUILDS OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
