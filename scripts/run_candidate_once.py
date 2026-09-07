#!/usr/bin/env python3
"""Run one candidate's forward pass once, delimited for a profiler.

The pass is wrapped in cudaProfilerStart/Stop so that, invoked under
`ncu --profile-from-start off`, only the candidate's own kernels are captured
rather than torch initialization noise. Run it once WITHOUT ncu first, which
warms the extension build cache, so compilation stays out of the profile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from cokernel import harness  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--build-dir", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=1)
    args = ap.parse_args()

    manifest = {e["slug"]: e for e in harness.load_manifest(args.corpus)}
    entry = manifest[args.slug]
    kdir = args.corpus / args.slug
    cand = harness.build_candidate(
        kdir / "kernel.cu", args.build_dir / args.slug, f"cand_{args.slug}_{entry['code_hash']}"
    )
    ref_mod = harness.load_ref(kdir)
    torch.manual_seed(7)
    inputs = [t.cuda() if isinstance(t, torch.Tensor) else t for t in ref_mod.get_inputs()]

    with torch.no_grad():
        cand.forward(*inputs)  # warmup, outside the profiled region
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for _ in range(args.iters):
            cand.forward(*inputs)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    print(f"profiled {args.slug}: {args.iters} iteration(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
