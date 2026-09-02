#!/usr/bin/env python3
"""Download the Sakana archive parquet and write the day-one corpus.

Needs network (run on a login node or locally):
    python scripts/prepare_corpus.py --out corpus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cokernel import corpus  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "corpus")
    ap.add_argument("--levels", nargs="+", default=["level_1"])
    ap.add_argument("--kernels-per-task", type=int, default=2)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    frames = []
    for level in args.levels:
        path = hf_hub_download(
            repo_id="SakanaAI/AI-CUDA-Engineer-Archive",
            filename=f"{level}.parquet",
            repo_type="dataset",
        )
        frames.append(corpus.load_level(path))

    import pandas as pd

    df = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(df)} kernels, {df['Correct'].sum()} marked correct")
    picks = corpus.select_corpus(df, kernels_per_task=args.kernels_per_task)
    for p in picks:
        print(f"  {p.slug}: {p.op_name} archive {p.archive_runtime_ms:.3f} ms hash {p.code_hash}")
    corpus.write_corpus(picks, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
