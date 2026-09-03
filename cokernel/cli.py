"""Console entry points, so an installed copy needs no repo-relative paths.

Each function forwards to the corresponding script in `scripts/`, which stays
runnable directly for anyone working from a checkout.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _run(script: str) -> None:
    path = SCRIPTS / script
    if not path.exists():
        raise SystemExit(
            f"{path} not found. The console commands require a source checkout; "
            f"clone the repository and `pip install -e .` from it."
        )
    sys.argv[0] = str(path)
    runpy.run_path(str(path), run_name="__main__")


def day_one() -> None:
    _run("day_one.py")


def prepare_corpus() -> None:
    _run("prepare_corpus.py")


def prebuild() -> None:
    _run("prebuild.py")
