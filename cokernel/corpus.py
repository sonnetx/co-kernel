"""Select candidate kernels from the Sakana AI CUDA Engineer archive.

The archive (HF: SakanaAI/AI-CUDA-Engineer-Archive) holds ~30k agent-generated
CUDA kernels over KernelBench tasks, each row carrying the full torch-extension
source, the PyTorch reference module, a correctness flag, and a measured
runtime. We treat archive runtimes as a pre-ranking only; the harness
re-measures everything isolated and co-located on the same device, and
re-verifies correctness in-process (the archive's screening is known to have
missed reward hacks, and our harness relaxes assumptions theirs made).

Day-one selection: tasks whose reference needs no init parameters (so the
candidate `forward(*get_inputs())` arity is unambiguous), with at least
`min_kernels` distinct correct kernels and a runtime floor that keeps event
timing above noise. From each task, the top-k fastest correct kernels with
distinct source.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Preferred day-one categories, matched against Op_Name in order. Spans
# compute-bound (matmuls) and bandwidth-bound (matvec, softmax, losses).
# Reductions/scans are excluded implicitly: their references take the
# reduction dim as an init parameter.
DEFAULT_CATEGORIES = [
    r"Square_matrix_multiplication",
    r"Batched_matrix_multiplication",
    r"Softmax",
    r"Matrix_vector_multiplication",
    r"Matmul_with_transposed",
    r"MSELoss|CrossEntropy|KLDiv|CosineSimilarity|HingeLoss",
]


@dataclass
class KernelPick:
    level: str
    task_id: int
    op_name: str
    kernel_rank: int
    kernel_name: str
    archive_runtime_ms: float
    archive_max_diff: str
    cuda_code: str
    ref_code: str

    @property
    def slug(self) -> str:
        return f"{self.level}_task{self.task_id}_k{self.kernel_rank}"

    @property
    def code_hash(self) -> str:
        return hashlib.sha256(self.cuda_code.encode()).hexdigest()[:12]


def _init_inputs_empty(ref_code: str) -> bool:
    """True when get_init_inputs() returns an empty list, judged statically.

    Static check keeps corpus selection runnable without executing dataset
    code. Accepts `return []` possibly followed by a comment.
    """
    m = re.search(
        r"def get_init_inputs\(\):\s*\n(?:\s*#[^\n]*\n|\s*\"\"\".*?\"\"\"\s*\n)*\s*return \[\]",
        ref_code,
        re.DOTALL,
    )
    return m is not None


def load_level(parquet_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(
        parquet_path,
        columns=[
            "Op_Name",
            "Level_ID",
            "Task_ID",
            "Kernel_Name",
            "CUDA_Runtime",
            "CUDA_Code",
            "PyTorch_Code_Module",
            "Correct",
            "Max_Diff",
        ],
    )
    df["CUDA_Runtime"] = pd.to_numeric(df["CUDA_Runtime"], errors="coerce")
    if df["Correct"].dtype == object:
        df["Correct"] = df["Correct"].astype(str) == "True"
    return df


def select_corpus(
    df: pd.DataFrame,
    categories: list[str] = DEFAULT_CATEGORIES,
    kernels_per_task: int = 2,
    min_kernels: int = 4,
    min_runtime_ms: float = 0.02,
) -> list[KernelPick]:
    """Pick one task per category, top-k fastest distinct correct kernels each."""
    correct = df[df["Correct"] & df["CUDA_Runtime"].notna() & (df["CUDA_Runtime"] > 0)]
    picks: list[KernelPick] = []
    used_tasks: set[tuple[str, int]] = set()

    for pattern in categories:
        cand = correct[correct["Op_Name"].str.contains(pattern, regex=True, na=False)]
        chosen_task = None
        # Rank matching tasks by number of distinct correct kernels, prefer
        # tasks whose fastest kernels sit above the runtime floor.
        stats = (
            cand.groupby(["Level_ID", "Task_ID"])
            .agg(n=("CUDA_Runtime", "size"), best=("CUDA_Runtime", "min"))
            .reset_index()
            .sort_values("n", ascending=False)
        )
        for floor in (min_runtime_ms, 0.0):  # drop the floor if it empties the category
            for _, row in stats.iterrows():
                key = (str(row["Level_ID"]), int(row["Task_ID"]))
                if key in used_tasks or row["n"] < min_kernels:
                    continue
                if row["best"] < floor:
                    continue
                task_rows = cand[
                    (cand["Level_ID"] == row["Level_ID"]) & (cand["Task_ID"] == row["Task_ID"])
                ]
                ref_code = task_rows.iloc[0]["PyTorch_Code_Module"]
                if not _init_inputs_empty(ref_code):
                    continue
                chosen_task = (key, task_rows)
                break
            if chosen_task is not None:
                break
        if chosen_task is None:
            print(f"WARNING: no eligible task for category {pattern!r}, skipping")
            continue

        (level, task_id), task_rows = chosen_task
        used_tasks.add((level, task_id))
        task_rows = task_rows.sort_values("CUDA_Runtime")
        seen_hashes: set[str] = set()
        rank = 0
        for _, r in task_rows.iterrows():
            h = hashlib.sha256(r["CUDA_Code"].encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            picks.append(
                KernelPick(
                    level=str(level),
                    task_id=int(task_id),
                    op_name=str(r["Op_Name"]),
                    kernel_rank=rank,
                    kernel_name=str(r["Kernel_Name"]),
                    archive_runtime_ms=float(r["CUDA_Runtime"]),
                    archive_max_diff=str(r["Max_Diff"]),
                    cuda_code=str(r["CUDA_Code"]),
                    ref_code=str(r["PyTorch_Code_Module"]),
                )
            )
            rank += 1
            if rank >= kernels_per_task:
                break
    return picks


def write_corpus(picks: list[KernelPick], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for p in picks:
        d = out / p.slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel.cu").write_text(p.cuda_code)
        (d / "ref.py").write_text(p.ref_code)
        meta = {
            "level": p.level,
            "task_id": p.task_id,
            "op_name": p.op_name,
            "kernel_rank": p.kernel_rank,
            "kernel_name": p.kernel_name,
            "archive_runtime_ms": p.archive_runtime_ms,
            "archive_max_diff": p.archive_max_diff,
            "code_hash": p.code_hash,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        manifest.append({"slug": p.slug, **meta})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(picks)} kernels to {out}")
