# Corpus format

A corpus is a directory of candidate kernels plus a manifest. The format is
deliberately plain files, so you can hand-write one, generate it from your own
kernel source, or export it from another benchmark without touching the
harness.

```
corpus/
  manifest.json
  <slug>/
    kernel.cu     candidate CUDA source, a torch extension exposing forward()
    ref.py        PyTorch reference defining Model, get_inputs, get_init_inputs
    meta.json     provenance for this candidate
```

`scripts/prepare_corpus.py` writes this layout from the public kernel archive,
but nothing downstream depends on that provenance.

## Slugs

A slug names one candidate and must be a valid C identifier once non-word
characters are replaced, because it becomes part of `TORCH_EXTENSION_NAME`.
Stick to letters, digits, and underscores. Avoid dots. The convention the
generator uses is `<level>_task<id>_k<rank>`, where rank 0 is the fastest
candidate for that task, but the harness only requires uniqueness.

Candidates are grouped into tasks by the `level` and `task_id` fields of
`meta.json`, not by the slug. Ranking and inversion analysis happen within a
task, so two candidates only compete if those two fields match.

## kernel.cu

A standard PyTorch C++ extension. It must define a `forward` function bound
through `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`, and `forward` must accept
exactly the tensors that `ref.py`'s `get_inputs()` returns, in order, and
return a single tensor.

```cpp
torch::Tensor forward(torch::Tensor A, torch::Tensor B) { ... }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "my kernel");
}
```

Two properties of a candidate affect measurement and are worth knowing about.
A candidate that launches on the default stream co-executes with the co-runner
as intended. A candidate that creates its own streams and blocks on them inside
`forward` can hang the CUDA context beyond the reach of the in-process
watchdog, which is why each candidate is measured in a subprocess under a hard
timeout. Such candidates are recorded as `hung` rather than crashing the sweep.

## ref.py

The correctness and input-shape authority, following the KernelBench
convention.

```python
import torch, torch.nn as nn

class Model(nn.Module):
    def forward(self, A, B):
        return torch.matmul(A, B)

def get_inputs():
    return [torch.randn(2048, 2048), torch.randn(2048, 2048)]

def get_init_inputs():
    return []          # constructor arguments for Model
```

The harness calls `Model(*get_init_inputs())`, moves `get_inputs()` tensors to
the device, and compares `forward` output against the reference at absolute and
relative tolerance 1e-2 over three trials with fresh seeds. Verification runs
in the measuring process before timing, so a candidate that another benchmark
marked correct still has to pass here. The corpus generator currently restricts
itself to tasks where `get_init_inputs()` returns an empty list, which keeps the
calling convention unambiguous, but the harness itself handles constructor
arguments.

Inputs are regenerated per trial with fixed seeds, so `get_inputs` must be
deterministic given the seed and must not depend on module state.

## meta.json

```json
{
  "level": "1",
  "task_id": 4,
  "op_name": "4_Matrix_vector_multiplication_",
  "kernel_rank": 0,
  "kernel_name": "...",
  "archive_runtime_ms": 0.065,
  "archive_max_diff": "0.001",
  "code_hash": "37450a9798b6"
}
```

`level` and `task_id` group candidates into tasks and must agree across the
candidates you want compared. `op_name` is a human label for reports.
`kernel_rank` records the candidate's position in whatever pre-ranking your
source supplies and is not used for analysis. `code_hash` disambiguates build
directories, so give distinct candidates distinct hashes. The remaining fields
are provenance the harness copies into results without interpreting.

## manifest.json

A list of the per-candidate metadata, each entry carrying its `slug` alongside
the `meta.json` fields. Candidates are measured in manifest order, which
`--limit` truncates.

```json
[
  {"slug": "1_task4_k0", "level": "1", "task_id": 4, "code_hash": "37450a9798b6", "...": "..."},
  {"slug": "1_task4_k1", "level": "1", "task_id": 4, "code_hash": "5b79bd28241a", "...": "..."}
]
```

## Minimum viable corpus

Inversion analysis needs at least two candidates sharing `level` and `task_id`.
A single candidate per task still yields fragility measurements, and the report
marks the task unrankable.
