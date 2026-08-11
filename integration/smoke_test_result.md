# BulkFormer tower — smoke-test status

## Status: **encoder integration verified end-to-end (Linux/CUDA, torch 2.8.0+cu128, 2026-08-11).**

Both parts of the claim now hold:

- ✅ **Wiring verified** — the TinyLLaVA-side integration (data path, tower
  pooling, connector, freezing, train step) runs end-to-end with a **stub**
  encoder.
- ✅ **Encoder integration verified end-to-end** — `python -m
  integration.smoke_test --real-encoder` loads the real frozen
  BulkFormer-127M checkpoint and passes: `[B-tower] out_shape=(3, 1, 643)`,
  `frozen_params_with_grad=0`, `loss=10.37624` (finite), `connector_grad_norm
  =0.859615`, `tower_grads=0`. Reproduced on a 2x RTX 4090 box (see repo
  history around 2026-08-11 for the dependency set that made this pass:
  torch_geometric/torch_scatter/torch_sparse wheels from
  data.pyg.org/whl/torch-2.8.0+cu128.html, performer_pytorch, and a
  `field(default_factory=...)` fix to the six `tinyllava/data/template/*.py`
  dataclasses, which used mutable/unhashable defaults that Python 3.11+'s
  stricter dataclass check rejects).

This clears the gate below — Stage 1 pretraining is no longer blocked on the
encoder-integration check itself (it may still be blocked on other things,
e.g. training data availability).

Reproduce (default = stub mode): `python -m integration.smoke_test`

## What the wiring test covered (STUB encoder)

Full integration surface on CPU with a tiny random Llama (LLM hidden size 16)
and 4 synthetic samples. BulkFormer's encoder forward was replaced by a
shape-correct stub returning `[B, 20010, dim+3]`.

| Stage | Boundary | Observed |
|---|---|---|
| **A. Data path** | `.npy` branch in `dataset.py` → collator | `images` **`(4, 20010)`**, `input_ids` `(4, 5)` |
| **B. Tower** | `BulkFormerVisionTower.forward` | `(3, 20010)` → **`(3, 1, 643)`**, finite; **0** encoder params require grad |
| **C. Connector** | `TranscriptLinearConnector` | `643` → **`16`** (LLM hidden) |
| **C. Train step** | tower → connector → LLM → loss → backward | **loss ≈ 10.36**, connector grad-norm ≈ 0.16–0.32, **0** gradients on the frozen tower |

Assertions: tower output is exactly `(B, 1, 643)`; encoder has no trainable
params; loss finite; `backward()` succeeds; frozen tower gets **no** gradients.

**Why the stub:** BulkFormer needs a full extra dependency set — at least
`torch_geometric` + `torch_sparse` + `torch_scatter` (its `GCNConv`) **and**
`performer_pytorch` (surfaced when `--real-encoder` tries to import
`utils.BulkFormer`), likely more per `bulkencoders/BulkFormer/bulkformer.yaml`.
Locally, `torch_geometric` installs but `torch_sparse`/`torch_scatter` have no
prebuilt wheel for this platform (macOS arm64 + torch 2.0.1) and fail to compile
from source — torch 2.0.1's C10 headers are rejected by Apple clang 21
(`'is_arithmetic' cannot be specialized`) — and `performer_pytorch` is absent
too. The linear-probe stage ran BulkFormer in a separate Python 3.12 conda env
(`bulkencoders/BulkFormer/bulkformer.yaml`); that is the known-good environment.

The stub is **explicitly gated**: it is applied only by `_patch_encoder()` inside
`integration/smoke_test.py`, in the default (stub) mode, and is confined to that
test process. Nothing in the training path (`train.py` →
`BulkFormerVisionTower`) references it, so a real training job cannot silently
run against the stub. Stub vs. real is a named CLI mode (`--real-encoder`), with
a loud banner printed either way.

## Required before Stage 1 pretraining (the gate that closes the encoder gap)

**On the target training environment (HPC / Linux / CUDA)**, run the *same*
smoke test with the real encoder — same test, right environment, real weights:

```
python -m integration.smoke_test --real-encoder
```

This removes the stub, loads `BulkFormer-127M.pt`, and asserts the real encoder
produces `[B, 1, 643]` and trains a step through the frozen tower. Stage 1
pretraining does **not** start until this passes. When it does, update the status
line at the top of this file to "encoder integration verified end-to-end
(<env>, <date>)".

### Pre-check BulkFormer's full dependency set on the HPC *now* (cheap, avoids mid-job surprises)

Wheel availability is **version-specific** — Linux+CUDA does not automatically
guarantee it — and PyG is not the only extra dep. Before relying on it, confirm
BulkFormer imports against the HPC's actual torch/CUDA versions:

```
# fill in the HPC's real values, e.g. torch 2.1.0 + cu121
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA}.html
pip install performer_pytorch
# confirms the PyG stack AND BulkFormer's own module import cleanly:
python -c "import torch_sparse, torch_scatter; from torch_geometric.typing import SparseTensor; print('PyG OK')"
python -c "import sys; sys.path.insert(0, 'bulkencoders/BulkFormer'); from utils.BulkFormer import BulkFormer; print('BulkFormer import OK')"
```

Simplest robust path: **reproduce the linear-probe conda env**
(`bulkencoders/BulkFormer/bulkformer.yaml`, Python 3.12) — the environment where
BulkFormer already runs — rather than piecing deps onto the TinyLLaVA `.venv`.
If no matching PyG wheel exists for the HPC's torch/CUDA combo, resolve it
(adjust the torch version to one with published wheels, or build with a
compatible compiler) **before** queuing the job, not after.

## Environment note (local `.venv`)

`torch_geometric==2.6.1` was installed into `.venv` (pure Python, harmless).
`torch_sparse`/`torch_scatter` were **not** installed (build failure above), so
`--real-encoder` is not runnable in this local `.venv` as-is — it is intended for
the HPC gate above.
