"""dryrun_stage1.py — run the real Stage-1 training entrypoint with a STUB encoder.

This is `tinyllava/train/train.py`'s `train()` verbatim, with exactly one thing
swapped: `BulkFormerVisionTower`'s encoder build/load helpers are replaced by the
shape-correct stub from `integration/smoke_test.py`. Everything else — argument
parsing, TinyLlavaConfig construction, the data path, the connector, the
training recipe, HF Trainer — is the production code path.

Why a stub: the real BulkFormer encoder needs torch_geometric + torch_sparse +
torch_scatter, which have no working macOS-arm64 build here (see
`integration/smoke_test_result.md`). The real encoder IS verified separately on
the CUDA box via `python -m integration.smoke_test --real-encoder`.

Scope: this validates that `scripts/train/train_stage1.sh`'s FLAGS and the
Stage-1 training loop are correct. It does NOT validate the real encoder or the
real Vicuna-7B backbone. Never use this to produce a real checkpoint.

    python -m integration.dryrun_stage1 --data_path ... --max_steps 5 ...
"""

from __future__ import annotations

import os
import sys

import torch.distributed as dist

from integration.smoke_test import _patch_encoder


def _init_single_process_group():
    """tinyllava.utils.logging.log() calls dist.get_rank() unconditionally, so a
    process group must exist. In production `deepspeed` sets one up; without a
    launcher we create a 1-rank gloo group ourselves."""
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29555")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def main():
    bar = "=" * 72
    print(bar, file=sys.stderr)
    print("STAGE-1 DRY RUN — STUB BulkFormer encoder. Not a real training run.", file=sys.stderr)
    print(bar, file=sys.stderr)
    _patch_encoder()
    _init_single_process_group()

    from tinyllava.train.train import train
    train()


if __name__ == "__main__":
    main()
