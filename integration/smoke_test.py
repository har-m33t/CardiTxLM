"""smoke_test.py — wiring-correctness check for the BulkFormer tower integration.

Two modes, selected explicitly on the command line:

  STUB mode  (default) — BulkFormer's encoder is replaced by a shape-correct
      stub returning `[B, 20010, dim+3]`. This runs anywhere (no
      torch_geometric / torch_sparse). It verifies the TinyLLaVA-side WIRING
      only; it does NOT verify the real encoder integration end-to-end.

  REAL mode  (--real-encoder) — loads the actual frozen BulkFormer checkpoint
      named by `integration/bulkformer_hf_config/config.json`, which is LOCKED to
      **BulkFormer-93M** (`BulkFormer-93M.pt`, embedding dim 512+3 = 515).
      Requires the torch_geometric + torch_sparse stack. This is the test that
      CLOSES the encoder-integration gap, and it MUST pass on the target training
      environment before Stage 1 pretraining starts. NOTE: the recorded pass in
      integration/smoke_test_result.md was against the retired 127M variant, so
      it must be re-run for 93M.

Both modes exercise: (A) dataset `.npy` branch → collated `[B, 20010]`;
(B) tower.forward pooling → `[B, 1, 515]`, encoder frozen; (C) full TinyLlava
forward → connector `[B, 1, hidden]`, LLM loss, backward.

Run (default stub):  python -m integration.smoke_test
Run (real encoder):  python -m integration.smoke_test --real-encoder
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CFG_DIR = HERE / "bulkformer_hf_config"
N_GENES = 20010
TINY_LLM = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# ---- stub the BulkFormer encoder (no torch_geometric needed) ---------------
import tinyllava.model.vision_tower.bulkformer as bf  # noqa: E402


class _StubBulkFormer(nn.Module):
    """Mimics BulkFormer.forward(x, mask_prob, output_expr)->[B, 20010, dim+3]."""

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.scale = nn.Parameter(torch.ones(emb_dim))  # a real param, so freezing is observable

    def forward(self, x, mask_prob=None, output_expr=False):
        b, g = x.shape
        return x.unsqueeze(-1).expand(b, g, self.emb_dim) * self.scale


def _patch_encoder():
    """Swap the real BulkFormer encoder for a shape-correct STUB.

    Confined entirely to this test process — it monkeypatches the tower module's
    build helpers here, and nothing in the training path (train.py →
    BulkFormerVisionTower) imports or calls this. A real training job therefore
    can NEVER pick up the stub. It only takes effect when this file's `main()`
    runs in the default (stub) mode.
    """
    def fake_build(variant):
        spec = bf._VARIANTS[variant]
        return _StubBulkFormer(spec["dim"] + 3), spec
    bf._build_bulkformer = fake_build
    bf._load_checkpoint = lambda model, path: None


def report(tag, **kv):
    print(f"[{tag}] " + " ".join(f"{k}={v}" for k, v in kv.items()))


def make_tiny_dataset(root: Path, n=4):
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    records = []
    for i in range(n):
        vec = rng.normal(size=N_GENES).astype(np.float32)
        np.save(img_dir / f"s{i}.npy", vec)
        label = "Yes" if i % 2 == 0 else "No"
        records.append({
            "id": f"s{i}", "image": f"s{i}.npy",
            "conversations": [
                {"from": "human", "value": "<image>"},
                {"from": "gpt", "value": f"{label}."},
            ],
        })
    (root / "data.json").write_text(json.dumps(records))
    return root / "data.json", img_dir


def _make_tinyllama_stub(scratch: Path) -> Path:
    """Save the tiny random Llama into a dir whose name contains 'tinyllama' so
    the substring-matching LLM factory selects the Llama loader."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    stub = scratch / "tinyllama_stub"
    if not (stub / "config.json").exists():
        AutoModelForCausalLM.from_pretrained(TINY_LLM).save_pretrained(stub)
        AutoTokenizer.from_pretrained(TINY_LLM).save_pretrained(stub)
    return stub


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="BulkFormer tower smoke test.")
    ap.add_argument("--real-encoder", action="store_true",
                    help="Load the REAL frozen BulkFormer checkpoint selected by "
                         "integration/bulkformer_hf_config (locked: BulkFormer-93M) "
                         "(requires torch_geometric + torch_sparse). Without this "
                         "flag a STUB encoder is used and encoder integration is "
                         "NOT verified.")
    args = ap.parse_args(argv)

    bar = "=" * 72
    if args.real_encoder:
        print(bar)
        print("SMOKE TEST MODE: REAL BulkFormer encoder — closes the encoder gap.")
        print(bar)
    else:
        print(bar)
        print("SMOKE TEST MODE: STUB encoder — WIRING ONLY.")
        print("  Encoder integration is NOT verified end-to-end in this mode.")
        print("  Before Stage 1 pretraining, re-run on the target training env")
        print("  with --real-encoder (needs torch_geometric + torch_sparse).")
        print(bar)
        _patch_encoder()

    from transformers import AutoConfig
    from tinyllava.model.vision_tower import VisionTowerFactory
    from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
    from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
    from tinyllava.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset

    scratch = REPO / "integration" / "_smoke_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    stub = _make_tinyllama_stub(scratch)

    # ---- (B) tower forward on a raw expression batch -----------------------
    cfg = AutoConfig.from_pretrained(str(CFG_DIR))
    cfg = getattr(cfg, "vision_config", cfg)
    tower = VisionTowerFactory("bulkformer")(cfg)
    tower.load_model(str(CFG_DIR))
    n_grad = sum(p.requires_grad for p in tower.parameters())
    x = torch.randn(3, N_GENES)
    tout = tower(x)
    report("B-tower", in_shape=tuple(x.shape), out_shape=tuple(tout.shape),
           frozen_params_with_grad=n_grad, finite=bool(torch.isfinite(tout).all()))
    assert tuple(tout.shape) == (3, 1, cfg.hidden_size), "tower output shape wrong"
    assert n_grad == 0, "encoder must be frozen"

    # ---- build the full TinyLlava model (real construction path) -----------
    margs = SimpleNamespace(
        model_name_or_path=str(stub), tokenizer_name_or_path=None,
        vision_tower=f"bulkformer:{CFG_DIR}", vision_tower2="",
        connector_type="transcript_linear", cache_dir=None,
        attn_implementation="eager", model_max_length=2048,
        mm_vision_select_layer=-2, mm_vision_select_feature="patch",
        image_aspect_ratio="square", tokenizer_use_fast=False,
    )
    mcfg = TinyLlavaConfig()
    mcfg.load_from_config(margs)
    model = TinyLlavaForConditionalGeneration(mcfg)
    model.load_llm(model_name_or_path=str(stub), cache_dir=None, attn_implementation="eager")
    model.load_vision_tower(model_name_or_path=str(CFG_DIR))
    model.load_connector(connector_type="transcript_linear")
    model.train()
    tokenizer = model.tokenizer

    # ---- (A) dataset .npy branch + collator --------------------------------
    data_json, img_dir = make_tiny_dataset(scratch, n=4)
    data_args = SimpleNamespace(conv_version="pretrain", image_folder=str(img_dir),
                                image_processor=model.vision_tower._image_processor,
                                image_aspect_ratio="square", is_multimodal=True,
                                image_grid_pinpoints=None, data_path=str(data_json))
    ds = LazySupervisedDataset(str(data_json), tokenizer, data_args)
    collate = DataCollatorForSupervisedDataset(tokenizer)
    batch = collate([ds[i] for i in range(len(ds))])
    report("A-dataset", images_shape=tuple(batch["images"].shape),
           input_ids_shape=tuple(batch["input_ids"].shape))
    assert tuple(batch["images"].shape) == (4, N_GENES), "collated expression batch wrong"

    # ---- (C) full model forward + backward ---------------------------------

    # freeze everything but the connector (stage-1 alignment regime)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.connector.parameters():
        p.requires_grad_(True)

    conn_out_dim = model.connector(tout.float()).shape[-1]
    report("C-connector", conn_in=cfg.hidden_size, conn_out=conn_out_dim,
           llm_hidden=model.config.hidden_size)

    batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                labels=batch["labels"], images=batch["images"])
    loss = out.loss
    loss.backward()
    conn_grad = next(p.grad for p in model.connector.parameters() if p.grad is not None)
    tower_grads = [p.grad for p in model.vision_tower.parameters() if p.grad is not None]
    report("C-step", loss=round(float(loss), 5),
           connector_grad_norm=round(float(conn_grad.norm()), 6),
           tower_grads=len(tower_grads))
    assert torch.isfinite(loss), "loss not finite"
    assert len(tower_grads) == 0, "frozen tower received gradients"

    if args.real_encoder:
        print("\nSMOKE TEST PASSED — REAL encoder. Wiring AND encoder integration verified.")
    else:
        print("\nSMOKE TEST PASSED — STUB encoder. Wiring verified; "
              "encoder integration NOT yet verified (re-run with --real-encoder).")


if __name__ == "__main__":
    sys.exit(main())
