"""BulkFormer vision tower.

Wraps the frozen BulkFormer transcriptomics encoder as a TinyLLaVA "vision
tower". Unlike the image towers, the input is a `[B, 20010]` gene-expression
vector (log1p-TPM, missing genes = -10.0), not pixels — so `forward` and
`_load_model` are overridden. The per-gene encoder output `[B, 20010, dim+3]`
is mean-pooled over genes to a single `[B, 1, dim+3]` token per sample (20,010
tokens would overflow the LLM context).

The encoder is loaded from its canonical `.pt` checkpoint in `__init__` and kept
frozen throughout (standing project decision). Construction/loading mirrors
`linear_probe/extract.py`. See `integration/repo_findings.md` §5/§8.
"""

import sys
from collections import OrderedDict
from pathlib import Path

import torch

from . import register_vision_tower
from .base import VisionTower

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]  # vision_tower -> model -> tinyllava -> <repo root>
_BULKFORMER_REPO = _REPO / "bulkencoders" / "BulkFormer"
_CKPT_ROOT = _REPO / "bulkencoders" / "checkpoints" / "bulkformer"

# Per-variant architecture (dim, p_repeat, checkpoint file). Shared fixed params
# match linear_probe/extract.py. Embedding dim exposed to the connector = dim + 3.
_VARIANTS = {
    "BulkFormer-37M":  dict(dim=128, p_repeat=1,  ckpt="BulkFormer-37M.pt"),
    "BulkFormer-50M":  dict(dim=256, p_repeat=2,  ckpt="BulkFormer-50M.pt"),
    "BulkFormer-93M":  dict(dim=512, p_repeat=6,  ckpt="BulkFormer-93M.pt"),
    "BulkFormer-127M": dict(dim=640, p_repeat=8,  ckpt="BulkFormer-127M.pt"),
    "BulkFormer-147M": dict(dim=640, p_repeat=12, ckpt="BulkFormer-147M.pt"),
}
_FIXED = dict(bins=0, gb_repeat=1, bin_head=12, full_head=8, gene_length=20010)


class _ExpressionProcessor:
    """Placeholder processor for the transcriptomic modality.

    BulkFormer consumes raw `[20010]` expression vectors that the data pipeline
    loads directly (the `.npy` branch in `tinyllava/data/dataset.py`), so no
    image-style preprocessing runs. This shim only satisfies attribute accesses
    (`crop_size`/`size`) that other call sites assume a processor exposes.
    """

    crop_size = {"height": 1, "width": 1}
    size = {"height": 1, "width": 1}
    image_mean = [0.0, 0.0, 0.0]

    def __call__(self, x, return_tensors=None):
        return {"pixel_values": [x]}


def _build_bulkformer(variant_name):
    if str(_BULKFORMER_REPO) not in sys.path:
        sys.path.insert(0, str(_BULKFORMER_REPO))
    from torch_geometric.typing import SparseTensor
    from utils.BulkFormer import BulkFormer

    support = _CKPT_ROOT / "support"
    graph_rc = torch.load(support / "G_tcga.pt", map_location="cpu", weights_only=False)
    graph_w = torch.load(support / "G_tcga_weight.pt", map_location="cpu", weights_only=False)
    graph = SparseTensor(row=graph_rc[1], col=graph_rc[0], value=graph_w).t()
    gene_emb = torch.load(support / "esm2_feature_concat.pt", map_location="cpu", weights_only=False)

    spec = _VARIANTS[variant_name]
    model = BulkFormer(dim=spec["dim"], p_repeat=spec["p_repeat"],
                       graph=graph, gene_emb=gene_emb, **_FIXED)
    return model, spec


def _load_checkpoint(model, ckpt_path):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    fixed = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in raw.items())
    model.load_state_dict(fixed, strict=True)


@register_vision_tower('bulkformer')
class BulkFormerVisionTower(VisionTower):
    def __init__(self, cfg):
        super().__init__(cfg)
        # BulkFormer-93M is the LOCKED encoder scale for this project (see
        # llm_training_plan.md §2). The fallback here matters only if a config
        # omits `bulkformer_variant`; the shipped
        # integration/bulkformer_hf_config/config.json sets it explicitly.
        variant = getattr(cfg, 'bulkformer_variant', 'BulkFormer-93M')
        if variant not in _VARIANTS:
            raise ValueError(f"unknown bulkformer_variant {variant!r}; "
                             f"choose from {list(_VARIANTS)}")
        model, spec = _build_bulkformer(variant)
        _load_checkpoint(model, _CKPT_ROOT / "models" / spec["ckpt"])
        model.eval()
        self._vision_tower = model
        self._image_processor = _ExpressionProcessor()
        self._variant = variant
        self.embed_dim = spec["dim"] + 3  # == cfg.hidden_size (515 for the locked 93M)

    def _load_model(self, vision_tower_name, **kwargs):
        # BulkFormer is frozen and fully loaded from its canonical checkpoint in
        # __init__. There are no stage-specific tower weights to (re)load, so we
        # ignore any pretrained_vision_tower_path the base flow would use.
        kwargs.pop('pretrained_vision_tower_path', None)

    def forward(self, x, **kwargs):
        # x: [B, 20010] (or [B, 1, 20010] after batch stacking) log1p-TPM.
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        in_dtype = x.dtype

        # PRE-ENCODED INPUT PASSTHROUGH.
        # integration/precompute_encoder_cache.py writes pooled [dim+3] vectors
        # for each unique sample; pointing --image_folder at that directory feeds
        # them straight in. The encoder is frozen, runs under no_grad and with
        # mask_prob=0.0, so its output per input is deterministic — encoding once
        # up front is exactly equivalent to re-encoding every epoch, and avoids
        # re-running the tower ~23x per unique vector on the Stage-1 corpus.
        # Widths are unambiguous: raw input is gene_length (20010), encoded is
        # dim+3 (515 for BulkFormer-93M).
        if x.shape[-1] == self.embed_dim:
            return x.unsqueeze(1).to(in_dtype)  # [B, 1, dim+3]
        # Two things a GPU/DeepSpeed run breaks that a CPU run never exposes.
        # Both are repaired here to match linear_probe/extract.py, the proven
        # CUDA path for this encoder (see its build_encoder: graph .to(device),
        # model .to(device), fp32 throughout).
        #
        # 1. DeepSpeed with fp16.enabled calls module.half() over the WHOLE
        #    model, including this frozen encoder. Its weights then disagree
        #    with the fp32 input below ("mat1 and mat2 must have the same
        #    dtype, but got Float and Half"). The encoder is frozen and carries
        #    no optimizer state, so keeping it in fp32 costs nothing and avoids
        #    unverified fp16 numerics through its graph convolutions.
        # 2. BulkFormer stores its interaction graph as a PLAIN ATTRIBUTE
        #    (utils/BulkFormer.py: `self.graph = graph`), not a buffer, so
        #    nn.Module.to()/.cuda() never moves it. It stays on CPU while the
        #    activations are on CUDA, and torch_sparse then dispatches to its
        #    CPU spmm and asserts "mat must be CPU tensor". The tower is built
        #    before the trainer moves the model, so this is done lazily here.
        if next(self._vision_tower.parameters()).dtype != torch.float32:
            self._vision_tower.float()
        if getattr(self, '_graph_device', None) != x.device:
            self._vision_tower.graph = self._vision_tower.graph.to(x.device)
            self._graph_device = x.device
        with torch.no_grad():
            out = self._vision_tower(x.float(), mask_prob=0.0, output_expr=False)  # [B, 20010, dim+3]
            sample = out.mean(dim=1)  # [B, dim+3]
        return sample.unsqueeze(1).to(in_dtype)  # [B, 1, dim+3]
