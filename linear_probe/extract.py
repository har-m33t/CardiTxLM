"""extract.py — step 3 of the linear-probe stage.

For each of the 5 BulkFormer variants, forward-pass every sample in the union
of the positive pool + the two negative pools, mean-pool over genes to a
sample-level embedding, and cache one parquet per variant. Downstream
StratifiedGroupKFold reads these parquets — extracting once here avoids
re-running the frozen encoder inside every fold × variant × task combination.

Pipeline per sample:
    1. read raw counts from ARCHS4 H5 (shape [gene_length_in_H5])
    2. TPM-normalize by gene length + total counts, then log1p
       (BulkFormer's `normalize_data` in the extract-feature notebook)
    3. reorder to BulkFormer's 20,010-gene vocab; genes missing from the H5
       get the -10 mask token (`main_gene_selection` in the notebook)
    4. batch through the frozen encoder → per-token embedding [B, 20010, dim+3]
    5. mean-pool over the gene axis → [B, dim+3] sample embedding

Uniform pipeline, sane defaults, CPU-only for correctness. A `--device mps`
knob exists but MPS fails on BulkFormer today because `torch-sparse` (used
by GCNConv) has CPU-only kernels — running on MPS will crash; keep it CPU
until GCNConv gets an MPS-compatible replacement.

CLI
---
    # toy validation across all 5 variants, 16 samples per pool
    python -m linear_probe.extract --n-per-pool 16 --batch-size 4

    # real run, only the 37M variant to start
    python -m linear_probe.extract --variants BulkFormer-37M
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BULKFORMER_REPO = REPO / "bulkencoders" / "BulkFormer"
CHECKPOINTS_ROOT = REPO / "bulkencoders" / "checkpoints" / "bulkformer"
DEFAULT_H5 = REPO / "eda" / "dataset" / "cvd_data" / "archs4" / "human_gene_v2.latest.h5"
DEFAULT_LABELS = HERE / "probe_sample_labels.parquet"
DEFAULT_OUTDIR = HERE / "embeddings"

# See bulkencoders/BulkFormer/model/config.py.
FIXED_PARAMS = {"bins": 0, "gb_repeat": 1, "bin_head": 12, "full_head": 8, "gene_length": 20010}


@dataclass(frozen=True)
class Variant:
    name: str
    ckpt_filename: str
    dim: int
    p_repeat: int


VARIANTS: dict[str, Variant] = {v.name: v for v in (
    Variant("BulkFormer-37M",  "BulkFormer-37M.pt",  dim=128, p_repeat=1),
    Variant("BulkFormer-50M",  "BulkFormer-50M.pt",  dim=256, p_repeat=2),
    Variant("BulkFormer-93M",  "BulkFormer-93M.pt",  dim=512, p_repeat=6),
    Variant("BulkFormer-127M", "BulkFormer-127M.pt", dim=640, p_repeat=8),
    Variant("BulkFormer-147M", "BulkFormer-147M.pt", dim=640, p_repeat=12),
)}


def _log() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger("linear_probe.extract")


# ----- sample selection ---------------------------------------------------

def build_sample_manifest(labels_path: Path, neg_ratio: int, n_per_pool: int | None,
                          seed: int, logger: logging.Logger) -> pd.DataFrame:
    """Union of positive + both negative pools, with pool tags per sample.

    `n_per_pool` caps the size of each of the three pools independently (for
    toy validation); when `None`, uses the full positive pool and negative
    pool (b) plus a `neg_ratio × n_positives` sub-sample of pool (a).
    """
    labels = pd.read_parquet(labels_path)

    pos = labels.loc[labels["is_positive"]].copy()
    neg_a = labels.loc[labels["is_neg_whole_corpus"]].copy()
    neg_b = labels.loc[labels["is_neg_hard"]].copy()

    if n_per_pool is not None:
        rng = np.random.default_rng(seed)
        pos   = pos.iloc[rng.permutation(len(pos))[:n_per_pool]]
        neg_a = neg_a.iloc[rng.permutation(len(neg_a))[:n_per_pool]]
        neg_b = neg_b.iloc[rng.permutation(len(neg_b))[:n_per_pool]]
        logger.info(f"toy mode: {len(pos)} pos + {len(neg_a)} neg_a + {len(neg_b)} neg_b")
    else:
        rng = np.random.default_rng(seed)
        n_neg_a_target = int(neg_ratio * len(pos))
        n_neg_a_target = min(n_neg_a_target, len(neg_a))
        take = rng.permutation(len(neg_a))[:n_neg_a_target]
        neg_a = neg_a.iloc[take]
        logger.info(f"full mode: {len(pos)} positives, "
                    f"{len(neg_a)} neg_a (=~{neg_ratio}× positives, capped at pool size), "
                    f"{len(neg_b)} neg_b (all)")

    pos["pool"]   = "positive"
    neg_a["pool"] = "neg_whole_corpus"
    neg_b["pool"] = "neg_hard"

    keep = ["sample_index", "geo_accession", "series_id", "cvd_subtype",
            "is_positive", "is_neg_whole_corpus", "is_neg_hard", "pool"]
    combined = pd.concat([pos[keep], neg_a[keep], neg_b[keep]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["sample_index"], keep="first")
    combined = combined.sort_values("sample_index").reset_index(drop=True)
    logger.info(f"union across pools (deduplicated): {len(combined)} samples")
    return combined


# ----- H5 counts + normalization ------------------------------------------

def load_bulkformer_vocab(logger: logging.Logger) -> tuple[list[str], dict[str, int]]:
    """The BulkFormer canonical 20,010-gene vocabulary + a name→length map."""
    gene_info = pd.read_csv(CHECKPOINTS_ROOT / "support" / "bulkformer_gene_info.csv")
    vocab = gene_info["ensg_id"].tolist()
    if len(vocab) != FIXED_PARAMS["gene_length"]:
        raise RuntimeError(f"vocab size mismatch: {len(vocab)} vs expected {FIXED_PARAMS['gene_length']}")

    length_df = pd.read_csv(CHECKPOINTS_ROOT / "support" / "gene_length_df.csv")
    length_dict = dict(zip(length_df["ensg_id"].astype(str), length_df["length"].astype(int)))
    logger.info(f"vocab={len(vocab)} genes, gene-length dict={len(length_dict)} entries")
    return vocab, length_dict


def _decode_h5_bytes(arr: np.ndarray) -> list[str]:
    return [x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in arr]


def normalize_and_align(counts: np.ndarray, h5_gene_symbols: list[str],
                        vocab: list[str], length_dict: dict[str, int],
                        logger: logging.Logger) -> tuple[np.ndarray, float]:
    """Convert raw counts (shape [B, N_h5_genes]) to a BulkFormer-shaped
    log(TPM+1) matrix (shape [B, 20010]), with -10 mask for missing genes.

    Returns (aligned matrix, mask_prob). `mask_prob` is the fraction of vocab
    genes absent from the H5 — passed to `model.forward(..., mask_prob=...)`
    so the model treats those positions as truly masked.
    """
    # TPM normalize.
    gene_lengths_kb = np.array(
        [length_dict.get(gid, 1000) / 1000.0 for gid in h5_gene_symbols],
        dtype=np.float64,
    )
    rate = counts.astype(np.float64) / gene_lengths_kb[None, :]
    sample_totals = rate.sum(axis=1, keepdims=True)
    sample_totals[sample_totals == 0] = 1e-6
    tpm = rate / sample_totals * 1e6
    log_tpm = np.log1p(tpm)

    # Align to BulkFormer vocab.
    h5_gene_to_col = {g: i for i, g in enumerate(h5_gene_symbols)}
    aligned = np.full((counts.shape[0], len(vocab)), -10.0, dtype=np.float32)
    missing = 0
    for j, gid in enumerate(vocab):
        col = h5_gene_to_col.get(gid)
        if col is None:
            missing += 1
            continue
        aligned[:, j] = log_tpm[:, col].astype(np.float32)
    mask_prob = missing / len(vocab)
    logger.info(f"aligned to BulkFormer vocab: {missing}/{len(vocab)} vocab genes "
                f"missing from H5 (mask_prob={mask_prob:.4f})")
    return aligned, mask_prob


def _read_one_column(args: tuple[str, int]) -> np.ndarray:
    """Read a single sample's full gene column. Module-level so it's picklable
    for multiprocessing."""
    h5_path, sample_idx = args
    with h5py.File(h5_path, "r") as f:
        return np.asarray(f["data/expression"][:, sample_idx], dtype=np.uint32)


class ArchS4CountReader:
    """Random-access reader over ARCHS4's `data/expression` for a fixed
    sample index list. Rows are genes, columns are samples in the on-disk
    layout, so we transpose after reading a batch.

    The H5 is chunked as (2000 genes, 1 sample) — every column read pulls 34
    separately gzip-compressed chunks over the network filesystem. A single
    h5py fancy-index read serializes those chunk fetches (~3 samples/s).
    Reading columns in a worker pool instead parallelizes the network I/O
    directly; measured ~8x throughput at 16 workers vs. serial reads.
    """

    N_WORKERS = 16

    def __init__(self, h5_path: Path, sample_indices: np.ndarray, logger: logging.Logger):
        self.h5_path = h5_path
        self.sample_indices = sample_indices
        self.logger = logger
        with h5py.File(h5_path, "r") as f:
            self.h5_gene_symbols = _decode_h5_bytes(f["meta/genes/ensembl_gene"][:])
            self.n_genes_h5 = len(self.h5_gene_symbols)
            logger.info(f"H5 gene axis: {self.n_genes_h5} genes")
        self._pool = multiprocessing.Pool(self.N_WORKERS)

    def read_batch(self, batch_sample_positions: np.ndarray) -> np.ndarray:
        """Read raw counts for a batch of sample-indices; returns shape
        [batch, n_genes_h5]."""
        idx = np.asarray(batch_sample_positions, dtype=np.int64)
        args = [(str(self.h5_path), int(i)) for i in idx]
        columns = self._pool.map(_read_one_column, args)   # order-preserving
        return np.stack(columns, axis=0)                   # [batch, n_genes_h5]

    def close(self) -> None:
        self._pool.close()
        self._pool.join()


# ----- model instantiation ------------------------------------------------

def _load_state_dict(model: torch.nn.Module, ckpt_path: Path) -> None:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    fixed = OrderedDict()
    for k, v in raw.items():
        fixed[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(fixed, strict=True)


def build_encoder(variant: Variant, device: torch.device, logger: logging.Logger):
    sys.path.insert(0, str(BULKFORMER_REPO))
    from torch_geometric.typing import SparseTensor
    from utils.BulkFormer import BulkFormer

    support = CHECKPOINTS_ROOT / "support"
    graph_rc = torch.load(support / "G_tcga.pt",       map_location="cpu", weights_only=False)
    graph_w  = torch.load(support / "G_tcga_weight.pt", map_location="cpu", weights_only=False)
    graph = SparseTensor(row=graph_rc[1], col=graph_rc[0], value=graph_w).t().to(device)
    gene_emb = torch.load(support / "esm2_feature_concat.pt", map_location="cpu", weights_only=False)

    params = {"dim": variant.dim, "p_repeat": variant.p_repeat,
              "graph": graph, "gene_emb": gene_emb, **FIXED_PARAMS}
    model = BulkFormer(**params).to(device)
    _load_state_dict(model, CHECKPOINTS_ROOT / "models" / variant.ckpt_filename)
    model.eval()
    logger.info(f"loaded {variant.name} on {device} "
                f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M trainable params)")
    return model


# ----- extraction loop ----------------------------------------------------

def extract_variant(variant: Variant, manifest: pd.DataFrame, reader: ArchS4CountReader,
                    vocab: list[str], length_dict: dict[str, int],
                    device: torch.device, batch_size: int, out_path: Path,
                    logger: logging.Logger, checkpoint_every: int = 50) -> dict:
    """Run frozen forward passes over `manifest`, mean-pool, write parquet.

    Checkpoints progress to `out_path.ckpt.npz` every `checkpoint_every` batches
    so an interrupted run resumes from the last checkpoint instead of redoing
    the whole variant.
    """
    model = build_encoder(variant, device, logger)

    n = len(manifest)
    emb_dim = variant.dim + 3
    sample_positions = manifest["sample_index"].to_numpy()

    ckpt_path = out_path.with_suffix(".ckpt.npz")
    start_at = 0
    emb_out = np.zeros((n, emb_dim), dtype=np.float32)
    all_mask_probs: list[float] = []
    if ckpt_path.exists():
        ckpt = np.load(ckpt_path)
        emb_out = ckpt["emb_out"]
        start_at = int(ckpt["next_start"])
        all_mask_probs = list(ckpt["mask_probs"])
        logger.info(f"[{variant.name}] resuming from checkpoint at {start_at}/{n} samples")

    t0 = time.perf_counter()
    for i, start in enumerate(range(start_at, n, batch_size)):
        end = min(start + batch_size, n)
        batch_positions = sample_positions[start:end]
        counts = reader.read_batch(batch_positions)          # [B, N_h5]
        aligned, mask_prob = normalize_and_align(counts, reader.h5_gene_symbols,
                                                 vocab, length_dict, logger)
        x = torch.from_numpy(aligned).to(device)             # [B, 20010]

        with torch.no_grad():
            gene_emb = model(x, mask_prob=mask_prob, output_expr=False)   # [B, 20010, dim+3]
            sample_emb = gene_emb.mean(dim=1)                             # [B, dim+3]
        emb_cpu = sample_emb.detach().float().cpu().numpy()
        emb_out[start:end] = emb_cpu
        all_mask_probs.append(mask_prob)

        seen = end
        elapsed = time.perf_counter() - t0
        rate = (seen - start_at) / max(elapsed, 1e-6)
        eta_s = (n - seen) / max(rate, 1e-6)
        logger.info(f"[{variant.name}] {seen}/{n} samples in {elapsed:.1f}s "
                    f"({rate:.2f}/s, ETA {eta_s / 60:.1f} min)")

        if (i + 1) % checkpoint_every == 0 or seen == n:
            np.savez(ckpt_path, emb_out=emb_out, next_start=seen,
                     mask_probs=np.array(all_mask_probs))

    emb = emb_out   # [n, dim+3]
    total_seconds = time.perf_counter() - t0

    # Build the embedding columns as a single DataFrame and concat once — inserting
    # 640+ columns one at a time triggers a per-insert copy inside pandas.
    emb_df = pd.DataFrame(emb, columns=[f"e{j:04d}" for j in range(emb.shape[1])])
    out_df = pd.concat([manifest.reset_index(drop=True), emb_df], axis=1)
    out_df.to_parquet(out_path, index=False)
    ckpt_path.unlink(missing_ok=True)

    stats = {
        "variant": variant.name,
        "n_samples": int(n),
        "embedding_dim": int(emb.shape[1]),
        "expected_dim": variant.dim + 3,
        "batch_size": batch_size,
        "device": str(device),
        "seconds": round(total_seconds, 2),
        "samples_per_second": round(n / max(total_seconds, 1e-6), 3),
        "mask_prob_mean": float(np.mean(all_mask_probs)),
        "mask_prob_std":  float(np.std(all_mask_probs)),
        "output_std":   float(emb.std()),
        "output_mean":  float(emb.mean()),
        "any_nan":      bool(np.isnan(emb).any()),
        "any_inf":      bool(np.isinf(emb).any()),
        "out_path": str(out_path.relative_to(REPO)),
    }
    logger.info(f"[{variant.name}] wrote {out_path} — {stats}")
    del model
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract BulkFormer embeddings (step 3).")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()),
                        help='Variants to run (e.g. "BulkFormer-37M BulkFormer-50M"). Default: all 5.')
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                        help="Torch device. MPS crashes on BulkFormer today — leave as cpu.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-per-pool", type=int, default=None,
                        help="Cap each of the 3 pools at N samples for toy validation. "
                             "None uses full positive + full neg_hard + neg_ratio×positives from neg_a.")
    parser.add_argument("--neg-ratio", type=int, default=3,
                        help="Non-CVD negative pool size, in multiples of the positive pool (full mode only).")
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args(argv)

    logger = _log()
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    manifest = build_sample_manifest(args.labels, args.neg_ratio, args.n_per_pool, args.seed, logger)
    manifest_path = args.outdir / "sample_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    logger.info(f"wrote {manifest_path}")

    vocab, length_dict = load_bulkformer_vocab(logger)
    reader = ArchS4CountReader(args.h5, manifest["sample_index"].to_numpy(), logger)

    per_variant_stats = []
    try:
        for name in args.variants:
            if name not in VARIANTS:
                logger.error(f"unknown variant {name!r}; choose from {list(VARIANTS)}")
                return 2
            variant = VARIANTS[name]
            out_path = args.outdir / f"embeddings_{name}.parquet"
            stats = extract_variant(variant, manifest, reader, vocab, length_dict,
                                    device, args.batch_size, out_path, logger)
            per_variant_stats.append(stats)
    finally:
        reader.close()

    manifest_out = {
        "device": str(device),
        "batch_size": args.batch_size,
        "n_per_pool": args.n_per_pool,
        "neg_ratio": args.neg_ratio,
        "seed": args.seed,
        "n_samples_in_manifest": int(len(manifest)),
        "variants": per_variant_stats,
    }
    (args.outdir / "extraction_manifest.json").write_text(json.dumps(manifest_out, indent=2))
    logger.info(f"wrote {args.outdir / 'extraction_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
