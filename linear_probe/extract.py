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


class ArchS4CountReader:
    """Random-access reader over ARCHS4's `data/expression` for a fixed
    sample index list. Rows are genes, columns are samples in the on-disk
    layout, so we transpose after reading a batch.
    """

    def __init__(self, h5_path: Path, sample_indices: np.ndarray, logger: logging.Logger):
        self.h5_path = h5_path
        self.sample_indices = sample_indices
        self.logger = logger
        with h5py.File(h5_path, "r") as f:
            self.h5_gene_symbols = _decode_h5_bytes(f["meta/genes/ensembl_gene"][:])
            self.n_genes_h5 = len(self.h5_gene_symbols)
            logger.info(f"H5 gene axis: {self.n_genes_h5} genes")

    def read_batch(self, batch_sample_positions: np.ndarray) -> np.ndarray:
        """Read raw counts for a batch of sample-indices; returns shape
        [batch, n_genes_h5]."""
        idx = np.asarray(batch_sample_positions, dtype=np.int64)
        # h5py fancy indexing requires sorted indices for good performance.
        order = np.argsort(idx)
        sorted_idx = idx[order]
        with h5py.File(self.h5_path, "r") as f:
            block = f["data/expression"][:, sorted_idx]  # [n_genes_h5, batch]
        # Restore original batch order.
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        block = block[:, inv]
        return block.T  # [batch, n_genes_h5]


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


# ----- shard checkpointing ------------------------------------------------
#
# The larger variants take tens of hours of CPU on a workstation (93M measured
# at ~0.42 samples/s, i.e. ~38 h for the 57,207-sample manifest). Holding every
# embedding in memory and writing one parquet at the very end means any crash,
# OOM, or reboot in that window throws the whole run away. Embeddings are
# therefore flushed to numbered shards as they are produced, and a restart
# resumes from the first uncovered sample.

def _shard_dir(out_path: Path, variant: Variant) -> Path:
    return out_path.parent / f"_shards_{variant.name}"


def _load_shards(shard_dir: Path, batch_size: int, n: int,
                 logger: logging.Logger) -> tuple[list[np.ndarray], list[float], int]:
    """Return (embedding blocks, mask_probs, n_samples_covered) from a prior run.

    Shards are only trusted as a contiguous prefix: the first gap stops the
    resume, so a partially-written shard can never silently misalign the
    embedding rows against the manifest.
    """
    meta_path = shard_dir / "_shard_meta.json"
    if not shard_dir.is_dir() or not meta_path.exists():
        return [], [], 0

    meta = json.loads(meta_path.read_text())
    if meta.get("n_samples") != n:
        logger.warning(f"shard dir {shard_dir} was written for a "
                       f"{meta.get('n_samples')}-sample manifest, current run has {n} "
                       "— ignoring shards and restarting this variant")
        return [], [], 0

    embs: list[np.ndarray] = []
    mask_probs: list[float] = []
    covered = 0
    for shard in sorted(shard_dir.glob("part_*.parquet")):
        start = int(shard.stem.split("_")[1])
        if start != covered:
            logger.warning(f"shard gap at sample {covered} (next shard starts at {start}) "
                           "— resuming from the gap, later shards will be recomputed")
            break
        try:
            df = pd.read_parquet(shard)
        except Exception as exc:                       # truncated by a hard kill
            logger.warning(f"unreadable shard {shard.name} ({exc}) — resuming from {covered}")
            break
        emb_cols = [c for c in df.columns if c.startswith("e")]
        embs.append(df[emb_cols].to_numpy(dtype=np.float32))
        mask_probs.extend(df["_mask_prob"].tolist())
        covered += len(df)

    if covered:
        logger.info(f"resume: {covered}/{n} samples recovered from {len(embs)} shards "
                    f"in {shard_dir.name}")
    return embs, mask_probs, covered


def _write_shard(shard_dir: Path, start: int, emb: np.ndarray,
                 mask_probs: list[float], sample_index: np.ndarray) -> None:
    df = pd.DataFrame(emb, columns=[f"e{j:04d}" for j in range(emb.shape[1])])
    df["_mask_prob"] = mask_probs
    df["_sample_index"] = sample_index
    tmp = shard_dir / f".part_{start:06d}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    # Rename is atomic on the same filesystem, so a shard is either absent or
    # complete — a kill mid-write can't leave a half-shard that resume trusts.
    tmp.replace(shard_dir / f"part_{start:06d}.parquet")


# ----- extraction loop ----------------------------------------------------

def extract_variant(variant: Variant, manifest: pd.DataFrame, reader: ArchS4CountReader,
                    vocab: list[str], length_dict: dict[str, int],
                    device: torch.device, batch_size: int, out_path: Path,
                    logger: logging.Logger, shard_every: int = 1000,
                    resume: bool = True) -> dict:
    """Run frozen forward passes over `manifest`, mean-pool, write parquet."""
    n = len(manifest)
    sample_positions = manifest["sample_index"].to_numpy()

    shard_dir = _shard_dir(out_path, variant)
    shard_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        all_embs, all_mask_probs, resume_from = _load_shards(shard_dir, batch_size, n, logger)
    else:
        all_embs, all_mask_probs, resume_from = [], [], 0
    (shard_dir / "_shard_meta.json").write_text(json.dumps(
        {"variant": variant.name, "n_samples": n, "batch_size": batch_size}, indent=2))

    if resume_from >= n:
        logger.info(f"[{variant.name}] all {n} samples already in shards — assembling parquet")
        model = None
    else:
        model = build_encoder(variant, device, logger)

    pending: list[np.ndarray] = []
    pending_mask: list[float] = []
    pending_start = resume_from

    t0 = time.perf_counter()
    for start in range(resume_from, n, batch_size):
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
        pending.append(emb_cpu)
        pending_mask.extend([mask_prob] * (end - start))

        if sum(len(p) for p in pending) >= shard_every or end >= n:
            block = np.vstack(pending)
            _write_shard(shard_dir, pending_start, block, pending_mask,
                         sample_positions[pending_start:pending_start + len(block)])
            all_embs.append(block)
            all_mask_probs.extend(pending_mask)
            logger.info(f"[{variant.name}] checkpointed shard at sample {pending_start}"
                        f"–{pending_start + len(block)}")
            pending_start += len(block)
            pending, pending_mask = [], []

        seen = end
        elapsed = time.perf_counter() - t0
        done_this_run = seen - resume_from
        rate = done_this_run / max(elapsed, 1e-6)
        eta_s = (n - seen) / max(rate, 1e-6)
        logger.info(f"[{variant.name}] {seen}/{n} samples in {elapsed:.1f}s "
                    f"({rate:.2f}/s, ETA {eta_s / 3600:.2f} h)")

    emb = np.vstack(all_embs)   # [n, dim+3]
    total_seconds = time.perf_counter() - t0

    if len(emb) != n:
        raise RuntimeError(f"[{variant.name}] assembled {len(emb)} embeddings for a "
                           f"{n}-sample manifest — refusing to write a misaligned parquet")

    # Build the embedding columns as a single DataFrame and concat once — inserting
    # 640+ columns one at a time triggers a per-insert copy inside pandas.
    emb_df = pd.DataFrame(emb, columns=[f"e{j:04d}" for j in range(emb.shape[1])])
    out_df = pd.concat([manifest.reset_index(drop=True), emb_df], axis=1)
    out_df.to_parquet(out_path, index=False)

    stats = {
        "variant": variant.name,
        "n_samples": int(n),
        "embedding_dim": int(emb.shape[1]),
        "expected_dim": variant.dim + 3,
        "batch_size": batch_size,
        "device": str(device),
        "seconds": round(total_seconds, 2),
        # Rate over what this invocation actually computed — on a resumed run,
        # dividing the full manifest by this run's wall clock would invent speed.
        "samples_per_second": round((n - resume_from) / max(total_seconds, 1e-6), 3),
        "mask_prob_mean": float(np.mean(all_mask_probs)),
        "mask_prob_std":  float(np.std(all_mask_probs)),
        "output_std":   float(emb.std()),
        "output_mean":  float(emb.mean()),
        "any_nan":      bool(np.isnan(emb).any()),
        "any_inf":      bool(np.isinf(emb).any()),
        "out_path": str(out_path.relative_to(REPO)) if REPO in out_path.parents else str(out_path),
        "resumed_from_sample": int(resume_from),
    }
    logger.info(f"[{variant.name}] wrote {out_path} — {stats}")

    # Only now that the assembled parquet is on disk are the shards redundant.
    for shard in shard_dir.glob("part_*.parquet"):
        shard.unlink()
    (shard_dir / "_shard_meta.json").unlink(missing_ok=True)
    shard_dir.rmdir()

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
    parser.add_argument("--shard-every", type=int, default=1000,
                        help="Flush a resumable shard every N samples. The large variants "
                             "run for tens of hours; this bounds what a crash costs.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing shards and recompute this variant from scratch.")
    parser.add_argument("--threads", type=int, default=None,
                        help="torch intra-op threads. Defaults to torch's own choice (4 on "
                             "this box); 10 measured ~2.5x faster for BulkFormer-93M.")
    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)

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
    for name in args.variants:
        if name not in VARIANTS:
            logger.error(f"unknown variant {name!r}; choose from {list(VARIANTS)}")
            return 2
        variant = VARIANTS[name]
        out_path = args.outdir / f"embeddings_{name}.parquet"
        stats = extract_variant(variant, manifest, reader, vocab, length_dict,
                                device, args.batch_size, out_path, logger,
                                shard_every=args.shard_every, resume=not args.no_resume)
        per_variant_stats.append(stats)

    # Merge into any existing manifest rather than replacing it — variants are
    # extracted in separate invocations (each takes hours), and a plain
    # overwrite silently drops the provenance of every previously-run variant.
    manifest_path = args.outdir / "extraction_manifest.json"
    prior_variants: list[dict] = []
    if manifest_path.exists():
        try:
            prior_variants = json.loads(manifest_path.read_text()).get("variants", [])
        except json.JSONDecodeError:
            logger.warning(f"{manifest_path} is unreadable — starting a fresh manifest")

    just_run = {s["variant"] for s in per_variant_stats}
    merged = [s for s in prior_variants if s.get("variant") not in just_run] + per_variant_stats
    merged.sort(key=lambda s: list(VARIANTS).index(s["variant"])
                if s.get("variant") in VARIANTS else len(VARIANTS))

    manifest_out = {
        "device": str(device),
        "batch_size": args.batch_size,
        "n_per_pool": args.n_per_pool,
        "neg_ratio": args.neg_ratio,
        "seed": args.seed,
        "n_samples_in_manifest": int(len(manifest)),
        "variants": merged,
    }
    manifest_path.write_text(json.dumps(manifest_out, indent=2))
    logger.info(f"wrote {args.outdir / 'extraction_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
