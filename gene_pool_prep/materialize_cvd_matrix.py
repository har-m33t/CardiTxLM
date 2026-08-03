"""
materialize_cvd_matrix.py — Track 1: materialise a clean CVD-only matrix.

Why this exists (and why it is *not* the elastic net's X.npy)
------------------------------------------------------------
`elasticnet_out/expression/X.npy` is a CVD-vs-random-negative *training*
pool: every CVD positive plus a 3:1 random subsample of non-CVD negatives.
Ranking genes on that population answers "which genes separate CVD tissue
from random other tissue" — which is the elastic net's job, and already
done. Gene-pool curation needs the other question: "which genes vary, and
which sit at the centre of co-expression structure, *within* CVD". That
requires a within-CVD population with no negatives in it at all, which is
what this module builds.

Pool definition (caller must choose explicitly — no default)
------------------------------------------------------------
`disease_confirmed`  → `is_cvd_disease` (keyword-confirmed disease samples)
`union_pool`         → `is_cvd_pool` (disease-confirmed ∪ CVD-tissue)

The standing rule from the extended-EDA fix is that the tissue-only-
unconfirmed bucket must never be treated as disease-positive, which argues
for `disease_confirmed`. `union_pool` is offered only so the choice is
recorded in the manifest rather than hard-coded silently.

Single-cell filter
------------------
`subsample.py` excludes samples with ARCHS4's `singlecellprobability`
> 0.5 from *both* classes before building any pool, "matching the same
'exclude sc, subsample from bulk pool' convention used everywhere else in
the EDA". This module keeps that convention on by default: single-cell
libraries have radically different dropout and count distributions, which
would distort both the per-gene variance ranking (Track 3) and the
co-expression correlations (Track 4). Disable with --no-singlecell-filter
only deliberately.

Library-size filter
-------------------
Samples below 1e5 raw counts are dropped (--min-library-size, 0 disables).
These are failed runs — 172 in the disease-confirmed pool, median 75 genes
detected against a cohort median of 30,469, the smallest with a library size
of 2. Applied here rather than in each consumer so Tracks 3, 4 and QA
generation all inherit one decision. Added after Track 5's confounder screen
surfaced them; see apply_library_size_filter.

Reused as-is, never recomputed
------------------------------
`kept_gene_mask.npy` / `gene_symbols.npy` from the elastic net run — the
49,231-gene QC universe (detected at count > 0 in >= 10% of the training
pool). A gene undetected essentially everywhere is a bad candidate under
any method, so the mask transfers even though it was computed against a
different sample population. This module validates the mask against the
H5 gene axis but does not modify or recompute it.

Outputs
-------
cvd_only_expression.npy       float32 (n_pool_samples, 49231), log2(count+1)
cvd_only_sample_index.npy     int64   (n_pool_samples,) — H5 sample indices,
                              row-aligned to the matrix above
cvd_only_matrix_manifest.json provenance + sanity-check results
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eda.dataset import io as archs4_io

logger = logging.getLogger(__name__)


# Must match load_expression.py — the whole point is scale consistency with
# the prior stages of this project.
LOG2_PSEUDOCOUNT = 1.0
DEFAULT_CHUNK_SIZE = 2048
# Raw-count floor below which a library is a failed run, not a shallow one.
# See apply_library_size_filter for the evidence behind this value.
DEFAULT_MIN_LIBRARY_SIZE = 1e5

# Column in sample_labels.parquet backing each pool definition. Verified
# against the parquet schema: is_cvd_pool == is_cvd_disease | is_cvd_tissue.
POOL_DEFINITIONS = {
    "disease_confirmed": "is_cvd_disease",
    "union_pool": "is_cvd_pool",
}

# The elastic net training pool size. A disease-confirmed CVD-only pool must
# come out below this; if it doesn't, the filter selected the wrong column.
ELASTICNET_POOL_N_SAMPLES = 34900


class SanityCheckError(RuntimeError):
    """Raised when a post-materialisation check fails — nothing is saved."""


def select_pool_indices(
    labels_parquet: Path, pool_definition: str
) -> tuple[np.ndarray, dict]:
    """Return the H5 sample indices for `pool_definition`, plus label stats.

    Indices are returned sorted ascending, so matrix row order is a stable
    function of the pool definition alone and reruns are byte-identical.
    """
    if pool_definition not in POOL_DEFINITIONS:
        raise ValueError(
            f"unknown pool definition {pool_definition!r}; "
            f"expected one of {sorted(POOL_DEFINITIONS)}"
        )
    column = POOL_DEFINITIONS[pool_definition]

    df = pd.read_parquet(
        labels_parquet,
        columns=["sample_index", "is_cvd_disease", "is_cvd_tissue", "is_cvd_pool"],
    )
    if column not in df.columns:
        raise KeyError(
            f"column {column!r} not found in {labels_parquet}; "
            f"available: {sorted(df.columns)}"
        )

    pool_idx = np.sort(df.loc[df[column].to_numpy(), "sample_index"].to_numpy())
    stats = {
        "labels_total_rows": int(len(df)),
        "n_is_cvd_disease": int(df["is_cvd_disease"].sum()),
        "n_is_cvd_tissue": int(df["is_cvd_tissue"].sum()),
        "n_is_cvd_pool": int(df["is_cvd_pool"].sum()),
        "n_tissue_only_unconfirmed": int(
            (df["is_cvd_tissue"] & ~df["is_cvd_disease"]).sum()
        ),
        "n_selected_before_singlecell_filter": int(pool_idx.size),
    }
    logger.info(
        "pool %r via column %r: %d samples selected (of %d corpus rows)",
        pool_definition, column, pool_idx.size, len(df),
    )
    return pool_idx.astype(np.int64), stats


def apply_singlecell_filter(
    h5, pool_idx: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Drop likely single-cell libraries, per the EDA-wide convention."""
    bulk_pool, filter_stats = archs4_io.filter_bulk_indices(h5)
    in_bulk = np.zeros(archs4_io.get_shape(h5).n_samples, dtype=bool)
    in_bulk[bulk_pool] = True
    kept = pool_idx[in_bulk[pool_idx]]
    logger.info(
        "single-cell filter: kept %d / %d pool samples (dropped %d)",
        kept.size, pool_idx.size, pool_idx.size - kept.size,
    )
    return kept, filter_stats


def apply_library_size_filter(
    h5, pool_idx: np.ndarray, min_library_size: float
) -> tuple[np.ndarray, dict]:
    """Drop effectively-empty libraries before they reach any downstream stage.

    Found by Track 5's confounder screen: 172 CVD-pool samples carry fewer
    than 1e5 total counts, 23 fewer than 100, and the smallest has a library
    size of **2**. Their median genes-detected is 75 against a cohort median
    of 30,469 — these are failed runs, not low-depth-but-usable samples.

    `eda/steps/qc.py` already flags 143 of them as `outlier_lib_size_lo` but
    is deliberately non-destructive (pre-CVD-selection EDA must not drop
    samples), so nothing had acted on the flag. Filtering here rather than in
    each consumer means variance (Track 3), centrality (Track 4) and QA
    generation all inherit one decision instead of re-deriving it.

    Note this removes essentially all of GSE53080/GSE53081 (115 of the 172),
    so it is a small cohort-composition change, not only noise removal.

    Library size is the sum of raw counts, matching qc.py's definition, and
    is computed here from the H5 rather than joined from the QC CSV so this
    module keeps its single-source-of-truth property.
    """
    # Read only the pool's samples — a full-corpus scan here would re-do
    # qc.py's pass over all 1.1M samples to answer a question about 8,725.
    pool_lib = np.empty(pool_idx.shape[0], dtype=np.int64)
    for start in range(0, pool_idx.shape[0], DEFAULT_CHUNK_SIZE):
        stop = min(start + DEFAULT_CHUNK_SIZE, pool_idx.shape[0])
        chunk = archs4_io.read_samples_by_index(h5, pool_idx[start:stop])
        pool_lib[start:stop] = chunk.sum(axis=0)

    kept = pool_idx[pool_lib >= min_library_size]
    dropped = pool_idx.size - kept.size
    stats = {
        "min_library_size": float(min_library_size),
        "n_before": int(pool_idx.size),
        "n_kept": int(kept.size),
        "n_dropped": int(dropped),
        "dropped_pct": round(100.0 * dropped / max(pool_idx.size, 1), 4),
        "dropped_library_size_min": int(pool_lib.min()) if dropped else None,
        "dropped_library_size_max": (
            int(pool_lib[pool_lib < min_library_size].max()) if dropped else None
        ),
    }
    logger.info(
        "library-size filter (>= %.0f): kept %d / %d pool samples (dropped %d)",
        min_library_size, kept.size, pool_idx.size, dropped,
    )
    return kept, stats


def load_gene_universe(expression_dir: Path, h5) -> tuple[np.ndarray, np.ndarray]:
    """Load the QC gene mask + symbols and validate them against the H5.

    Reused verbatim — this function deliberately has no path that would
    recompute a detection-rate filter.
    """
    keep_mask = np.load(expression_dir / "kept_gene_mask.npy")
    symbols = np.load(expression_dir / "gene_symbols.npy", allow_pickle=True)

    n_genes_h5 = archs4_io.get_shape(h5).n_genes
    if keep_mask.shape != (n_genes_h5,):
        raise SanityCheckError(
            f"kept_gene_mask.npy has shape {keep_mask.shape} but the H5 gene "
            f"axis is {n_genes_h5} — mask was built against a different H5."
        )
    if int(keep_mask.sum()) != symbols.shape[0]:
        raise SanityCheckError(
            f"mask keeps {int(keep_mask.sum())} genes but gene_symbols.npy has "
            f"{symbols.shape[0]} entries — the two files disagree."
        )
    logger.info("QC gene universe: %d kept of %d H5 genes (reused as-is)",
                int(keep_mask.sum()), n_genes_h5)
    return keep_mask, symbols


def _streaming_load_and_log2(
    h5, pool_indices: np.ndarray, keep_mask: np.ndarray, chunk_size: int
) -> np.ndarray:
    """Build the (n_pool, n_kept_genes) log2 matrix, streaming over samples.

    Same two-step-per-chunk shape as load_expression.py: read raw counts for
    a chunk of samples, mask to kept genes, log2 in place, write transposed
    straight into the preallocated output.
    """
    n_pool = int(pool_indices.shape[0])
    n_kept = int(keep_mask.sum())
    x = np.empty((n_pool, n_kept), dtype=np.float32)
    for start in range(0, n_pool, chunk_size):
        stop = min(start + chunk_size, n_pool)
        chunk_counts = archs4_io.read_samples_by_index(h5, pool_indices[start:stop])
        chunk_kept = chunk_counts[keep_mask, :].astype(np.float32, copy=False)
        chunk_kept += LOG2_PSEUDOCOUNT
        np.log2(chunk_kept, out=chunk_kept)
        x[start:stop, :] = chunk_kept.T
        logger.info("log2 load: %d / %d pool samples written", stop, n_pool)
    return x


def run_sanity_checks(
    x: np.ndarray, pool_idx: np.ndarray, n_expected_genes: int, pool_definition: str
) -> dict:
    """Validate the matrix before anything is written. Raises on failure."""
    checks: dict = {}
    failures: list[str] = []

    checks["shape"] = list(x.shape)
    checks["shape_matches_expected"] = x.shape == (pool_idx.size, n_expected_genes)
    if not checks["shape_matches_expected"]:
        failures.append(
            f"shape {x.shape} != expected ({pool_idx.size}, {n_expected_genes})"
        )

    n_nan = int(np.isnan(x).sum())
    n_inf = int(np.isinf(x).sum())
    checks["n_nan"] = n_nan
    checks["n_inf"] = n_inf
    checks["no_nan_or_inf"] = (n_nan == 0 and n_inf == 0)
    if not checks["no_nan_or_inf"]:
        failures.append(f"found {n_nan} NaN and {n_inf} inf values post-transform")

    # log2(count + 1) on non-negative integer counts is >= 0 by construction;
    # a negative would mean we read something that isn't a raw count matrix.
    min_val = float(x.min())
    checks["min_value"] = min_val
    checks["min_value_non_negative"] = min_val >= 0.0
    if not checks["min_value_non_negative"]:
        failures.append(f"min value {min_val} < 0 — source is not raw counts")

    checks["max_value"] = float(x.max())
    checks["elasticnet_pool_n_samples"] = ELASTICNET_POOL_N_SAMPLES
    checks["smaller_than_elasticnet_pool"] = int(x.shape[0]) < ELASTICNET_POOL_N_SAMPLES
    if not checks["smaller_than_elasticnet_pool"]:
        # Only a hard failure for the disease-confirmed pool, which is a strict
        # subset of the elastic net's positives-plus-negatives population.
        msg = (
            f"sample count {x.shape[0]} >= elastic net pool "
            f"{ELASTICNET_POOL_N_SAMPLES}"
        )
        if pool_definition == "disease_confirmed":
            failures.append(msg)
        else:
            logger.warning("%s — expected for pool_definition=%r", msg, pool_definition)

    checks["passed"] = not failures
    checks["failures"] = failures
    if failures:
        raise SanityCheckError(
            "sanity checks failed, nothing saved:\n  - " + "\n  - ".join(failures)
        )
    return checks


def _h5_provenance(h5_path: Path, h5) -> dict:
    stat = h5_path.stat()
    shape = archs4_io.get_shape(h5)
    return {
        "path": str(h5_path),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "n_genes": int(shape.n_genes),
        "n_samples": int(shape.n_samples),
        "expression_dataset": archs4_io.EXPRESSION_PATH,
    }


def run(
    h5_path: Path,
    labels_parquet: Path,
    expression_dir: Path,
    outdir: Path,
    pool_definition: str,
    singlecell_filter: bool = True,
    min_library_size: float = DEFAULT_MIN_LIBRARY_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    pool_idx, label_stats = select_pool_indices(labels_parquet, pool_definition)
    if pool_idx.size == 0:
        raise SanityCheckError(
            f"pool definition {pool_definition!r} selected 0 samples — refusing "
            "to materialise an empty matrix."
        )

    with archs4_io.open_h5(h5_path) as h5:
        h5_info = _h5_provenance(h5_path, h5)
        sc_stats = None
        if singlecell_filter:
            pool_idx, sc_stats = apply_singlecell_filter(h5, pool_idx)
            if pool_idx.size == 0:
                raise SanityCheckError(
                    "single-cell filter removed every pool sample — refusing to "
                    "materialise an empty matrix."
                )
        lib_stats = None
        if min_library_size > 0:
            pool_idx, lib_stats = apply_library_size_filter(h5, pool_idx, min_library_size)
            if pool_idx.size == 0:
                raise SanityCheckError(
                    "library-size filter removed every pool sample — refusing to "
                    "materialise an empty matrix."
                )
        keep_mask, symbols = load_gene_universe(expression_dir, h5)
        x = _streaming_load_and_log2(h5, pool_idx, keep_mask, chunk_size)

    logger.info("materialised CVD-only matrix: shape %s (%.2f GB)",
                x.shape, x.nbytes / 1e9)

    checks = run_sanity_checks(x, pool_idx, symbols.shape[0], pool_definition)

    np.save(outdir / "cvd_only_expression.npy", x)
    np.save(outdir / "cvd_only_sample_index.npy", pool_idx)

    manifest = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "pool_definition": pool_definition,
        "pool_definition_column": POOL_DEFINITIONS[pool_definition],
        "pool_definition_note": (
            "disease_confirmed = is_cvd_disease only; the tissue-only-"
            "unconfirmed bucket is excluded per the standing extended-EDA rule "
            "that it must never be treated as disease-positive."
        ),
        "n_samples": int(x.shape[0]),
        "n_genes": int(x.shape[1]),
        "shape": list(x.shape),
        "singlecell_filter_applied": bool(singlecell_filter),
        "singlecell_filter_stats": sc_stats,
        "library_size_filter_applied": bool(min_library_size > 0),
        "library_size_filter_stats": lib_stats,
        "label_stats": label_stats,
        "log2_pseudocount": LOG2_PSEUDOCOUNT,
        "transform": "log2(raw_count + 1.0), matching load_expression.py",
        "chunk_size": int(chunk_size),
        "sources": {
            "labels_parquet": str(labels_parquet),
            "archs4_h5": h5_info,
            "gene_universe_dir": str(expression_dir),
            "kept_gene_mask": str(expression_dir / "kept_gene_mask.npy"),
            "gene_symbols": str(expression_dir / "gene_symbols.npy"),
        },
        "gene_universe_note": (
            "QC gene mask reused verbatim from the elastic net run "
            "(>= 10% detection at count > 0); not recomputed here."
        ),
        "sanity_checks": checks,
        "outputs": {
            "expression": "cvd_only_expression.npy",
            "sample_index": "cvd_only_sample_index.npy",
        },
        "not_derived_from": (
            "elasticnet_out/expression/X.npy — wrong population (CVD-vs-random-"
            "negative training pool); this matrix is built fresh from the H5."
        ),
    }
    with open(outdir / "cvd_only_matrix_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("wrote %s", outdir / "cvd_only_expression.npy")
    return outdir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--archs4-h5-path", required=True, type=Path)
    p.add_argument(
        "--labels-parquet", required=True, type=Path,
        help="extended_eda_out/labels/sample_labels.parquet",
    )
    p.add_argument(
        "--expression-dir", required=True, type=Path,
        help="elasticnet_out/expression — source of the QC gene mask/symbols.",
    )
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument(
        "--pool-definition", required=True, choices=sorted(POOL_DEFINITIONS),
        help="No default: this is a recorded scientific decision, not a flag.",
    )
    p.add_argument(
        "--min-library-size", type=float, default=DEFAULT_MIN_LIBRARY_SIZE,
        help="drop pool samples below this raw library size (0 disables); "
             "default 1e5 removes failed runs with ~75 genes detected.",
    )
    p.add_argument(
        "--no-singlecell-filter", dest="singlecell_filter",
        action="store_false", default=True,
        help="Keep likely single-cell libraries (off by default, matching the EDA).",
    )
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(
        h5_path=args.archs4_h5_path,
        labels_parquet=args.labels_parquet,
        expression_dir=args.expression_dir,
        outdir=args.outdir,
        pool_definition=args.pool_definition,
        singlecell_filter=args.singlecell_filter,
        min_library_size=args.min_library_size,
        chunk_size=args.chunk_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
