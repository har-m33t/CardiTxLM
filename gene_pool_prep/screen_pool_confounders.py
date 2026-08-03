"""
screen_pool_confounders.py — confounder screen on the curated pool's arms.

Why this exists
---------------
Track 4's in-degree centrality is a co-expression hub measure, and its own
README already flags that the top of the ranking is dominated by lncRNAs,
pseudogenes and unannotated ENSG identifiers whose mean expression sits
*below* the universe mean — "the familiar co-expression hub artifact —
sparsely detected features co-vary through shared detection/library-depth
patterns rather than biology."

That is a hypothesis about a technical confounder, and it is testable. If
in-degree is partly measuring library depth or batch structure rather than
biology, then the genes centrality selects *and variance/KEGG do not* should
track those technical variables more tightly than the rest of the pool does.
This screens for exactly that, using the confounders the whole-corpus EDA
stage already computes:

    library depth  — log10(library_size) from eda/steps/qc.py, per sample
    batch          — series_id from the extended-EDA label table (GEO series
                     is the standard batch unit for a corpus assembled from
                     many independent submissions)

Two effect sizes, per gene, across the 8,725 CVD-only samples
-------------------------------------------------------------
depth_r2   Squared Pearson correlation between the gene's log2 expression
           and log10 library size. Squared so it is on the same 0..1
           variance-explained scale as the batch measure.

series_eta2  One-way ANOVA effect size — the fraction of a gene's variance
           explained by which GEO series the sample came from.
           SS_between / SS_total. This is biased upward by the number of
           groups, which is why it is only ever read as a *contrast between
           arms* on the same sample set and never against an absolute
           threshold. A matched random-universe baseline is computed for the
           same reason.

Arms compared
-------------
centrality_only, variance_only, kegg, multi_source (>=2 categories), and a
random draw from the QC universe as the null. The comparison that matters is
centrality_only vs variance_only: same sample set, same confounders, same
estimator, differing only in which track selected the gene.

Column selection
----------------
The pool is symbol-keyed but the matrix is column-keyed, and 1,534 symbols
map to several columns. Each gene is screened on the column that actually
earned it its place in the arm (best variance rank for variance-selected,
best centrality rank for centrality-selected) — the point is to test the
selected column, not an arbitrary sibling.

Reads the same CVD-only matrix as Tracks 3 and 4. Recomputes nothing
upstream and writes no change to the pool; this is a read-only diagnostic.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GENE_BLOCK = 4096  # columns held in memory at once
RANDOM_SEED = 0
N_RANDOM_BASELINE = 4000


def load_confounders(
    matrix_dir: Path, labels_path: Path, qc_path: Path, min_library_size: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample log10 library size and series_id, ordered like the matrix.

    Returns the keep-mask alongside them. `min_library_size` drops effectively
    empty samples, which matter more here than anywhere else in the project:
    the CVD matrix retains 172 samples under 1e5 counts and 23 under 100
    counts (the smallest has a library size of 2). In such a sample every
    gene reads ~0, so every well-expressed gene picks up a large spurious
    correlation with depth. Leaving them in makes the depth screen mostly a
    detector of "is this gene normally expressed", which is not the question.
    """
    sample_index = np.load(matrix_dir / "cvd_only_sample_index.npy")

    labels = pd.read_parquet(labels_path, columns=["sample_index", "geo_accession", "series_id"])
    labels = labels.set_index("sample_index").loc[sample_index]

    qc = pd.read_csv(qc_path, usecols=["geo_accession", "library_size"])
    qc = qc.set_index("geo_accession")

    missing = labels["geo_accession"].isin(qc.index)
    if not missing.all():
        raise RuntimeError(
            f"{(~missing).sum()} CVD samples have no QC row — cannot screen library depth"
        )

    lib = qc.loc[labels["geo_accession"], "library_size"].to_numpy(dtype=np.float64)
    keep = lib >= min_library_size
    log_lib = np.log10(np.maximum(lib, 1.0))
    series = labels["series_id"].to_numpy()

    logger.info(
        "confounders: %d of %d samples kept (min library size %.0f), "
        "%d distinct series, log10 lib size %.2f-%.2f",
        int(keep.sum()),
        len(keep),
        min_library_size,
        len(np.unique(series[keep])),
        log_lib[keep].min(),
        log_lib[keep].max(),
    )
    return log_lib[keep], series[keep], keep


def screen(
    matrix_path: Path,
    log_lib: np.ndarray,
    series: np.ndarray,
    columns: np.ndarray,
    keep: np.ndarray | None = None,
) -> pd.DataFrame:
    """depth_r2 and series_eta2 for the requested matrix columns."""
    matrix = np.load(matrix_path, mmap_mode="r")
    if keep is None:
        keep = np.ones(matrix.shape[0], dtype=bool)
    n_samples = int(keep.sum())
    if len(log_lib) != n_samples:
        raise RuntimeError(f"confounder length {len(log_lib)} != kept samples {n_samples}")

    # Centred depth vector, reused for every gene block.
    depth = log_lib - log_lib.mean()
    depth_ss = float(depth @ depth)

    # One-hot series indicator for the ANOVA between-group sums.
    codes, counts = pd.factorize(series)[0], None
    n_groups = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    onehot = np.zeros((n_samples, n_groups), dtype=np.float32)
    onehot[np.arange(n_samples), codes] = 1.0

    columns = np.asarray(columns)
    depth_r2 = np.empty(len(columns), dtype=np.float64)
    series_eta2 = np.empty(len(columns), dtype=np.float64)
    # Carried so the arm contrast can be read within matched expression
    # strata — a sparsely-detected gene correlates weakly with everything,
    # so a raw arm-vs-arm depth_r2 gap partly just reflects detection rate.
    detections = np.empty(len(columns), dtype=np.float64)
    means = np.empty(len(columns), dtype=np.float64)

    for start in range(0, len(columns), GENE_BLOCK):
        block_cols = columns[start : start + GENE_BLOCK]
        # Fancy-indexing a memmap materialises only the requested columns.
        block = np.asarray(matrix[:, block_cols], dtype=np.float64)[keep]

        centred = block - block.mean(axis=0, keepdims=True)
        detection = (block > 0).mean(axis=0)
        mean_expr = block.mean(axis=0)
        detections[start : start + len(block_cols)] = detection
        means[start : start + len(block_cols)] = mean_expr
        total_ss = np.einsum("ij,ij->j", centred, centred)
        safe_total = np.where(total_ss > 0, total_ss, np.nan)

        # Depth: r^2 = (x.y)^2 / (SS_x * SS_y)
        cov = depth @ centred
        depth_r2[start : start + len(block_cols)] = (cov**2) / (depth_ss * safe_total)

        # Series: SS_between = sum_g n_g * (mean_g - grand_mean)^2. `centred`
        # is already grand-mean-centred, so group means are the deviations.
        group_means = (onehot.T.astype(np.float64) @ centred) / counts[:, None]
        between_ss = (counts[:, None] * group_means**2).sum(axis=0)
        series_eta2[start : start + len(block_cols)] = between_ss / safe_total

    return pd.DataFrame(
        {
            "column": columns,
            "depth_r2": depth_r2,
            "series_eta2": series_eta2,
            "detection": detections,
            "mean_expr": means,
        }
    )


def build_arms(pool: pd.DataFrame, variance: pd.DataFrame, centrality: pd.DataFrame) -> dict:
    """Map each arm's genes to the matrix column that earned their selection."""
    # The ranking CSVs are sorted by rank, so the matrix column must come from
    # the gene's position in the *universe* order, which is the order symbols
    # appear in gene_symbols.npy. Both rankings carry every column exactly
    # once, so we recover column ids by matching against that order.
    variance_sel = (
        variance.loc[variance["selected"]]
        .sort_values("variance_rank", kind="mergesort")
        .drop_duplicates("gene")
        .set_index("gene")["column"]
    )
    centrality_sel = (
        centrality.loc[centrality["selected"]]
        .sort_values("centrality_rank", kind="mergesort")
        .drop_duplicates("gene")
        .set_index("gene")["column"]
    )
    any_column = variance.drop_duplicates("gene").set_index("gene")["column"]

    v_only = pool.loc[
        pool["in_variance_set"] & ~pool["in_centrality_set"] & ~pool["in_kegg_set"], "gene"
    ]
    c_only = pool.loc[
        ~pool["in_variance_set"] & pool["in_centrality_set"] & ~pool["in_kegg_set"], "gene"
    ]
    kegg = pool.loc[pool["in_kegg_set"], "gene"]
    multi = pool.loc[pool["n_sources"] >= 2, "gene"]

    return {
        "centrality_only": centrality_sel.loc[c_only].to_numpy(),
        "variance_only": variance_sel.loc[v_only].to_numpy(),
        "kegg": any_column.loc[kegg].to_numpy(),
        "multi_source": any_column.loc[multi].to_numpy(),
    }


def summarize_arm(name: str, stats: pd.DataFrame) -> dict:
    return {
        "arm": name,
        "n_genes": len(stats),
        "depth_r2_median": float(stats["depth_r2"].median()),
        "depth_r2_mean": float(stats["depth_r2"].mean()),
        "depth_r2_p90": float(stats["depth_r2"].quantile(0.90)),
        "frac_depth_r2_gt_0.10": float((stats["depth_r2"] > 0.10).mean()),
        "frac_depth_r2_gt_0.25": float((stats["depth_r2"] > 0.25).mean()),
        "series_eta2_median": float(stats["series_eta2"].median()),
        "series_eta2_mean": float(stats["series_eta2"].mean()),
        "series_eta2_p90": float(stats["series_eta2"].quantile(0.90)),
        "frac_series_eta2_gt_0.50": float((stats["series_eta2"] > 0.50).mean()),
    }


def matched_contrast(
    per_gene: pd.DataFrame, focus: str = "centrality_only", reference: str = "random_universe"
) -> pd.DataFrame:
    """Contrast two arms within matched detection-rate strata.

    The raw arm comparison is confounded by expression level: the centrality
    arm is detected in ~66% of samples against ~87% for the variance arm, and
    a sparsely-detected gene correlates weakly with *everything*, confounder
    or not. Comparing only genes with similar detection rates removes that.
    """
    subset = per_gene[per_gene["arm"].isin([focus, reference])].copy()
    edges = np.arange(0.0, 1.01, 0.2)
    subset["stratum"] = pd.cut(subset["detection"], bins=edges, include_lowest=True)

    rows = []
    for stratum, group in subset.groupby("stratum", observed=True):
        focus_group = group[group["arm"] == focus]
        reference_group = group[group["arm"] == reference]
        if len(focus_group) < 30 or len(reference_group) < 30:
            continue
        rows.append(
            {
                "detection_stratum": str(stratum),
                f"n_{focus}": len(focus_group),
                f"n_{reference}": len(reference_group),
                f"depth_r2_{focus}": focus_group["depth_r2"].median(),
                f"depth_r2_{reference}": reference_group["depth_r2"].median(),
                f"series_eta2_{focus}": focus_group["series_eta2"].median(),
                f"series_eta2_{reference}": reference_group["series_eta2"].median(),
            }
        )
    return pd.DataFrame(rows)


def run(
    matrix_dir: Path,
    labels_path: Path,
    qc_path: Path,
    pool_path: Path,
    indir: Path,
    outdir: Path,
    min_library_size: float = 0.0,
) -> pd.DataFrame:
    pool = pd.read_csv(pool_path)
    variance = pd.read_csv(indir / "high_variance_genes.csv")
    centrality = pd.read_csv(indir / "high_centrality_genes.csv")

    # Recover each ranking row's matrix column from the universe symbol order.
    symbols = np.load(
        matrix_dir.parent / "elasticnet_out" / "expression" / "gene_symbols.npy", allow_pickle=True
    ).astype(str)
    order = pd.DataFrame({"gene": symbols, "column": np.arange(len(symbols))})
    variance = _attach_columns(variance, order, "variance_rank")
    centrality = _attach_columns(centrality, order, "centrality_rank")

    log_lib, series, keep = load_confounders(
        matrix_dir, labels_path, qc_path, min_library_size=min_library_size
    )
    arms = build_arms(pool, variance, centrality)

    rng = np.random.default_rng(RANDOM_SEED)
    arms["random_universe"] = rng.choice(
        len(symbols), size=min(N_RANDOM_BASELINE, len(symbols)), replace=False
    )

    matrix_path = matrix_dir / "cvd_only_expression.npy"
    rows, per_gene = [], []
    for name, columns in arms.items():
        logger.info("screening %s (%d genes)", name, len(columns))
        stats = screen(matrix_path, log_lib, series, columns, keep=keep)
        stats["arm"] = name
        stats["gene"] = symbols[stats["column"].to_numpy()]
        per_gene.append(stats)
        rows.append(summarize_arm(name, stats))

    summary = pd.DataFrame(rows)
    all_genes = pd.concat(per_gene, ignore_index=True)
    matched = pd.concat(
        [
            matched_contrast(all_genes, "centrality_only", "random_universe"),
            matched_contrast(all_genes, "centrality_only", "variance_only"),
        ],
        keys=["vs_random_universe", "vs_variance_only"],
        names=["reference"],
    ).reset_index(level=0)

    outdir.mkdir(parents=True, exist_ok=True)
    all_genes.to_csv(outdir / "confounder_screen_per_gene.csv", index=False)
    summary.to_csv(outdir / "confounder_screen_summary.csv", index=False)
    matched.to_csv(outdir / "confounder_screen_matched.csv", index=False)
    (outdir / "confounder_screen_manifest.json").write_text(
        json.dumps(
            {
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_samples": int(len(log_lib)),
                "n_samples_dropped_low_depth": int((~keep).sum()),
                "min_library_size": min_library_size,
                "n_series": int(len(np.unique(series))),
                "confounders": {
                    "depth": "log10(library_size) from eda/steps/qc.py",
                    "batch": "series_id from extended_eda label table",
                },
                "note": (
                    "series_eta2 is biased upward by group count and is only "
                    "read as a contrast between arms on a shared sample set, "
                    "never against an absolute threshold."
                ),
                "arms": rows,
            },
            indent=2,
        )
    )
    return summary


def _attach_columns(ranking: pd.DataFrame, order: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    """Give every ranking row its matrix column id.

    Both the ranking and the universe list each column exactly once, so
    sorting each by (gene, its own tiebreak) and pairing them off would be
    fragile. Instead we pair by gene with a stable within-gene ordering on
    both sides — for duplicated symbols any assignment within the group is
    equivalent, since duplicate columns are near-identical by construction.
    """
    ranking = ranking.sort_values(rank_col, kind="mergesort").reset_index(drop=True)
    ranking["_k"] = ranking.groupby("gene").cumcount()
    order = order.copy()
    order["_k"] = order.groupby("gene").cumcount()
    merged = ranking.merge(order, on=["gene", "_k"], how="left", validate="one_to_one")
    if merged["column"].isna().any():
        raise RuntimeError("could not map every ranking row to a matrix column")
    return merged.drop(columns="_k")


def main() -> None:
    parser = argparse.ArgumentParser(description="Confounder screen on curated pool arms")
    root = Path("eda/dataset/cvd_data")
    parser.add_argument("--matrix-dir", type=Path, default=root / "gene_pool_prep")
    parser.add_argument(
        "--labels", type=Path, default=root / "extended_eda_out/labels/sample_labels.parquet"
    )
    parser.add_argument("--qc", type=Path, default=root / "eda_out/qc/qc_full_dataset.csv")
    parser.add_argument("--pool", type=Path, default=Path("curated_gene_pool.csv"))
    parser.add_argument("--indir", type=Path, default=Path("gene_pool_prep"))
    parser.add_argument("--outdir", type=Path, default=Path("gene_pool_prep/confounder_screen"))
    parser.add_argument(
        "--min-library-size",
        type=float,
        default=1e5,
        help="drop samples below this raw library size before screening (default 1e5)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = run(
        args.matrix_dir,
        args.labels,
        args.qc,
        args.pool,
        args.indir,
        args.outdir,
        min_library_size=args.min_library_size,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
