"""
compute_centrality_genes.py — Track 4: co-expression in-degree centrality.

What "centrality" means here
----------------------------
For each gene we take its top-100 most-correlated partners and draw a
directed edge A -> B for each partner B. The centrality score is
**in-degree**: how many *other* genes name this gene among their top 100.

Out-degree is deliberately not computed or reported. By construction every
gene names exactly 100 partners, so out-degree is the constant 100 for all
49,231 genes and carries zero discriminating signal. A real hub is a gene
that many others point *at*, even though its own outgoing list is capped.

Computational approach
----------------------
A dense 49,231 x 49,231 float32 correlation matrix is ~9.7 GB, on a machine
with 25.7 GB total and meaningful existing pressure. It is also unnecessary:
only the top 100 entries of each row are ever used.

So the correlation is computed in row blocks. Each gene column is first
z-scored across samples and scaled by 1/sqrt(n-1), after which the Pearson
correlation of a block against all genes is exactly `Zt[block] @ Z` — one
BLAS sgemm per block. Peak extra memory is one block of the correlation
matrix (2,048 x 49,231 float32 = ~403 MB) rather than the full 9.7 GB.

Population
----------
Track 1's `cvd_only_expression.npy` (8,725 disease-confirmed bulk CVD
samples) only. Never `elasticnet_out/expression/X.npy`, which is the
CVD-vs-random-negative training pool — the wrong population for asking
which genes are central *within* cardiovascular disease.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DATA_DIR = REPO / "eda" / "dataset" / "cvd_data" / "gene_pool_prep"
EXPRESSION_DIR = REPO / "eda" / "dataset" / "cvd_data" / "elasticnet_out" / "expression"

MATRIX = DATA_DIR / "cvd_only_expression.npy"
MATRIX_MANIFEST = DATA_DIR / "cvd_only_matrix_manifest.json"
VARIANCE_MANIFEST = HERE / "variance_manifest.json"

TOP_K = 100  # ARCHS4's documented top-100 co-expression convention
CARDIAC_SPOT_CHECK = ("TTN", "MYH7", "ACTC1", "MYBPC3", "NPPA", "ACTN2")


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def confirm_inputs() -> tuple[dict, float, int]:
    """Verify Track 1's outputs exist and Track 3's threshold is available."""
    if not MATRIX.exists() or not MATRIX_MANIFEST.exists():
        raise SystemExit(
            "STOP: Track 1 output missing.\n"
            f"  expected {MATRIX}\n"
            f"  expected {MATRIX_MANIFEST}\n"
            "Track 1 (matrix materialization) must complete first. Refusing to "
            "substitute elasticnet_out/expression/X.npy — wrong population."
        )

    manifest = json.loads(MATRIX_MANIFEST.read_text())

    if manifest.get("n_genes") != 49231:
        raise SystemExit(
            f"STOP: Track 1 manifest reports n_genes={manifest.get('n_genes')}, "
            "expected 49231. Population/gene-count mismatch — not proceeding."
        )
    if manifest.get("pool_definition") != "disease_confirmed":
        raise SystemExit(
            f"STOP: Track 1 manifest reports pool_definition="
            f"{manifest.get('pool_definition')!r}, expected 'disease_confirmed'. "
            "Population mismatch — not proceeding."
        )

    if not VARIANCE_MANIFEST.exists():
        raise SystemExit(
            "STOP: Track 3 has not completed — no variance_manifest.json, so the "
            "top-N% threshold is unconfirmed. Refusing to pick one silently."
        )
    var_manifest = json.loads(VARIANCE_MANIFEST.read_text())
    top_percent = float(var_manifest["top_percent"])
    n_selected_track3 = int(var_manifest["n_selected"])
    log(
        f"threshold confirmed from Track 3: top {top_percent}% "
        f"({n_selected_track3} genes selected there)"
    )
    return manifest, top_percent, n_selected_track3


def standardize(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the matrix and z-score each gene, scaled so Z.T @ Z is Pearson r.

    Returns (Z, zero_variance_mask). Genes with zero variance would divide by
    zero and poison the whole correlation matrix with NaN, so they are given
    an all-zero column (correlation 0 with everything) and reported instead.
    """
    log(f"loading {path.name} ...")
    X = np.load(path, mmap_mode="r")
    n_samples, n_genes = X.shape
    log(f"  shape {X.shape}, computing per-gene mean/std in float64")

    Z = np.empty((n_samples, n_genes), dtype=np.float32)
    stds = np.empty(n_genes, dtype=np.float64)

    # ddof=1, matching Track 3's variance convention.
    denom = np.sqrt(n_samples - 1.0)
    block = 4096
    for start in range(0, n_genes, block):
        stop = min(start + block, n_genes)
        chunk = np.asarray(X[:, start:stop], dtype=np.float64)
        mu = chunk.mean(axis=0)
        centered = chunk - mu
        sd = np.sqrt((centered**2).sum(axis=0) / (n_samples - 1.0))
        stds[start:stop] = sd
        safe = np.where(sd > 0, sd, 1.0)
        Z[:, start:stop] = (centered / (safe * denom)).astype(np.float32)

    zero_var = stds <= 0
    if zero_var.any():
        Z[:, zero_var] = 0.0
        log(f"  WARNING: {int(zero_var.sum())} zero-variance genes zeroed out")
    else:
        log(f"  no zero-variance genes (min std {stds.min():.6g})")

    if not np.isfinite(Z).all():
        raise SystemExit("STOP: non-finite values in standardized matrix")
    log("  standardized matrix is finite")
    return Z, zero_var


def compute_in_degree(
    Z: np.ndarray, zero_var: np.ndarray, block: int, sym_codes: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Blocked correlation -> top-100 per gene -> in-degree accumulation."""
    n_genes = Z.shape[1]
    in_degree = np.zeros(n_genes, dtype=np.int64)

    n_blocks = (n_genes + block - 1) // block
    n_nonfinite = 0
    corr_min, corr_max = np.inf, -np.inf
    n_same_symbol_edges = 0

    for bi, start in enumerate(range(0, n_genes, block)):
        stop = min(start + block, n_genes)
        # Transposing one block at a time (B x samples, ~71 MB) rather than
        # materializing a full 1.72 GB contiguous Z.T up front.
        block_t = np.ascontiguousarray(Z[:, start:stop].T)
        C = block_t @ Z  # (B x n_genes) Pearson correlations

        bad = ~np.isfinite(C)
        if bad.any():
            n_nonfinite += int(bad.sum())
            C[bad] = -np.inf

        # Exclude self-correlation so a gene never names itself. This happens
        # BEFORE the bounds check: a gene's self-correlation is exactly 1.0,
        # which float32 renders as up to ~1.0002, and measuring that would
        # just be measuring float32 epsilon on a value we discard anyway.
        rows = np.arange(stop - start)
        C[rows, np.arange(start, stop)] = -np.inf

        finite = C[np.isfinite(C)]
        if finite.size:
            corr_min = min(corr_min, float(finite.min()))
            corr_max = max(corr_max, float(finite.max()))

        # Top-100 partners per row. argpartition is O(n) vs a full sort.
        top = np.argpartition(-C, TOP_K, axis=1)[:, :TOP_K]
        in_degree += np.bincount(top.ravel(), minlength=n_genes)

        # Diagnostic: edges landing on a column that shares this gene's symbol.
        # 1,534 symbols map to more than one column in the QC universe, and
        # some of those columns are byte-identical, so they are guaranteed to
        # be each other's top partner and inflate one another's in-degree.
        n_same_symbol_edges += int(
            (sym_codes[top] == sym_codes[start:stop][:, None]).sum()
        )

        if bi % 5 == 0 or stop == n_genes:
            log(f"  block {bi + 1}/{n_blocks} (genes {start}-{stop})")

    # Zero-variance genes correlate with nothing meaningfully; they should not
    # be credited with in-degree earned from an all-zero correlation row.
    if zero_var.any():
        in_degree[zero_var] = 0

    stats = {
        "n_nonfinite_correlations": n_nonfinite,
        "corr_min": corr_min,
        "corr_max": corr_max,
        "n_same_symbol_edges": n_same_symbol_edges,
    }
    return in_degree, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--outdir", type=Path, default=HERE)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    matrix_manifest, top_percent, n_selected_track3 = confirm_inputs()

    symbols = np.load(EXPRESSION_DIR / "gene_symbols.npy", allow_pickle=True)
    genes = np.array([str(s) for s in symbols])
    n_genes = genes.shape[0]
    if n_genes != matrix_manifest["n_genes"]:
        raise SystemExit(
            f"STOP: gene_symbols.npy has {n_genes} entries but Track 1's manifest "
            f"reports {matrix_manifest['n_genes']} — refusing to proceed."
        )

    Z, zero_var = standardize(MATRIX)
    if Z.shape[1] != n_genes:
        raise SystemExit(
            f"STOP: matrix has {Z.shape[1]} gene columns, gene list has {n_genes}"
        )

    # Integer code per column, so same-symbol columns share a code.
    uniq_syms, sym_codes = np.unique(genes, return_inverse=True)
    n_dup_symbols = int(len(genes) - len(uniq_syms))

    log(f"computing blocked correlations (block={args.block_size}, top-{TOP_K})")
    in_degree, corr_stats = compute_in_degree(
        Z, zero_var, args.block_size, sym_codes
    )
    del Z

    # Rank descending. Gene index is the tiebreaker so runs are reproducible.
    order = np.lexsort((np.arange(n_genes), -in_degree))
    rank = np.empty(n_genes, dtype=np.int64)
    rank[order] = np.arange(1, n_genes + 1)

    # Match Track 3's selected count exactly, for a consistent union in Track 5.
    n_select = n_selected_track3
    selected = rank <= n_select

    # ---- sanity checks -------------------------------------------------
    total_edges = int(in_degree.sum())
    expected_edges = n_genes * TOP_K - int(zero_var.sum()) * TOP_K
    near_100 = int(((in_degree >= 90) & (in_degree <= 110)).sum())

    checks = {
        "n_nonfinite_correlations": corr_stats["n_nonfinite_correlations"],
        "no_nonfinite_correlations": corr_stats["n_nonfinite_correlations"] == 0,
        "corr_min": corr_stats["corr_min"],
        "corr_max": corr_stats["corr_max"],
        # Off-diagonal only (self-correlation is masked before this is taken).
        # Tolerance is sized for float32 sgemm over 8,725 samples: measured
        # max deviation from a float64 reference is 5.9e-5, so 1e-3 is a
        # genuine bound rather than a rubber stamp.
        "corr_bounds_tolerance": 1e-3,
        "corr_within_bounds": corr_stats["corr_min"] >= -1.0 - 1e-3
        and corr_stats["corr_max"] <= 1.0 + 1e-3,
        "n_zero_variance_genes": int(zero_var.sum()),
        "in_degree_min": int(in_degree.min()),
        "in_degree_max": int(in_degree.max()),
        "in_degree_mean": float(in_degree.mean()),
        "in_degree_median": float(np.median(in_degree)),
        "in_degree_std": float(in_degree.std()),
        "total_edges": total_edges,
        "expected_edges": expected_edges,
        "edge_count_consistent": total_edges <= expected_edges,
        # If in-degree were accidentally out-degree, every gene would sit at
        # exactly 100 and this fraction would be ~1.0.
        "frac_in_degree_near_100": round(near_100 / n_genes, 4),
        "in_degree_varies": int(in_degree.max()) > int(in_degree.min()),
        "not_degenerate_out_degree": near_100 / n_genes < 0.5,
        "n_duplicate_symbol_columns": n_dup_symbols,
        "n_same_symbol_edges": corr_stats["n_same_symbol_edges"],
        "frac_same_symbol_edges": round(
            corr_stats["n_same_symbol_edges"] / max(total_edges, 1), 5
        ),
        "n_selected": int(selected.sum()),
        "n_selected_expected": n_select,
        "selected_matches_track3": int(selected.sum()) == n_selected_track3,
        "ranks_unique_and_complete": bool(
            np.array_equal(np.sort(rank), np.arange(1, n_genes + 1))
        ),
    }
    failures = [
        k
        for k in (
            "no_nonfinite_correlations",
            "corr_within_bounds",
            "edge_count_consistent",
            "in_degree_varies",
            "not_degenerate_out_degree",
            "selected_matches_track3",
            "ranks_unique_and_complete",
        )
        if not checks[k]
    ]
    checks["passed"] = not failures
    checks["failures"] = failures

    spot: dict[str, dict] = {}
    for name in CARDIAC_SPOT_CHECK:
        hits = np.flatnonzero(genes == name)
        if hits.size == 0:
            spot[name] = {"present": False}
            continue
        i = int(hits[0])
        spot[name] = {
            "present": True,
            "in_degree": int(in_degree[i]),
            "centrality_rank": int(rank[i]),
            "percentile": round(100.0 * rank[i] / n_genes, 3),
            "selected": bool(selected[i]),
        }

    # ---- outputs -------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.outdir / "high_centrality_genes.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gene", "in_degree", "centrality_rank", "selected"])
        for i in order:
            writer.writerow(
                [genes[i], int(in_degree[i]), int(rank[i]), bool(selected[i])]
            )

    finished = datetime.now(timezone.utc)
    manifest = {
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "elapsed_seconds": round((finished - started).total_seconds(), 1),
        "centrality_measure": "in_degree",
        "centrality_definition": (
            "Number of other genes that list this gene among their top-100 most "
            "correlated partners. Out-degree is NOT used: it is exactly 100 for "
            "every gene by construction and carries no discriminating signal."
        ),
        "top_k_partners": TOP_K,
        "top_percent": top_percent,
        "threshold_source": (
            "matched to Track 3's variance_manifest.json (top_percent and "
            "n_selected both reused verbatim, so the two rankings contribute "
            "equally sized sets to Track 5's union)"
        ),
        "n_genes_total": n_genes,
        "n_selected": int(selected.sum()),
        "correlation": {
            "method": "Pearson across samples, gene x gene",
            "approach": "blocked sgemm on z-scored columns",
            "block_size": args.block_size,
            "rationale": (
                "A dense 49,231^2 float32 correlation matrix is ~9.7 GB against "
                "25.7 GB of system RAM under existing pressure, and only the top "
                "100 entries per row are ever needed. Blocking holds one "
                f"{args.block_size} x {n_genes} float32 slab (~403 MB) at a time."
            ),
            "self_correlation": "excluded (diagonal set to -inf before top-k)",
            "tie_handling": (
                "argpartition; ties at the top-100 boundary resolve arbitrarily "
                "but deterministically for a given input"
            ),
        },
        "sources": {
            "matrix": str(MATRIX.relative_to(REPO)),
            "matrix_manifest": str(MATRIX_MANIFEST.relative_to(REPO)),
            "gene_symbols": str((EXPRESSION_DIR / "gene_symbols.npy").relative_to(REPO)),
            "variance_manifest": str(VARIANCE_MANIFEST.relative_to(REPO)),
        },
        "source_population": {
            "pool_definition": matrix_manifest["pool_definition"],
            "n_samples": matrix_manifest["n_samples"],
            "singlecell_filter_applied": matrix_manifest["singlecell_filter_applied"],
        },
        "not_derived_from": (
            "elasticnet_out/expression/X.npy — wrong population "
            "(CVD-vs-random-negative training pool). Centrality here is "
            "within-CVD co-expression only."
        ),
        "sanity_checks": checks,
        "cardiac_spot_check": spot,
        "outputs": {"ranking": "high_centrality_genes.csv"},
    }
    (args.outdir / "centrality_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    log("")
    log(f"in-degree: min {checks['in_degree_min']}, max {checks['in_degree_max']}, "
        f"mean {checks['in_degree_mean']:.1f}, median {checks['in_degree_median']:.0f}")
    log(f"frac of genes with in-degree in [90,110]: {checks['frac_in_degree_near_100']}")
    log(f"selected {checks['n_selected']} genes (top {top_percent}%)")
    log(f"sanity checks passed: {checks['passed']} {checks['failures']}")
    for name, info in spot.items():
        log(f"  {name}: {info}")
    log(f"wrote {out_csv}")
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
