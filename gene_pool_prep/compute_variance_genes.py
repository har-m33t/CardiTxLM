"""
compute_variance_genes.py — Track 3: rank genes by within-CVD variance.

What this computes, and against what
------------------------------------
Per-gene variance along the *sample* axis of Track 1's CVD-only matrix —
one variance per gene column, across the 8,725 bulk disease-confirmed
samples. High variance within CVD means the gene actually moves across
cardiovascular-disease samples, which is what makes it a useful QA-pool
candidate.

This must run against `cvd_only_expression.npy` and nothing else. Running
it against `elasticnet_out/expression/X.npy` would rank genes by how much
they vary across a CVD-vs-random-negative *training* pool — i.e. mostly
tissue-of-origin signal from the 26,175 random negatives — which is the
exact bug the whole gene-pool-prerequisites effort exists to fix. The
loader hard-fails on a matrix whose manifest doesn't declare a CVD-only
population; see `_validate_manifest`.

Transform
---------
The matrix is already log2(raw_count + 1.0) from Track 1. Variance is taken
on those values as-is — no re-transform, no re-normalisation, no z-scoring.

Variance definition
-------------------
Unbiased sample variance (ddof=1), computed two-pass in float64: streaming
mean first, then streaming sum of squared deviations. Two-pass rather than
the sum/sum-of-squares shortcut because the latter cancels catastrophically
on genes with a high mean and low spread — exactly the housekeeping genes
sitting near the selection boundary. ddof choice cannot change the ranking
(it is a constant factor across genes); it is recorded for citability.

Output
------
high_variance_genes.csv — every gene in the QC universe, not just the
selected ones, so downstream Track 5 can cite a rank for any gene:
    gene, variance, variance_rank (1 = highest), selected (bool)
variance_manifest.json — threshold, counts, sanity checks, provenance.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_CHUNK_SIZE = 512  # samples per streaming pass
VARIANCE_DDOF = 1

# Well-known cardiac genes used only as a plausibility read-out. These are
# reported, never enforced — a gene landing low is a prompt to look, not a
# failure. Chosen because all three are canonical cardiac-stress markers
# with large documented dynamic range in diseased myocardium.
CARDIAC_SPOT_CHECK = ("NPPA", "NPPB", "MYH7", "MYH6", "TTN", "ACTA1")

# Track 1's population. A manifest disagreeing with this means we were
# pointed at the wrong matrix.
EXPECTED_POOL_DEFINITION = "disease_confirmed"
ELASTICNET_POOL_N_SAMPLES = 34900


class InputValidationError(RuntimeError):
    """Raised when the inputs aren't Track 1's CVD-only artifacts."""


class SanityCheckError(RuntimeError):
    """Raised when a post-computation check fails — nothing is written."""


def _validate_manifest(manifest: dict, matrix_path: Path) -> None:
    """Refuse to run against anything but a Track 1 CVD-only matrix."""
    pool = manifest.get("pool_definition")
    if pool != EXPECTED_POOL_DEFINITION:
        raise InputValidationError(
            f"{matrix_path} manifest declares pool_definition={pool!r}, expected "
            f"{EXPECTED_POOL_DEFINITION!r}. This is not Track 1's CVD-only "
            "matrix — refusing to compute variance on the wrong population."
        )
    n_samples = manifest.get("n_samples")
    if not isinstance(n_samples, int) or n_samples >= ELASTICNET_POOL_N_SAMPLES:
        raise InputValidationError(
            f"manifest n_samples={n_samples!r} is not a CVD-only sample count "
            f"(must be < {ELASTICNET_POOL_N_SAMPLES}, the elastic net pool). "
            "This looks like the elastic net training matrix, not Track 1's."
        )
    if not manifest.get("sanity_checks", {}).get("passed"):
        raise InputValidationError(
            f"{matrix_path} manifest records failing sanity checks — Track 1's "
            "output is not trustworthy; rerun Track 1 before Track 3."
        )


def load_inputs(matrix_dir: Path, expression_dir: Path):
    """Load the CVD-only matrix (mmap), its manifest, and the gene universe.

    Cross-checks the gene axis three ways — manifest, `gene_symbols.npy`,
    and `kept_gene_mask.npy` — before any arithmetic happens.
    """
    matrix_path = matrix_dir / "cvd_only_expression.npy"
    manifest_path = matrix_dir / "cvd_only_matrix_manifest.json"
    for p in (matrix_path, manifest_path):
        if not p.exists():
            raise InputValidationError(
                f"{p} not found — Track 1 (matrix materialisation) must complete "
                "first. Do not substitute another matrix."
            )

    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest, matrix_path)

    x = np.load(matrix_path, mmap_mode="r")
    symbols = np.load(expression_dir / "gene_symbols.npy", allow_pickle=True)
    keep_mask = np.load(expression_dir / "kept_gene_mask.npy")

    n_kept = int(keep_mask.sum())
    if not (x.shape[1] == symbols.shape[0] == n_kept == manifest["n_genes"]):
        raise InputValidationError(
            "gene-axis mismatch: matrix has "
            f"{x.shape[1]} columns, gene_symbols.npy has {symbols.shape[0]}, "
            f"kept_gene_mask.npy keeps {n_kept}, manifest records "
            f"{manifest['n_genes']}. All four must agree."
        )
    if x.shape[0] != manifest["n_samples"]:
        raise InputValidationError(
            f"sample-axis mismatch: matrix has {x.shape[0]} rows, manifest "
            f"records {manifest['n_samples']}."
        )
    logger.info(
        "inputs validated: matrix %s, pool=%s, %d genes",
        x.shape, manifest["pool_definition"], x.shape[1],
    )
    return x, manifest, symbols


def compute_variance(x, chunk_size: int = DEFAULT_CHUNK_SIZE) -> np.ndarray:
    """Two-pass per-gene variance across samples, float64 accumulators."""
    n_samples, n_genes = x.shape

    total = np.zeros(n_genes, dtype=np.float64)
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        total += np.asarray(x[start:stop], dtype=np.float64).sum(axis=0)
    mean = total / n_samples
    logger.info("pass 1 (mean) complete over %d samples", n_samples)

    sq_dev = np.zeros(n_genes, dtype=np.float64)
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        block = np.asarray(x[start:stop], dtype=np.float64) - mean
        sq_dev += np.einsum("ij,ij->j", block, block)
    logger.info("pass 2 (squared deviations) complete")

    return sq_dev / (n_samples - VARIANCE_DDOF)


def rank_genes(symbols: np.ndarray, variance: np.ndarray) -> pd.DataFrame:
    """Rank descending by variance; ties broken by gene symbol for determinism."""
    df = pd.DataFrame({"gene": symbols.astype(str), "variance": variance})
    df = df.sort_values(
        ["variance", "gene"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    df["variance_rank"] = np.arange(1, len(df) + 1, dtype=np.int64)
    return df


def select_top_percent(df: pd.DataFrame, top_percent: float) -> tuple[pd.DataFrame, int]:
    """Mark the top `top_percent`% of genes by rank."""
    if not 0.0 < top_percent <= 100.0:
        raise ValueError(f"top_percent must be in (0, 100]; got {top_percent}")
    n_select = int(math.ceil(len(df) * top_percent / 100.0))
    df = df.copy()
    df["selected"] = df["variance_rank"] <= n_select
    return df, n_select


def spot_check_cardiac_genes(df: pd.DataFrame, n_select: int) -> dict:
    """Report where canonical cardiac genes land. Advisory, never enforced."""
    lookup = df.set_index("gene")
    results = {}
    for gene in CARDIAC_SPOT_CHECK:
        if gene not in lookup.index:
            results[gene] = {"present": False}
            continue
        row = lookup.loc[gene]
        rank = int(row["variance_rank"])
        results[gene] = {
            "present": True,
            "variance": float(row["variance"]),
            "variance_rank": rank,
            "percentile": round(100.0 * rank / len(df), 3),
            "selected": bool(rank <= n_select),
        }
    return results


def run_sanity_checks(
    df: pd.DataFrame, variance: np.ndarray, n_select: int, top_percent: float
) -> dict:
    checks: dict = {}
    failures: list[str] = []

    n_nan = int(np.isnan(variance).sum())
    n_inf = int(np.isinf(variance).sum())
    checks["n_nan"] = n_nan
    checks["n_inf"] = n_inf
    checks["no_nan_or_inf"] = (n_nan == 0 and n_inf == 0)
    if not checks["no_nan_or_inf"]:
        failures.append(f"variance has {n_nan} NaN and {n_inf} inf values")

    min_var = float(variance.min())
    checks["min_variance"] = min_var
    checks["max_variance"] = float(variance.max())
    checks["variance_non_negative"] = min_var >= 0.0
    if not checks["variance_non_negative"]:
        failures.append(f"negative variance {min_var} — computation error")

    # Selected count must land within one gene of the requested fraction.
    expected = math.ceil(len(df) * top_percent / 100.0)
    checks["n_genes_total"] = int(len(df))
    checks["n_selected"] = int(n_select)
    checks["n_selected_expected"] = int(expected)
    checks["selected_fraction_pct"] = round(100.0 * n_select / len(df), 4)
    checks["selected_count_consistent"] = abs(n_select - expected) <= 1
    if not checks["selected_count_consistent"]:
        failures.append(
            f"selected {n_select} genes but expected ~{expected} for "
            f"top {top_percent}% of {len(df)}"
        )

    checks["ranks_unique_and_complete"] = bool(
        df["variance_rank"].is_unique and df["variance_rank"].max() == len(df)
    )
    if not checks["ranks_unique_and_complete"]:
        failures.append("variance_rank is not a complete 1..N permutation")

    checks["passed"] = not failures
    checks["failures"] = failures
    if failures:
        raise SanityCheckError(
            "sanity checks failed, nothing written:\n  - " + "\n  - ".join(failures)
        )
    return checks


def run(
    matrix_dir: Path,
    expression_dir: Path,
    outdir: Path,
    top_percent: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    x, matrix_manifest, symbols = load_inputs(matrix_dir, expression_dir)
    variance = compute_variance(x, chunk_size=chunk_size)
    df = rank_genes(symbols, variance)
    df, n_select = select_top_percent(df, top_percent)

    checks = run_sanity_checks(df, variance, n_select, top_percent)
    spot_check = spot_check_cardiac_genes(df, n_select)

    csv_path = outdir / "high_variance_genes.csv"
    df.to_csv(csv_path, index=False)

    manifest = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "top_percent": float(top_percent),
        "n_genes_total": int(len(df)),
        "n_selected": int(n_select),
        "variance_ddof": VARIANCE_DDOF,
        "variance_axis": "samples (axis=0); one variance per gene column",
        "transform_note": (
            "Variance computed on Track 1's existing log2(raw_count + 1.0) "
            "values. No re-transform or re-normalisation applied."
        ),
        "method_note": (
            "Two-pass float64: streaming mean, then streaming sum of squared "
            "deviations. Avoids the catastrophic cancellation of the "
            "sum/sum-of-squares shortcut on high-mean low-spread genes."
        ),
        "sources": {
            "matrix": str(matrix_dir / "cvd_only_expression.npy"),
            "matrix_manifest": str(matrix_dir / "cvd_only_matrix_manifest.json"),
            "gene_symbols": str(expression_dir / "gene_symbols.npy"),
            "kept_gene_mask": str(expression_dir / "kept_gene_mask.npy"),
        },
        "source_population": {
            "pool_definition": matrix_manifest["pool_definition"],
            "pool_definition_column": matrix_manifest["pool_definition_column"],
            "n_samples": matrix_manifest["n_samples"],
            "singlecell_filter_applied": matrix_manifest["singlecell_filter_applied"],
        },
        "not_derived_from": (
            "elasticnet_out/expression/X.npy — wrong population (CVD-vs-random-"
            "negative training pool). Variance here is within-CVD only."
        ),
        "sanity_checks": checks,
        "cardiac_spot_check": spot_check,
        "outputs": {"ranking": "high_variance_genes.csv"},
    }
    with open(outdir / "variance_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("selected %d / %d genes (top %.4g%%)", n_select, len(df), top_percent)
    logger.info("wrote %s", csv_path)
    return csv_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track 3 — variance-based gene ranking.")
    p.add_argument(
        "--matrix-dir", required=True, type=Path,
        help="Directory holding Track 1's cvd_only_expression.npy + manifest.",
    )
    p.add_argument(
        "--expression-dir", required=True, type=Path,
        help="elasticnet_out/expression — source of the QC gene universe.",
    )
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument(
        "--top-percent", required=True, type=float,
        help="No default: the top-N%% threshold is a recorded decision "
             "(gene_pool_prerequisites_todo.md, decision 2), not a flag default.",
    )
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(
        matrix_dir=args.matrix_dir,
        expression_dir=args.expression_dir,
        outdir=args.outdir,
        top_percent=args.top_percent,
        chunk_size=args.chunk_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
