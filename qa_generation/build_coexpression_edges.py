"""Materialize the top-100 co-expression edge list used by Interaction Network Query.

Track 4 (`gene_pool_prep/compute_centrality_genes.py`) computed exactly these
correlations but persisted only the in-degree *counts* — `top` is a local that is
folded into `np.bincount` and dropped (`compute_centrality_genes.py:182-183`).
The partner sets themselves were never written to disk, so `gt_functions.py`
has nothing to read. This script recovers them.

Methodology is Track 4's, not a new one:

  * same source matrix   — `cvd_only_expression.npy` (8,553 disease-confirmed
    samples x 49,231 QC genes, log2(count + 1))
  * same standardization — Pearson via z-scored columns scaled by sqrt(n - 1),
    zero-variance columns zeroed
  * same partner universe — ALL 49,231 QC columns are eligible partners; the
    correlation is not restricted to pool-to-pool
  * same top-k          — 100 partners per gene, self-correlation excluded

Two narrowings are applied to *which genes participate*, neither of which touches
the correlation method itself:

  * **Rows** — only curated-pool genes, since those are the only ones QA asks
    about. Now the 5,797-gene BulkFormer-filtered pool.
  * **Partner candidacy** — restricted to symbols carrying an in-vocabulary ENSG.
    Per `gene_pool_prep/bulkformer_vocab_check.md`, a partner outside BulkFormer's
    20,010-gene vocabulary is a fact about a gene the model never receives as
    input, so it cannot be a valid answer. Standardization still happens over all
    49,231 columns, so every gene's z-scores — and therefore every correlation
    value — are numerically identical to Track 4's; only the set of columns
    eligible to be *selected* shrinks.

Two column-level details Track 4 did not have to resolve, because in-degree is a
column-level statistic while QA is symbol-level:

  * A pool symbol carried by k > 1 columns contributes all k columns as source
    rows; their partner sets are merged, keeping the highest r per partner
    symbol. This is the union of Track 4's edges leaving that symbol's nodes.
  * Partners sharing the source symbol are dropped. Track 4 kept them but
    flagged them as an artifact (`centrality_manifest.json`
    -> `n_same_symbol_edges`: 6,470 edges among byte-identical duplicate
    columns, which are guaranteed to be each other's top partner). "TTC34 is
    co-expressed with TTC34" is not a usable QA fact.

Outputs `coexpression/coexpression_edges.parquet` and a manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CVD_DIR = REPO / "eda/dataset/cvd_data"
MATRIX = CVD_DIR / "gene_pool_prep/cvd_only_expression.npy"
MATRIX_MANIFEST = CVD_DIR / "gene_pool_prep/cvd_only_matrix_manifest.json"
GENE_SYMBOLS = CVD_DIR / "elasticnet_out/expression/gene_symbols.npy"
# The BulkFormer-filtered pool, adopted per bulkformer_vocab_check.md. The
# original curated_gene_pool.csv is retained on disk for provenance but is no
# longer read by any QA-generation code.
GENE_POOL = REPO / "gene_pool_prep/curated_gene_pool_bulkformer_filtered.csv"
SYMBOL_VOCAB_MAP = REPO / "qa_generation/bulkformer_input/symbol_vocab_map.parquet"
OUTDIR = REPO / "qa_generation/coexpression"

TOP_K = 100  # partners kept per gene — Track 4's TOP_K
K_RAW = 400  # columns pulled per source column before symbol-dedup


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def standardize(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Z-score columns and scale by sqrt(n - 1), so Z.T @ Z is Pearson.

    Ported from compute_centrality_genes.py:standardize — same float64
    accumulation, same float32 storage, same zero-variance handling.
    """
    X = np.load(path, mmap_mode="r")
    n_samples, n_genes = X.shape
    log(f"standardizing {n_samples} x {n_genes}")

    Z = np.empty((n_samples, n_genes), dtype=np.float32)
    stds = np.empty(n_genes, dtype=np.float64)
    denom = np.sqrt(n_samples - 1.0)
    for start in range(0, n_genes, 4096):
        stop = min(start + 4096, n_genes)
        chunk = np.asarray(X[:, start:stop], dtype=np.float64)
        centered = chunk - chunk.mean(axis=0)
        sd = np.sqrt((centered**2).sum(axis=0) / (n_samples - 1.0))
        stds[start:stop] = sd
        safe = np.where(sd > 0, sd, 1.0)
        Z[:, start:stop] = (centered / (safe * denom)).astype(np.float32)

    zero_var = stds <= 0
    if zero_var.any():
        Z[:, zero_var] = 0.0
    if not np.isfinite(Z).all():
        raise SystemExit("STOP: non-finite values in standardized matrix")
    log(f"  standardized, {int(zero_var.sum())} zero-variance columns")
    return Z, zero_var


def build_edges(
    Z: np.ndarray,
    symbols: np.ndarray,
    pool_genes: list[str],
    vocab_symbols: set[str],
    block: int,
) -> tuple[pd.DataFrame, dict]:
    """Top-K_RAW partner columns per pool source column, merged to symbols.

    Partner candidates are restricted to columns whose symbol carries an
    in-vocabulary ENSG. Z itself is still standardized over all 49,231 columns,
    so the correlation values are unchanged from Track 4's — this narrows only
    which columns can be *selected*.
    """
    cols_by_symbol: dict[str, list[int]] = defaultdict(list)
    for idx, sym in enumerate(symbols):
        cols_by_symbol[sym].append(idx)

    source_cols = np.array(
        sorted(c for g in pool_genes for c in cols_by_symbol[g]), dtype=np.int64
    )
    log(f"{len(pool_genes)} pool genes -> {len(source_cols)} source columns")

    partner_cols = np.array(
        [i for i, sym in enumerate(symbols) if sym in vocab_symbols], dtype=np.int64
    )
    log(
        f"partner candidates: {len(partner_cols)}/{Z.shape[1]} columns "
        f"in BulkFormer vocabulary"
    )
    Zp = np.ascontiguousarray(Z[:, partner_cols])
    partner_symbols = symbols[partner_cols]
    # Positions *within partner space* for each symbol, for self-symbol masking.
    partner_pos_by_symbol: dict[str, list[int]] = defaultdict(list)
    for pos, sym in enumerate(partner_symbols):
        partner_pos_by_symbol[sym].append(pos)

    # partner symbol -> best r seen, across every source column of this symbol
    best: dict[str, dict[str, float]] = {g: {} for g in pool_genes}
    corr_min, corr_max = np.inf, -np.inf

    n_blocks = (len(source_cols) + block - 1) // block
    for bi, start in enumerate(range(0, len(source_cols), block)):
        cols = source_cols[start : start + block]
        block_t = np.ascontiguousarray(Z[:, cols].T)
        C = block_t @ Zp  # (B x n_partner_candidates) Pearson correlations

        bad = ~np.isfinite(C)
        if bad.any():
            C[bad] = -np.inf

        # Every partner column sharing this row's symbol is excluded, which
        # subsumes Track 4's diagonal exclusion and additionally drops the
        # duplicate-column artifact edges its manifest flagged.
        for row, col in enumerate(cols):
            same = partner_pos_by_symbol.get(symbols[col])
            if same:
                C[row, same] = -np.inf

        finite = C[np.isfinite(C)]
        if finite.size:
            corr_min = min(corr_min, float(finite.min()))
            corr_max = max(corr_max, float(finite.max()))

        top = np.argpartition(-C, K_RAW, axis=1)[:, :K_RAW]
        for row, col in enumerate(cols):
            partners = top[row]
            rs = C[row, partners]
            order = np.argsort(-rs, kind="stable")
            slot = best[symbols[col]]
            for p, r in zip(partners[order], rs[order]):
                if not np.isfinite(r):
                    continue
                psym = partner_symbols[p]
                if r > slot.get(psym, -np.inf):
                    slot[psym] = float(r)

        if bi % 5 == 0 or start + block >= len(source_cols):
            log(f"  block {bi + 1}/{n_blocks}")

    rows = []
    short = []
    for gene in pool_genes:
        ranked = sorted(best[gene].items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K]
        if len(ranked) < TOP_K:
            short.append(gene)
        for rank, (partner, r) in enumerate(ranked, start=1):
            rows.append((gene, partner, rank, r))

    if short:
        raise SystemExit(
            f"STOP: {len(short)} genes yielded < {TOP_K} distinct partner symbols "
            f"from the top {K_RAW} columns (e.g. {short[:5]}) — raise K_RAW."
        )

    edges = pd.DataFrame(rows, columns=["gene", "partner", "rank", "pearson_r"])
    edges["rank"] = edges["rank"].astype(np.int16)
    edges["pearson_r"] = edges["pearson_r"].astype(np.float32)
    stats = {
        "n_source_columns": int(len(source_cols)),
        "n_partner_candidate_columns": int(len(partner_cols)),
        "n_partner_candidate_symbols": int(len(set(partner_symbols))),
        "corr_min": corr_min,
        "corr_max": corr_max,
    }
    return edges, stats


def verify(
    edges: pd.DataFrame,
    pool_genes: list[str],
    symbols: np.ndarray,
    vocab_symbols: set[str],
) -> dict:
    """Structural checks plus an independent Pearson recomputation."""
    checks: dict = {}
    failures: list[str] = []

    per_gene = edges.groupby("gene").size()
    checks["n_genes"] = int(len(per_gene))
    checks["all_genes_have_top_k"] = bool((per_gene == TOP_K).all())
    checks["covers_full_pool"] = set(per_gene.index) == set(pool_genes)
    checks["no_self_symbol_edges"] = bool((edges.gene != edges.partner).all())
    checks["corr_within_bounds"] = bool(edges.pearson_r.abs().max() <= 1.001)
    # The point of the rebuild: no answer may name a gene the model cannot see.
    oov = sorted(set(edges.partner.astype(str)) - vocab_symbols)
    checks["n_out_of_vocab_partners"] = len(oov)
    checks["all_partners_in_vocab"] = not oov
    checks["all_sources_in_vocab"] = not (set(edges.gene.astype(str)) - vocab_symbols)

    ranks_ok = True
    monotone_ok = True
    for _, grp in edges.groupby("gene", sort=False):
        if list(grp["rank"]) != list(range(1, TOP_K + 1)):
            ranks_ok = False
        if not (np.diff(grp.pearson_r.to_numpy()) <= 1e-6).all():
            monotone_ok = False
    checks["ranks_complete"] = ranks_ok
    checks["r_non_increasing_with_rank"] = monotone_ok

    # Independent recomputation: np.corrcoef straight off the raw matrix, no
    # z-scored shortcut, for a spread of ranks across a few genes.
    X = np.load(MATRIX, mmap_mode="r")
    col_of = defaultdict(list)
    for idx, sym in enumerate(symbols):
        col_of[sym].append(idx)

    rng = np.random.default_rng(0)
    sample = edges.iloc[rng.choice(len(edges), size=25, replace=False)]
    max_err = 0.0
    for _, row in sample.iterrows():
        # The stored r is the max over the symbol's source columns.
        best_r = -np.inf
        for gc in col_of[row.gene]:
            a = np.asarray(X[:, gc], dtype=np.float64)
            for pc in col_of[row.partner]:
                b = np.asarray(X[:, pc], dtype=np.float64)
                if a.std() == 0 or b.std() == 0:
                    continue
                best_r = max(best_r, float(np.corrcoef(a, b)[0, 1]))
        max_err = max(max_err, abs(best_r - float(row.pearson_r)))
    checks["independent_recompute_n"] = int(len(sample))
    checks["independent_recompute_max_abs_err"] = max_err
    checks["independent_recompute_matches"] = bool(max_err < 1e-4)

    for key, val in checks.items():
        if isinstance(val, bool) and not val:
            failures.append(key)
    checks["passed"] = not failures
    checks["failures"] = failures
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    for path in (MATRIX, MATRIX_MANIFEST, GENE_SYMBOLS, GENE_POOL, SYMBOL_VOCAB_MAP):
        if not path.exists():
            raise SystemExit(f"STOP: missing required input {path}")

    vocab_symbols = set(
        pd.read_parquet(SYMBOL_VOCAB_MAP).gene_symbol.astype(str)
    )
    log(f"{len(vocab_symbols)} symbols carry an in-vocabulary ENSG")

    matrix_manifest = json.loads(MATRIX_MANIFEST.read_text())
    symbols = np.array(
        [str(s) for s in np.load(GENE_SYMBOLS, allow_pickle=True)], dtype=object
    )
    if len(symbols) != matrix_manifest["n_genes"]:
        raise SystemExit(
            f"STOP: gene_symbols.npy has {len(symbols)} entries, matrix manifest "
            f"reports {matrix_manifest['n_genes']}"
        )

    pool = pd.read_csv(GENE_POOL)
    pool_genes = sorted(set(pool["gene"].astype(str)))
    known = set(symbols)
    unknown = [g for g in pool_genes if g not in known]
    if unknown:
        raise SystemExit(
            f"STOP: {len(unknown)} pool genes absent from the matrix "
            f"(e.g. {unknown[:5]}) — pool and matrix are out of sync."
        )

    Z, _ = standardize(MATRIX)
    edges, stats = build_edges(Z, symbols, pool_genes, vocab_symbols, args.block_size)
    del Z

    log("verifying")
    checks = verify(edges, pool_genes, symbols, vocab_symbols)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "coexpression_edges.parquet"
    edges.to_parquet(out, index=False)

    finished = datetime.now(timezone.utc)
    manifest = {
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "elapsed_seconds": round((finished - started).total_seconds(), 1),
        "purpose": (
            "Top-100 co-expression partner list per curated-pool gene, for "
            "interaction_network_query in qa_generation/gt_functions.py."
        ),
        "why_this_exists": (
            "Track 4 (gene_pool_prep/compute_centrality_genes.py) computed the "
            "same correlations but persisted only in-degree counts; the "
            "top-100 partner sets were never written to disk."
        ),
        "methodology": {
            "inherited_from": "gene_pool_prep/compute_centrality_genes.py",
            "correlation": "Pearson across samples, blocked sgemm on z-scored columns",
            "partner_universe": (
                f"{stats['n_partner_candidate_columns']} of {len(symbols)} "
                "QC-filtered columns — those whose symbol carries an "
                "in-vocabulary ENSG. Not restricted to the curated pool (Track 4 "
                "behaviour retained), but restricted to genes BulkFormer "
                "actually receives, per bulkformer_vocab_check.md."
            ),
            "partner_restriction_rationale": (
                "a co-expression partner outside the 20,010-gene vocabulary is a "
                "fact about a gene the model cannot see, so it cannot be a valid "
                "answer; standardization still spans all columns, so correlation "
                "values are unchanged"
            ),
            "source_rows": (
                f"{len(pool_genes)} curated-pool genes only (Track 4 used all "
                f"{len(symbols)} columns as rows because in-degree needs every "
                "edge; QA only ever queries pool genes)"
            ),
            "top_k_partners": TOP_K,
            "k_raw_columns_per_source": K_RAW,
            "duplicate_symbol_columns": (
                "all columns carrying a pool symbol act as source rows; their "
                "partner sets are merged keeping the highest r per partner symbol"
            ),
            "self_symbol_partners": (
                "excluded — subsumes Track 4's diagonal exclusion and drops the "
                "duplicate-column artifact edges its manifest flagged "
                "(n_same_symbol_edges = 6470)"
            ),
            "tie_handling": "ties on r resolved by partner symbol, ascending",
        },
        "source_population": {
            "pool_definition": matrix_manifest["pool_definition"],
            "n_samples": matrix_manifest["n_samples"],
            "n_genes": matrix_manifest["n_genes"],
        },
        "sources": {
            "matrix": str(MATRIX.relative_to(REPO)),
            "gene_symbols": str(GENE_SYMBOLS.relative_to(REPO)),
            "gene_pool": str(GENE_POOL.relative_to(REPO)),
        },
        "n_pool_genes": len(pool_genes),
        "n_edges": int(len(edges)),
        "corr_min": stats["corr_min"],
        "corr_max": stats["corr_max"],
        "n_source_columns": stats["n_source_columns"],
        "sanity_checks": checks,
        "outputs": {"edges": out.name},
    }
    (args.outdir / "coexpression_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    log(f"wrote {len(edges)} edges -> {out}")
    if not checks["passed"]:
        log(f"FAILED checks: {checks['failures']}")
        return 1
    log(f"all checks passed ({manifest['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
