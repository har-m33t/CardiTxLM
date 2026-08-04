"""Deterministic ground-truth computation for QA generation (Step 1).

One function per verified template category. Each takes a sample id plus any
bound placeholder values and returns the real answer computed from project
data. No LLM is involved: these produce the facts that Step 4's paraphrasing
model rewords, and it is never allowed to invent what is missing here.

Every function returns a `GTResult`. When the required source data does not
cover the given sample or arguments, the result carries
`status == "insufficient_data"` and a machine-readable `reason` — never an
approximated or fabricated answer.

All numeric expression values are **BulkFormer's actual model input** — TPM ->
log1p over the 20,010-gene ENSG vocabulary, materialized by
`build_bulkformer_matrix.py` from `linear_probe/extract.py`'s own transform. The
project's `cvd_only_expression.npy` (log2(count + 1), 49,231 QC genes) is no
longer read here: `gene_pool_prep/bulkformer_vocab_check.md` showed the two
disagree on gene *ordering*, so GT computed from it did not describe what the
model receives.

Scope constraints enforced at this layer, not left to the generation prompt:

  * `gene_driver_reasoning` is broad-CVD only. The elastic-net ranking is a
    single global CVD-vs-random-tissue model (see
    `gene_pool_prep/elastic_net_ranking_audit.md`); no per-subtype ranking
    exists. There is deliberately no subtype-conditioned variant of this
    function, and it takes no `condition` argument.
  * Every gene answered about is in BulkFormer's vocabulary. The gene pool is the
    5,797-gene BulkFormer-filtered pool, co-expression partners are
    vocabulary-restricted, and `gene_driver_reasoning`'s ranking is filtered the
    same way. A question about a gene the model never receives cannot be grounded.
  * `comparative_differential_reasoning` accepts only the `neg_hard` comparison
    group. The random-bulk-tissue group is confounded — the linear probe's own
    ROC-AUC falls from 0.925 to 0.781 once tissue identity is controlled for —
    so any other group returns insufficient_data rather than a substituted one.

Populations differ by category and do not nest cleanly. Step 2's sampling plan
should intersect them explicitly rather than assume a single eligible set:

  * 8,553 samples have an expression row (`bulkformer_expression.npy`) — the only
    samples any Stage 1 function can answer for. Same rows as the old
    `cvd_only_expression.npy`, so this correction shifted no population.
  * 10,557 samples are `is_cvd_disease` — the gate for `gene_driver_reasoning`.
  * 8,725 samples are probe positives — the gate for
    `comparative_differential_reasoning`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CVD_DIR = REPO / "eda/dataset/cvd_data"

BULKFORMER_DIR = REPO / "qa_generation/bulkformer_input"
EXPRESSION_NPY = BULKFORMER_DIR / "bulkformer_expression.npy"
EXPRESSION_SAMPLE_INDEX = BULKFORMER_DIR / "bulkformer_sample_index.npy"
SYMBOL_VOCAB_MAP = BULKFORMER_DIR / "symbol_vocab_map.parquet"
SAMPLE_LABELS = CVD_DIR / "extended_eda_out/labels/sample_labels.parquet"
GENE_RANKING = CVD_DIR / "elasticnet_out/gene_signal/gene_signal_ranking.csv"
# The BulkFormer-filtered pool, adopted per bulkformer_vocab_check.md. The
# original curated_gene_pool.csv stays on disk for provenance but is not read.
GENE_POOL = REPO / "gene_pool_prep/curated_gene_pool_bulkformer_filtered.csv"
PROBE_LABELS = REPO / "linear_probe/probe_sample_labels.parquet"
PROBE_RESULTS = REPO / "linear_probe/results"
COEXPRESSION_EDGES = REPO / "qa_generation/coexpression/coexpression_edges.parquet"

INSUFFICIENT_DATA = "insufficient_data"
OK = "ok"

#: Expression units. BulkFormer's input transform: raw ARCHS4 counts divided by
#: gene length, scaled to TPM, then log1p — `linear_probe/extract.py:159-170`.
EXPRESSION_UNITS = "log1p(TPM)"

#: The `{comparison_group}` bindings this layer will answer for. Both name the
#: same pool; `stage2.yaml` refers to it by the subtype string, the linear probe
#: by the column name.
VALID_COMPARISON_GROUPS = frozenset({"neg_hard", "tissue_only_disease_unconfirmed"})

#: Subtype values that are not a resolved disease subtype.
NON_SUBTYPE_LABELS = frozenset(
    {"", "disease_matched_subtype_unresolved", "tissue_only_disease_unconfirmed"}
)

_ABOVE = frozenset({"above", "exceed", "exceeds", "greater", "higher", "over"})
_BELOW = frozenset({"below", "under", "less", "lower", "beneath"})
_TOP = frozenset({"top", "highest", "upper", "greatest", "first"})
_BOTTOM = frozenset({"bottom", "lowest", "least", "last"})


class MissingArtifactError(RuntimeError):
    """A required input file is absent.

    Distinct from insufficient_data on purpose: insufficient_data means the data
    exists but does not cover this sample, which is a normal, expected outcome
    that Step 2 filters on. A missing artifact is a broken installation and must
    never be silently reported as a data-coverage gap.
    """


@dataclass(frozen=True)
class GTResult:
    """The computed answer, or an explicit refusal to answer."""

    category: str
    sample_id: str
    status: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == OK

    def __bool__(self) -> bool:
        return self.ok


def _insufficient(category: str, sample_id: Any, reason: str) -> GTResult:
    return GTResult(
        category=category, sample_id=str(sample_id), status=INSUFFICIENT_DATA,
        payload={}, reason=reason,
    )


def _require(path: Path) -> Path:
    if not path.exists():
        raise MissingArtifactError(
            f"required input not found: {path}. "
            f"See qa_generation/gt_functions_report.md for how it is produced."
        )
    return path


# ---------------------------------------------------------------------------
# Loaders. Cached because Step 3 calls these functions hundreds of thousands of
# times; the expression matrix itself stays memory-mapped and is never read
# whole.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _symbol_columns() -> dict[str, np.ndarray]:
    """Gene symbol -> its column(s) in BulkFormer's 20,010-gene input vector.

    Built by `build_bulkformer_matrix.py` from the H5 gene axis, pairing
    `meta/genes/symbol` with `meta/genes/ensembl_gene` positionally and looking
    the ENSG up in the vocabulary — the reconciliation established in
    `bulkformer_vocab_check.md` section 2, not a name-match.

    Only 3 of 20,007 symbols (PINX1, POLR2J3, TBCE) occupy two vocabulary
    columns; their reported expression is the mean of the two, the same rule the
    old matrix path used for redundant columns.
    """
    mapping = pd.read_parquet(_require(SYMBOL_VOCAB_MAP))
    out: dict[str, list[int]] = {}
    for sym, pos in zip(mapping.gene_symbol.astype(str), mapping.vocab_pos):
        out.setdefault(sym, []).append(int(pos))
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in out.items()}


@lru_cache(maxsize=1)
def _expression() -> np.memmap:
    return np.load(_require(EXPRESSION_NPY), mmap_mode="r")


@lru_cache(maxsize=1)
def _sample_labels() -> pd.DataFrame:
    lab = pd.read_parquet(_require(SAMPLE_LABELS))
    return lab.set_index("geo_accession", drop=False)


@lru_cache(maxsize=1)
def _expression_rows() -> dict[str, int]:
    """geo_accession -> row in bulkformer_expression.npy."""
    idx = np.load(_require(EXPRESSION_SAMPLE_INDEX))
    lab = _sample_labels()
    accessions = lab.loc[lab.sample_index.isin(idx)].sort_values("sample_index")
    order = {int(s): i for i, s in enumerate(idx)}
    return {
        str(acc): order[int(si)]
        for acc, si in zip(accessions.geo_accession, accessions.sample_index)
    }


@lru_cache(maxsize=1)
def _pool_genes() -> tuple[str, ...]:
    pool = pd.read_csv(_require(GENE_POOL))
    return tuple(sorted(set(pool["gene"].astype(str))))


@lru_cache(maxsize=1)
def _pool_segments() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat column indices, reduceat offsets and per-gene column counts.

    Lets a whole-pool expression vector be built with one fancy-index and one
    segmented sum instead of 7,166 dict lookups per sample.
    """
    cols_by_symbol = _symbol_columns()
    flat: list[int] = []
    offsets: list[int] = []
    counts: list[int] = []
    for gene in _pool_genes():
        cols = cols_by_symbol[gene]
        offsets.append(len(flat))
        counts.append(len(cols))
        flat.extend(int(c) for c in cols)
    return (
        np.array(flat, dtype=np.int64),
        np.array(offsets, dtype=np.int64),
        np.array(counts, dtype=np.float64),
    )


@lru_cache(maxsize=1)
def _stable_signal_genes() -> pd.DataFrame:
    """The cross-fold-stable subset of the elastic-net ranking.

    `nonzero_frac == 1.0` means the gene held a nonzero coefficient in all five
    outer folds. The remaining ~96% of the ranking is zero-coefficient noise and
    must never be sampled from. Genes outside BulkFormer's vocabulary are then
    dropped, since the model never receives them.
    """
    rank = pd.read_csv(_require(GENE_RANKING))
    stable = rank[rank.nonzero_frac == 1.0].copy()
    # A symbol spanning several columns appears once per column; keep its
    # strongest coefficient so the answer lists each gene exactly once.
    stable = (
        stable.sort_values("abs_mean_coef", ascending=False)
        .drop_duplicates(subset="gene_symbol", keep="first")
        .reset_index(drop=True)
    )
    # Drop drivers outside BulkFormer's vocabulary. This is a filter on the
    # existing ranking, not a re-run: the elastic net's coefficients are
    # untouched, and genes the model never receives simply cannot be named as
    # drivers of what it sees. Removes 92 of 1,234, including RPL23AP42 (rank 8)
    # and NEAT1 (rank 9) — see bulkformer_vocab_check.md section 6.2.
    in_vocab = set(_symbol_columns())
    stable = stable[stable.gene_symbol.astype(str).isin(in_vocab)].reset_index(
        drop=True
    )
    stable["direction"] = np.where(stable.mean_coef > 0, "up_in_cvd", "down_in_cvd")
    stable["rank"] = np.arange(1, len(stable) + 1)
    return stable


@lru_cache(maxsize=1)
def _stable_counts() -> dict[str, int]:
    """Stable-gene counts at each filtering stage, so the figures reconcile."""
    rank = pd.read_csv(_require(GENE_RANKING))
    stable = rank[rank.nonzero_frac == 1.0]
    deduped = stable.drop_duplicates(subset="gene_symbol")
    in_vocab = set(_symbol_columns())
    return {
        "rows_before_symbol_dedup": int(len(stable)),
        "unique_symbols": int(len(deduped)),
        "dropped_out_of_vocab": int(
            (~deduped.gene_symbol.astype(str).isin(in_vocab)).sum()
        ),
    }


@lru_cache(maxsize=1)
def _coexpression_edges() -> dict[str, list[tuple[str, float]]]:
    """gene -> [(partner, pearson_r), ...] ordered by rank."""
    edges = pd.read_parquet(_require(COEXPRESSION_EDGES)).sort_values(
        ["gene", "rank"], kind="stable"
    )
    out: dict[str, list[tuple[str, float]]] = {}
    for gene, grp in edges.groupby("gene", sort=False):
        out[str(gene)] = list(
            zip(grp.partner.astype(str), grp.pearson_r.astype(float))
        )
    return out


@lru_cache(maxsize=1)
def _probe_labels() -> pd.DataFrame:
    return pd.read_parquet(_require(PROBE_LABELS)).set_index("geo_accession")


@lru_cache(maxsize=1)
def _probe_results() -> dict[str, dict[str, Any]]:
    """{negative_pool: {variant: summary}} from the linear probe run."""
    out: dict[str, dict[str, Any]] = {}
    root = _require(PROBE_RESULTS)
    for path in sorted(root.glob("*/*/probe_results.json")):
        res = json.loads(path.read_text())
        out.setdefault(res["negative_pool"], {})[res["variant"]] = {
            "roc_auc_mean": res["summary"]["roc_auc_mean"],
            "roc_auc_std": res["summary"]["roc_auc_std"],
            "n_positive": res["n_positive"],
            "n_negative": res["n_negative"],
            "n_series": res["n_series"],
            "k_folds": res["k_folds"],
        }
    return out


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_sample(sample_id: Any) -> str | None:
    """Accept a GEO accession or a global sample_index; return the accession."""
    if isinstance(sample_id, (int, np.integer)) and not isinstance(sample_id, bool):
        lab = _sample_labels()
        hit = lab.loc[lab.sample_index == int(sample_id), "geo_accession"]
        return str(hit.iloc[0]) if len(hit) else None
    sid = str(sample_id)
    return sid if sid in _sample_labels().index else None


def _row(accession: str) -> np.ndarray | None:
    """This sample's 20,010-column BulkFormer input row, or None if absent."""
    row_idx = _expression_rows().get(accession)
    if row_idx is None:
        return None
    return np.asarray(_expression()[row_idx], dtype=np.float64)


def _gene_value(row: np.ndarray, gene: str) -> tuple[float, int] | None:
    """(mean log1p(TPM) across the symbol's vocabulary columns, n columns)."""
    cols = _symbol_columns().get(gene)
    if cols is None:
        return None
    return float(row[cols].mean()), int(len(cols))


def _pool_vector(row: np.ndarray) -> np.ndarray:
    """Pool-gene expression vector, aligned to `_pool_genes()` order."""
    flat, offsets, counts = _pool_segments()
    return np.add.reduceat(row[flat], offsets) / counts


def _direction(value: str, positive: frozenset, negative: frozenset) -> str | None:
    token = str(value).strip().lower()
    if token in positive:
        return sorted(positive)[0] if token in positive else None
    return None


def _canon_direction(value: Any) -> str | None:
    token = str(value).strip().lower()
    if token in _ABOVE:
        return "above"
    if token in _BELOW:
        return "below"
    return None


def _canon_rank_direction(value: Any) -> str | None:
    token = str(value).strip().lower()
    if token in _TOP:
        return "top"
    if token in _BOTTOM:
        return "bottom"
    return None


def _parse_size(spec: Any, universe: int) -> tuple[int, str, float | None] | None:
    """Resolve {N} or {percentile} to a concrete count.

    Accepts an int count, a "10%" string, or a fraction in (0, 1). Returns
    (n, mode, percentile) or None if the spec is unusable.
    """
    percentile: float | None = None
    if isinstance(spec, str):
        text = spec.strip().rstrip("%")
        try:
            value = float(text)
        except ValueError:
            return None
        if spec.strip().endswith("%"):
            percentile = value
        elif value.is_integer():
            return (int(value), "count", None) if 0 < value <= universe else None
        else:
            percentile = value * 100.0
    elif isinstance(spec, bool):
        return None
    elif isinstance(spec, (int, np.integer)):
        return (int(spec), "count", None) if 0 < int(spec) <= universe else None
    elif isinstance(spec, (float, np.floating)):
        if not math.isfinite(float(spec)):
            return None
        if float(spec).is_integer():
            value = int(spec)
            return (value, "count", None) if 0 < value <= universe else None
        percentile = float(spec) * 100.0
    else:
        return None

    if percentile is None or not (0.0 < percentile <= 100.0):
        return None
    n = int(math.ceil(percentile / 100.0 * universe))
    return (max(n, 1), "percentile", percentile)


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


def direct_abundance_query(sample_id: Any, gene: str) -> GTResult:
    """That gene's expression value in that sample."""
    category = "direct_abundance_query"
    accession = _resolve_sample(sample_id)
    if accession is None:
        return _insufficient(category, sample_id, "unknown_sample_id")
    row = _row(accession)
    if row is None:
        return _insufficient(category, accession, "sample_not_in_expression_matrix")
    hit = _gene_value(row, str(gene))
    if hit is None:
        return _insufficient(category, accession, "gene_not_in_bulkformer_vocab")

    value, n_cols = hit
    return GTResult(
        category=category, sample_id=accession, status=OK,
        payload={
            "gene": str(gene),
            "expression": round(value, 4),
            "units": EXPRESSION_UNITS,
            "n_vocab_columns": n_cols,
            "in_curated_pool": str(gene) in set(_pool_genes()),
        },
    )


def threshold_query(
    sample_id: Any, threshold: float, direction: str = "above"
) -> GTResult:
    """Genes in that sample whose expression is strictly above/below a threshold.

    The universe is the 7,166-gene curated pool, not all 49,231 QC columns:
    the pool is the documented gene universe QA operates over, and an answer
    listing tens of thousands of genes is not a usable training target.
    """
    category = "threshold_query"
    accession = _resolve_sample(sample_id)
    if accession is None:
        return _insufficient(category, sample_id, "unknown_sample_id")
    canon = _canon_direction(direction)
    if canon is None:
        return _insufficient(category, accession, f"unrecognized_direction:{direction}")
    try:
        cut = float(threshold)
    except (TypeError, ValueError):
        return _insufficient(category, accession, "threshold_not_numeric")
    if not math.isfinite(cut):
        return _insufficient(category, accession, "threshold_not_finite")
    row = _row(accession)
    if row is None:
        return _insufficient(category, accession, "sample_not_in_expression_matrix")

    genes = np.array(_pool_genes(), dtype=object)
    values = _pool_vector(row)
    mask = values > cut if canon == "above" else values < cut
    hits = np.flatnonzero(mask)
    order = hits[np.argsort(-values[hits] if canon == "above" else values[hits],
                            kind="stable")]

    return GTResult(
        category=category, sample_id=accession, status=OK,
        payload={
            "threshold": cut,
            "direction": canon,
            "comparison": "strictly_greater" if canon == "above" else "strictly_less",
            "units": EXPRESSION_UNITS,
            "n_universe": int(len(genes)),
            "n_matching": int(len(order)),
            "genes": [str(g) for g in genes[order]],
            "expression": [round(float(v), 4) for v in values[order]],
            # 0 matches and "everything matches" are both true answers but make
            # degenerate QA items; Step 2 should drop rather than reword them.
            "degenerate": bool(len(order) == 0 or len(order) == len(genes)),
        },
    )


def ranking_query(
    sample_id: Any, n_or_percentile: Any, direction: str = "top"
) -> GTResult:
    """Top/bottom N genes, or the top/bottom percentile, by expression.

    `n_or_percentile` takes an int count, a "10%" string, or a fraction in
    (0, 1). Ties are broken by gene symbol so the answer is reproducible, and
    `boundary_tie` flags the case where the cut falls inside a run of equal
    values — those items are ambiguous and Step 2 should drop them.
    """
    category = "ranking_query"
    accession = _resolve_sample(sample_id)
    if accession is None:
        return _insufficient(category, sample_id, "unknown_sample_id")
    canon = _canon_rank_direction(direction)
    if canon is None:
        return _insufficient(category, accession, f"unrecognized_direction:{direction}")
    row = _row(accession)
    if row is None:
        return _insufficient(category, accession, "sample_not_in_expression_matrix")

    genes = np.array(_pool_genes(), dtype=object)
    values = _pool_vector(row)
    parsed = _parse_size(n_or_percentile, len(genes))
    if parsed is None:
        return _insufficient(
            category, accession, f"unusable_size_spec:{n_or_percentile!r}"
        )
    n, mode, percentile = parsed

    sign = -1.0 if canon == "top" else 1.0
    order = np.lexsort((genes.astype(str), sign * values))
    selected = order[:n]
    boundary_tie = bool(
        n < len(genes) and values[order[n - 1]] == values[order[n]]
    )

    return GTResult(
        category=category, sample_id=accession, status=OK,
        payload={
            "direction": canon,
            "mode": mode,
            "n": int(n),
            "percentile": percentile,
            "units": EXPRESSION_UNITS,
            "n_universe": int(len(genes)),
            "genes": [str(g) for g in genes[selected]],
            "expression": [round(float(v), 4) for v in values[selected]],
            "tie_break": "gene_symbol_ascending",
            "boundary_tie": boundary_tie,
        },
    )


def comparative_query(sample_id: Any, gene_a: str, gene_b: str) -> GTResult:
    """Both genes' values plus which is higher and by how much."""
    category = "comparative_query"
    accession = _resolve_sample(sample_id)
    if accession is None:
        return _insufficient(category, sample_id, "unknown_sample_id")
    row = _row(accession)
    if row is None:
        return _insufficient(category, accession, "sample_not_in_expression_matrix")

    hit_a = _gene_value(row, str(gene_a))
    hit_b = _gene_value(row, str(gene_b))
    if hit_a is None or hit_b is None:
        missing = [
            g for g, h in ((gene_a, hit_a), (gene_b, hit_b)) if h is None
        ]
        return _insufficient(
            category, accession, f"gene_not_in_bulkformer_vocab:{','.join(missing)}"
        )

    value_a, value_b = hit_a[0], hit_b[0]
    if value_a > value_b:
        higher, lower = str(gene_a), str(gene_b)
    elif value_b > value_a:
        higher, lower = str(gene_b), str(gene_a)
    else:
        higher, lower = None, None

    return GTResult(
        category=category, sample_id=accession, status=OK,
        payload={
            "gene_a": str(gene_a),
            "gene_b": str(gene_b),
            "expression_a": round(value_a, 4),
            "expression_b": round(value_b, 4),
            "units": EXPRESSION_UNITS,
            "higher": higher,
            "lower": lower,
            "equal": higher is None,
            "difference": round(abs(value_a - value_b), 4),
            "signed_difference_a_minus_b": round(value_a - value_b, 4),
            # Values are ln(TPM + 1), so the difference divided by ln 2 is
            # log2((TPM_a + 1) / (TPM_b + 1)) — a pseudocount-1 log2 fold change.
            # The raw difference is NOT one, unlike under the old log2 matrix.
            "log2_fold_change_a_vs_b": round((value_a - value_b) / math.log(2), 4),
        },
    )


def interaction_network_query(sample_id: Any, gene: str, n: Any) -> GTResult:
    """Top N co-expressed partners of a gene, with their values in this sample.

    `n` is required and must be bound explicitly, exactly as `ranking_query`
    requires a size bound. It previously defaulted to a silent 10 while none of
    the category's templates carried an `{N}` placeholder, so the answer asserted
    a count the question never asked for. The templates now bind `{N}`, and an
    unbound `n` is refused rather than guessed.

    Partner sets come from the saved edge list
    (`qa_generation/coexpression/coexpression_edges.parquet`), computed once by
    `build_coexpression_edges.py` using Track 4's correlation methodology.
    Partner candidacy is restricted to BulkFormer's vocabulary (21,016 of the
    49,231 QC columns are eligible), so no answer names a gene the model cannot
    see. The reported fact is always each partner's real expression in this
    sample under BulkFormer's own transform, never the correlation strength.
    """
    category = "interaction_network_query"
    accession = _resolve_sample(sample_id)
    if accession is None:
        return _insufficient(category, sample_id, "unknown_sample_id")
    row = _row(accession)
    if row is None:
        return _insufficient(category, accession, "sample_not_in_expression_matrix")

    partners = _coexpression_edges().get(str(gene))
    if not partners:
        return _insufficient(category, accession, "gene_not_in_coexpression_edges")

    if (
        n is None
        or isinstance(n, bool)
        or not isinstance(n, (int, np.integer))
        or int(n) < 1
        or int(n) > len(partners)
    ):
        return _insufficient(category, accession, f"unusable_partner_count:{n!r}")
    partners = partners[: int(n)]

    self_hit = _gene_value(row, str(gene))
    records = []
    for rank, (partner, r) in enumerate(partners, start=1):
        hit = _gene_value(row, partner)
        if hit is None:  # unreachable: partners come from the same matrix
            return _insufficient(
                category, accession, f"partner_not_in_bulkformer_vocab:{partner}"
            )
        records.append(
            {
                "gene": partner,
                "rank": rank,
                "expression": round(hit[0], 4),
                "pearson_r": round(float(r), 4),
            }
        )

    return GTResult(
        category=category, sample_id=accession, status=OK,
        payload={
            "gene": str(gene),
            "gene_expression": round(self_hit[0], 4) if self_hit else None,
            "units": EXPRESSION_UNITS,
            "n_partners": len(records),
            "partners": records,
            "partner_source": "coexpression_edges.parquet (Track 4 methodology)",
            "correlation_population": "8,553 disease-confirmed CVD samples",
            "partners_vocabulary_restricted": True,
        },
    )


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


def disease_subtype_classification(sample_id: Any) -> GTResult:
    """That sample's disease-confirmed CVD subtype label.

    insufficient_data for the `disease_matched_subtype_unresolved` bucket (CVD
    confirmed but no subtype resolved) and for tissue-only-unconfirmed samples,
    which the standing extended-EDA rule forbids treating as disease-positive.
    """
    category = "disease_subtype_classification"
    accession = _resolve_sample(sample_id)
    if accession is None:
        return _insufficient(category, sample_id, "unknown_sample_id")
    rec = _sample_labels().loc[accession]
    if not bool(rec.is_cvd_disease):
        return _insufficient(category, accession, "not_disease_confirmed")

    subtype = str(rec.cvd_subtype)
    if subtype in NON_SUBTYPE_LABELS:
        return _insufficient(category, accession, f"subtype_unresolved:{subtype}")

    return GTResult(
        category=category, sample_id=accession, status=OK,
        payload={
            "subtype": subtype,
            "disease_category": str(rec.disease_category),
            "series_id": str(rec.series_id),
            "n_disease_categories_matched": int(rec.n_disease_categories_matched),
            "label_source": "extended_eda_out/labels/sample_labels.parquet",
            "in_expression_matrix": accession in _expression_rows(),
        },
    )


def comparative_differential_reasoning(
    sample_id: Any, comparison_group: str
) -> GTResult:
    """How disease-confirmed CVD separates from the `neg_hard` control pool.

    Only `neg_hard` (`tissue_only_disease_unconfirmed`) is answerable. The
    random-bulk-tissue group conflates tissue identity with disease signal, so
    any other binding returns insufficient_data rather than being silently
    swapped for a confounded one.

    The payload carries corpus-level separability, not per-gene differential
    expression: no expression matrix exists for the neg_hard pool
    (the materialized matrices cover disease-confirmed samples only), so no
    per-gene contrast has ever been computed at this comparison. The field is
    present and explicitly null so Step 4 cannot mistake absence for licence to
    invent one.
    """
    category = "comparative_differential_reasoning"
    accession = _resolve_sample(sample_id)
    if accession is None:
        return _insufficient(category, sample_id, "unknown_sample_id")
    group = str(comparison_group).strip().lower()
    if group not in VALID_COMPARISON_GROUPS:
        return _insufficient(
            category, accession, f"comparison_group_not_permitted:{comparison_group}"
        )

    probe = _probe_labels()
    if accession not in probe.index:
        return _insufficient(category, accession, "sample_not_in_probe_labels")
    rec = probe.loc[accession]
    if not bool(rec.is_positive):
        return _insufficient(category, accession, "sample_not_a_probe_positive")

    results = _probe_results()
    if "neg_hard" not in results:
        return _insufficient(category, accession, "neg_hard_probe_results_missing")
    neg_hard = results["neg_hard"]
    primary = max(neg_hard, key=lambda v: neg_hard[v]["roc_auc_mean"])
    confounded = results.get("neg_whole_corpus", {}).get(primary)

    subtype = str(rec.cvd_subtype)
    return GTResult(
        category=category, sample_id=accession, status=OK,
        payload={
            "sample_subtype": subtype if subtype not in NON_SUBTYPE_LABELS else None,
            "comparison_group": "neg_hard",
            "comparison_group_definition": (
                "tissue_only_disease_unconfirmed — CVD-relevant tissue, bulk, "
                "with no disease confirmation in the metadata"
            ),
            "n_positive": neg_hard[primary]["n_positive"],
            "n_comparison": neg_hard[primary]["n_negative"],
            "n_series": neg_hard[primary]["n_series"],
            "separability": {
                "metric": "linear probe ROC-AUC, grouped 5-fold by series",
                "primary_variant": primary,
                "roc_auc_mean": round(neg_hard[primary]["roc_auc_mean"], 4),
                "roc_auc_std": round(neg_hard[primary]["roc_auc_std"], 4),
                "by_variant": {
                    v: round(s["roc_auc_mean"], 4) for v, s in neg_hard.items()
                },
            },
            "confound_context": (
                None
                if confounded is None
                else {
                    "random_tissue_roc_auc": round(confounded["roc_auc_mean"], 4),
                    "note": (
                        "The random-bulk-tissue comparison scores higher only "
                        "because tissue identity is uncontrolled; that group is "
                        "not valid ground truth for disease-specific claims."
                    ),
                }
            ),
            "per_gene_differential": None,
            "per_gene_differential_reason": (
                "no expression matrix exists for the neg_hard pool, so no "
                "per-gene contrast has been computed at this comparison"
            ),
        },
    )


def gene_driver_reasoning(sample_id: Any, top_n: Any) -> GTResult:
    """Top N genes driving broad CVD-vs-tissue classification, for a qualifying sample.

    `top_n` is required and must be bound explicitly. It previously defaulted to
    `None`, which returned all 1,142 stable genes — while every template in this
    category asks for "the top molecular signals" or "the highest-ranking signal
    genes". A 1,142-gene list is not what that question asks for, so the bound is
    now stated by the caller and an unbound value is refused rather than
    silently meaning "all".

    Broad CVD only, and deliberately not parameterised by subtype. The
    elastic-net ranking behind this answer comes from one global
    CVD-vs-random-tissue model; no per-subtype ranking exists anywhere in the
    project (`gene_pool_prep/elastic_net_ranking_audit.md`). Every
    disease-confirmed CVD sample therefore shares the same gene list — this is a
    corpus-level fact. The function's job is to confirm the sample qualifies and
    return that shared ranking, not to compute anything per sample.

    The list is the cross-fold-stable subset only: `nonzero_frac == 1.0`, i.e.
    a nonzero coefficient in all five outer folds.
    """
    category = "gene_driver_reasoning"
    accession = _resolve_sample(sample_id)
    if accession is None:
        return _insufficient(category, sample_id, "unknown_sample_id")
    rec = _sample_labels().loc[accession]
    if not bool(rec.is_cvd_disease):
        return _insufficient(category, accession, "not_disease_confirmed")

    stable = _stable_signal_genes()
    if (
        top_n is None
        or isinstance(top_n, bool)
        or not isinstance(top_n, (int, np.integer))
        or int(top_n) < 1
        or int(top_n) > len(stable)
    ):
        return _insufficient(category, accession, f"unusable_top_n:{top_n!r}")
    stable = stable.head(int(top_n))

    return GTResult(
        category=category, sample_id=accession, status=OK,
        payload={
            "scope": "broad_cardiovascular_disease",
            "not_subtype_specific": True,
            "scope_note": (
                "Ranking is from a single global CVD-vs-random-bulk-tissue "
                "elastic-net model. It supports no subtype-specific claim."
            ),
            "n_returned": int(len(stable)),
            "n_stable_genes": int(len(_stable_signal_genes())),
            "n_stable_rows_before_symbol_dedup": _stable_counts()[
                "rows_before_symbol_dedup"
            ],
            "n_dropped_out_of_bulkformer_vocab": _stable_counts()[
                "dropped_out_of_vocab"
            ],
            "stability_criterion": "nonzero_frac == 1.0 (all 5 outer folds)",
            "vocabulary_filter": (
                "genes outside BulkFormer's 20,010-gene input vocabulary removed; "
                "a filter on the existing ranking, not an elastic-net re-run"
            ),
            "genes": [
                {
                    "gene": str(r.gene_symbol),
                    "rank": int(r.rank),
                    "mean_coef": round(float(r.mean_coef), 5),
                    "abs_mean_coef": round(float(r.abs_mean_coef), 5),
                    "direction": str(r.direction),
                    "in_clingen_hcvd": bool(r.in_clingen_hcvd),
                }
                for r in stable.itertuples()
            ],
            "ranked_by": "abs_mean_coef, descending",
            "sample_role": "eligibility_gate_only",
            "in_expression_matrix": accession in _expression_rows(),
        },
    )


#: Category name -> function, for Step 3's generic template filling.
GT_FUNCTIONS = {
    "direct_abundance_query": direct_abundance_query,
    "threshold_query": threshold_query,
    "ranking_ordering_query": ranking_query,
    "comparative_query": comparative_query,
    "interaction_network_query": interaction_network_query,
    "disease_subtype_classification": disease_subtype_classification,
    "comparative_differential_reasoning": comparative_differential_reasoning,
    "gene_driver_reasoning": gene_driver_reasoning,
}
