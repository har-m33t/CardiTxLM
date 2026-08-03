"""
build_curated_pool.py — Track 5: union the three gene-pool sources.

What this does, and what it deliberately does not
-------------------------------------------------
Unions three independently-computed selected sets:

    high-variance (Track 3)  ∪  high-centrality (Track 4)  ∪  KEGG (Track 2)

and records, per gene, which categories it qualified through plus each
source's own metric. It does not recompute, reweight or second-guess any
upstream ranking — a gene is in the variance set iff Track 3 marked it
`selected`, full stop.

Three sources, not four. ClinGen HCVD curation was dropped with Track 2b;
`elasticnet/clingen.py` stays a post-fit validation cross-check, and
promoting it to a pool input would make that check circular. See the Track 5
open question in `.claude/gene_pool_prerequisites_todo.md`.

Deduplication on symbol
-----------------------
The QC universe is 49,231 *columns* but only 46,540 distinct symbols: 1,534
symbols map to more than one column (up to 15 each), many byte-identical.
Track 4's README calls this out and tells Track 5 to deduplicate on symbol.

The pool is keyed by symbol because that is what the QA-generation pipeline
consumes — it asks questions about "MYH7", not about a particular matrix
column. A symbol therefore enters a set if *any* of its columns cleared that
track's threshold, and carries that column's rank (the best, i.e. lowest, of
its columns). This matters: 198 symbols have columns that disagree on
variance selection, 63 on centrality. Taking the max rank instead would
silently drop genes an upstream track did select.

The KEGG source has no `selected` column by construction — every row in it
is already a surviving pathway member (114 of 114 intersected against the
universe), so membership in the file *is* selection.

Identifier consistency
----------------------
All three sources are keyed on the same ARCHS4 uppercase spelling from
`gene_symbols.npy`. Track 2 resolved KEGG's HGNC casing onto the universe's
own spelling at build time, so no case-folding is needed or done here. The
universe carries 15,304 bare `ENSG…` identifiers for genes with no assigned
symbol; that is one universe with a documented fallback, present identically
in every source, not a cross-file format mismatch. `_check_identifiers`
hard-fails on an actual mismatch rather than guessing a reconciliation.

Output
------
curated_gene_pool.csv — one row per pooled symbol:
    gene, in_variance_set, in_centrality_set, in_kegg_set, n_sources,
    variance, variance_rank, in_degree, centrality_rank, kegg_pathway
union_summary.md — pool size, per-category and per-overlap counts, and the
Task 6 sanity-check results.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

VARIANCE_FILE = "high_variance_genes.csv"
CENTRALITY_FILE = "high_centrality_genes.csv"
KEGG_FILE = "kegg_cardiomyopathy_genes.csv"

SUMMARY_FILE = "union_summary.md"
DEFAULT_POOL_FILE = "curated_gene_pool.csv"
UNFILTERED_POOL_FILE = "curated_gene_pool_unfiltered.csv"

# ARCHS4 falls back to the bare Ensembl gene ID when a gene has no assigned
# HGNC symbol. Matches ENSG00000227948 but never a real symbol.
UNANNOTATED_PATTERN = r"ENSG\d+"

CATEGORIES = ("variance", "centrality", "kegg")


class InputMissingError(RuntimeError):
    """A required upstream track has not produced its deliverable."""


class IdentifierMismatchError(RuntimeError):
    """Gene identifier conventions disagree across the three sources."""


class SanityCheckError(RuntimeError):
    """The union failed a check that means the result must not be saved."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _require(path: Path, track: str, produced_by: str) -> Path:
    """Fail with the *track* that needs to run, not just a missing path."""
    if not path.is_file():
        raise InputMissingError(
            f"{path.name} not found at {path} — {track} has not completed. "
            f"Run {produced_by} first; this union requires all three sources "
            f"and will not substitute a placeholder for a missing one."
        )
    return path


def load_inputs(indir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three sources, failing loudly on a missing or malformed one."""
    variance_path = _require(
        indir / VARIANCE_FILE, "Track 3 (variance)", "compute_variance_genes.py"
    )
    centrality_path = _require(
        indir / CENTRALITY_FILE, "Track 4 (centrality)", "compute_centrality_genes.py"
    )
    kegg_path = _require(
        indir / KEGG_FILE, "Track 2 (KEGG)", "build_kegg_cardiomyopathy.py"
    )

    variance = pd.read_csv(variance_path)
    centrality = pd.read_csv(centrality_path)
    # Track 2 writes a '#'-prefixed provenance preamble above the header.
    kegg = pd.read_csv(kegg_path, comment="#")

    _check_columns(variance, VARIANCE_FILE, ["gene", "variance", "variance_rank", "selected"])
    _check_columns(
        centrality, CENTRALITY_FILE, ["gene", "in_degree", "centrality_rank", "selected"]
    )
    _check_columns(kegg, KEGG_FILE, ["gene", "source_pathway"])

    for frame, name in ((variance, VARIANCE_FILE), (centrality, CENTRALITY_FILE)):
        if frame["selected"].dtype != bool:
            raise IdentifierMismatchError(
                f"{name}: 'selected' is {frame['selected'].dtype}, expected bool — "
                f"refusing to guess a truthiness rule for the selected set."
            )
        if not frame["selected"].any():
            raise InputMissingError(
                f"{name}: no genes marked selected — the upstream track "
                f"produced a ranking but never applied its threshold."
            )

    logger.info(
        "loaded: variance %d rows (%d selected), centrality %d rows (%d selected), "
        "KEGG %d pathway members",
        len(variance),
        int(variance["selected"].sum()),
        len(centrality),
        int(centrality["selected"].sum()),
        len(kegg),
    )
    return variance, centrality, kegg


def _check_columns(frame: pd.DataFrame, name: str, required: list[str]) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise InputMissingError(
            f"{name} is missing required column(s) {missing}; has {list(frame.columns)}"
        )


# --------------------------------------------------------------------------
# Identifier consistency
# --------------------------------------------------------------------------


def _classify(symbols: pd.Series) -> dict:
    text = symbols.astype(str)
    return {
        "n": len(text),
        "n_unique": int(text.nunique()),
        "n_ensembl": int(text.str.fullmatch(r"ENSG\d+(\.\d+)?").sum()),
        "n_numeric": int(text.str.fullmatch(r"\d+").sum()),
        "n_not_uppercase": int((text != text.str.upper()).sum()),
        "n_untrimmed": int((text != text.str.strip()).sum()),
        "n_empty": int((text.str.strip() == "").sum()),
    }


def check_identifiers(
    variance: pd.DataFrame, centrality: pd.DataFrame, kegg: pd.DataFrame
) -> dict:
    """Confirm all three sources speak the same identifier convention.

    Stops on a genuine mismatch (one file on Ensembl IDs, another on
    symbols; stray casing or whitespace that would break exact matching)
    rather than attempting a reconciliation.
    """
    profiles = {
        "variance": _classify(variance["gene"]),
        "centrality": _classify(centrality["gene"]),
        "kegg": _classify(kegg["gene"]),
    }

    problems: list[str] = []
    for name, prof in profiles.items():
        if prof["n_numeric"]:
            problems.append(f"{name}: {prof['n_numeric']} bare numeric (Entrez-style) IDs")
        if prof["n_not_uppercase"]:
            problems.append(
                f"{name}: {prof['n_not_uppercase']} symbols not in the universe's "
                f"uppercase convention"
            )
        if prof["n_untrimmed"]:
            problems.append(f"{name}: {prof['n_untrimmed']} symbols with surrounding whitespace")
        if prof["n_empty"]:
            problems.append(f"{name}: {prof['n_empty']} empty symbols")

    # Tracks 3 and 4 rank the same universe; they must agree column-for-column.
    variance_counts = variance["gene"].value_counts()
    centrality_counts = centrality["gene"].value_counts()
    same_universe = variance_counts.equals(centrality_counts.reindex(variance_counts.index))
    if not same_universe:
        only_variance = sorted(set(variance["gene"]) - set(centrality["gene"]))[:5]
        only_centrality = sorted(set(centrality["gene"]) - set(variance["gene"]))[:5]
        problems.append(
            f"variance and centrality do not cover the same universe "
            f"(variance-only e.g. {only_variance}, centrality-only e.g. {only_centrality})"
        )

    # KEGG was intersected against this universe upstream; anything outside it
    # means the two were built against different universe snapshots.
    kegg_orphans = sorted(set(kegg["gene"]) - set(variance["gene"]))
    if kegg_orphans:
        problems.append(
            f"{len(kegg_orphans)} KEGG genes absent from the ranked universe "
            f"(e.g. {kegg_orphans[:5]}) — sources built against different universes"
        )

    if problems:
        raise IdentifierMismatchError(
            "Gene identifier formats are inconsistent across the three sources; "
            "stopping rather than reconciling them here:\n  - "
            + "\n  - ".join(problems)
        )

    logger.info(
        "identifiers consistent: shared universe of %d columns / %d distinct symbols "
        "(%d bare ENSG fallbacks); KEGG's %d members all resolve into it",
        profiles["variance"]["n"],
        profiles["variance"]["n_unique"],
        profiles["variance"]["n_ensembl"],
        profiles["kegg"]["n"],
    )
    return profiles


# --------------------------------------------------------------------------
# Per-source selected sets, deduplicated onto symbols
# --------------------------------------------------------------------------


def _best_ranked_selected(
    frame: pd.DataFrame, rank_column: str, value_column: str
) -> pd.DataFrame:
    """Collapse a ranking to one row per *selected* symbol, keeping its best rank.

    Best = lowest rank number. A symbol whose columns disagree on `selected`
    still enters the set: the upstream track did select one of its columns,
    and the pool is keyed by symbol.
    """
    selected = frame.loc[frame["selected"], ["gene", value_column, rank_column]]
    best = selected.sort_values(rank_column, kind="mergesort").drop_duplicates("gene")
    return best.set_index("gene")


def _kegg_pathways(kegg: pd.DataFrame) -> pd.Series:
    """One pathway label per symbol, merging duplicates as 'both'."""

    def merge(labels: pd.Series) -> str:
        distinct = sorted(set(labels))
        if len(distinct) == 1:
            return distinct[0]
        # A symbol listed under each pathway separately is in both.
        return "both" if set(distinct) <= {"hsa05410", "hsa05414"} else "|".join(distinct)

    return kegg.groupby("gene")["source_pathway"].apply(merge)


def build_pool(
    variance: pd.DataFrame, centrality: pd.DataFrame, kegg: pd.DataFrame
) -> pd.DataFrame:
    """Union the three selected sets and attach provenance plus source metrics."""
    variance_sel = _best_ranked_selected(variance, "variance_rank", "variance")
    centrality_sel = _best_ranked_selected(centrality, "centrality_rank", "in_degree")
    kegg_sel = _kegg_pathways(kegg)

    genes = sorted(set(variance_sel.index) | set(centrality_sel.index) | set(kegg_sel.index))
    pool = pd.DataFrame(index=pd.Index(genes, name="gene"))

    pool["in_variance_set"] = pool.index.isin(variance_sel.index)
    pool["in_centrality_set"] = pool.index.isin(centrality_sel.index)
    pool["in_kegg_set"] = pool.index.isin(kegg_sel.index)
    pool["n_sources"] = (
        pool[["in_variance_set", "in_centrality_set", "in_kegg_set"]].sum(axis=1).astype(int)
    )

    pool["variance"] = variance_sel["variance"]
    pool["variance_rank"] = variance_sel["variance_rank"].astype("Int64")
    pool["in_degree"] = centrality_sel["in_degree"].astype("Int64")
    pool["centrality_rank"] = centrality_sel["centrality_rank"].astype("Int64")
    pool["kegg_pathway"] = kegg_sel

    return pool.reset_index()


# --------------------------------------------------------------------------
# Summary statistics and sanity checks
# --------------------------------------------------------------------------


def filter_unannotated(pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop pooled genes that carry a bare Ensembl ID instead of a symbol.

    This is a curation step applied *after* the union, deliberately kept
    separate from it: the union is the honest three-source result and is
    preserved on disk unfiltered, while this trims what the QA-generation
    pipeline can actually ask a meaningful question about. A question about
    `ENSG00000227948` has no interpretable answer and no way to be graded.

    It is not a quality judgement on the upstream tracks and not a response
    to the confounder screen — the screen found the centrality arm no more
    depth- or batch-confounded than the variance arm (see
    `confounder_screen/FINDINGS.md`). These genes are dropped because they
    are unnameable, not because they are suspect.

    Scope: 1,452 of 8,664 genes, none of them KEGG members. 1,432 qualified
    through a single source. The 20 that cleared two thresholds and the one
    at variance rank 59 are the real cost of the rule, accepted because a
    question about an unnamed locus is unusable regardless of its rank.
    """
    is_unannotated = pool["gene"].str.fullmatch(UNANNOTATED_PATTERN)
    dropped = pool[is_unannotated]

    stats = {
        "n_dropped": int(is_unannotated.sum()),
        "n_kept": int((~is_unannotated).sum()),
        "dropped_variance_only": int(
            (dropped["in_variance_set"] & ~dropped["in_centrality_set"]).sum()
        ),
        "dropped_centrality_only": int(
            (~dropped["in_variance_set"] & dropped["in_centrality_set"]).sum()
        ),
        "dropped_multi_source": int((dropped["n_sources"] >= 2).sum()),
        "dropped_kegg": int(dropped["in_kegg_set"].sum()),
        "best_variance_rank_dropped": (
            None if dropped["variance_rank"].isna().all() else int(dropped["variance_rank"].min())
        ),
    }
    if stats["dropped_kegg"]:
        raise SanityCheckError(
            f"{stats['dropped_kegg']} KEGG pathway members matched the unannotated "
            f"pattern — every KEGG member has a symbol by construction, so this "
            f"means the filter or the KEGG source is wrong."
        )

    logger.info(
        "unannotated filter: dropped %d bare-ENSG genes (%d single-source, "
        "%d multi-source, 0 KEGG), %d remain",
        stats["n_dropped"],
        stats["n_dropped"] - stats["dropped_multi_source"],
        stats["dropped_multi_source"],
        stats["n_kept"],
    )
    return pool[~is_unannotated].reset_index(drop=True), stats


def summarize(pool: pd.DataFrame, source_sizes: dict[str, int]) -> dict:
    """Per-category, pairwise and three-way counts, plus the Task 6 checks."""
    in_variance = pool["in_variance_set"]
    in_centrality = pool["in_centrality_set"]
    in_kegg = pool["in_kegg_set"]

    total = len(pool)
    naive_sum = sum(source_sizes.values())
    largest_source = max(source_sizes.values())

    stats = {
        "total_pool_size": total,
        "source_selected_counts": dict(source_sizes),
        "naive_sum": naive_sum,
        "largest_single_category": largest_source,
        # Genes qualifying through exactly one category.
        "unique_to_variance": int((in_variance & ~in_centrality & ~in_kegg).sum()),
        "unique_to_centrality": int((~in_variance & in_centrality & ~in_kegg).sum()),
        "unique_to_kegg": int((~in_variance & ~in_centrality & in_kegg).sum()),
        # Pairwise overlaps, counted inclusively (a three-way gene is in all
        # three pairs) — the standard reading of "overlap between A and B".
        "overlap_variance_centrality": int((in_variance & in_centrality).sum()),
        "overlap_variance_kegg": int((in_variance & in_kegg).sum()),
        "overlap_centrality_kegg": int((in_centrality & in_kegg).sum()),
        # Exactly-two regions, so the seven Venn regions sum to the total.
        "exactly_variance_and_centrality": int((in_variance & in_centrality & ~in_kegg).sum()),
        "exactly_variance_and_kegg": int((in_variance & ~in_centrality & in_kegg).sum()),
        "exactly_centrality_and_kegg": int((~in_variance & in_centrality & in_kegg).sum()),
        "all_three": int((in_variance & in_centrality & in_kegg).sum()),
        "n_sources_1": int((pool["n_sources"] == 1).sum()),
        "n_sources_2": int((pool["n_sources"] == 2).sum()),
        "n_sources_3": int((pool["n_sources"] == 3).sum()),
    }

    stats["overlap_reduction"] = naive_sum - total
    stats["checks"] = {
        "smaller_than_naive_sum": total < naive_sum,
        "larger_than_largest_category": total > largest_source,
        "venn_regions_sum_to_total": (
            stats["unique_to_variance"]
            + stats["unique_to_centrality"]
            + stats["unique_to_kegg"]
            + stats["exactly_variance_and_centrality"]
            + stats["exactly_variance_and_kegg"]
            + stats["exactly_centrality_and_kegg"]
            + stats["all_three"]
        )
        == total,
        "no_duplicate_genes": int(pool["gene"].duplicated().sum()) == 0,
        "every_gene_has_a_source": bool((pool["n_sources"] >= 1).all()),
    }
    stats["passed"] = all(stats["checks"].values())
    return stats


def assert_sane(stats: dict) -> None:
    """Refuse to save a pool that fails a structural check."""
    if stats["passed"]:
        return
    failed = [name for name, ok in stats["checks"].items() if not ok]
    raise SanityCheckError(
        "Union failed sanity check(s) "
        + ", ".join(failed)
        + f" — pool size {stats['total_pool_size']} against naive sum "
        + f"{stats['naive_sum']} and largest category "
        + f"{stats['largest_single_category']}. Not saving; the union logic "
        + "or an input is wrong."
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _check_line(ok: bool, text: str) -> str:
    return f"- {'✅' if ok else '❌'} {text}"


def _render_filter_section(union_stats: dict, filter_stats: dict | None) -> str:
    if filter_stats is None:
        return (
            "## Unannotated-gene filter\n\n"
            "Not applied on this run (`--no-drop-unannotated`). The pool is the "
            "raw three-source union.\n"
        )
    union_total = union_stats["total_pool_size"]
    kept = filter_stats["n_kept"]
    return f"""## Unannotated-gene filter

The raw union is **{union_total:,} genes**; this file describes the
**{kept:,}-gene** pool that remains after dropping genes whose only
identifier is a bare Ensembl ID (`ENSG…`, ARCHS4's fallback where no HGNC
symbol exists). The unfiltered union is preserved at
`gene_pool_prep/{UNFILTERED_POOL_FILE}`.

**Rationale.** The QA-generation pipeline asks questions about named genes. A
question about `ENSG00000227948` has no interpretable answer and no way to be
graded, so at ~29 questions per gene these {filter_stats['n_dropped']:,} genes
would have absorbed roughly {filter_stats['n_dropped'] * 29:,} of the 250K
question budget for no usable training signal.

This is a naming-based curation step, **not** a quality judgement on any
track and not a consequence of the confounder screen — that screen found the
centrality arm no more depth- or batch-confounded than the variance arm (see
`confounder_screen/FINDINGS.md`). These genes are dropped because they are
unnameable, not because they are suspect.

| Dropped | Genes |
|---|---:|
| Centrality-only | {filter_stats['dropped_centrality_only']:,} |
| Variance-only | {filter_stats['dropped_variance_only']:,} |
| Qualified through ≥2 sources | {filter_stats['dropped_multi_source']:,} |
| KEGG members | {filter_stats['dropped_kegg']} |
| **Total** | **{filter_stats['n_dropped']:,}** |

No KEGG member is affected — every pathway member has a symbol by
construction, and the filter hard-fails if one ever matches. The real cost of
the rule is the {filter_stats['dropped_multi_source']} genes that cleared two
independent thresholds and the best-ranked casualty at variance rank
{filter_stats['best_variance_rank_dropped']}; accepted because an unnamed
locus is unusable as a question subject regardless of its rank.

Note this filter targets *only* bare Ensembl IDs. Named-but-uncharacterised
features (`LINC…`, `…-AS1`, `…-DT`) are kept — they have stable symbols and
literature, so a question about them is at least answerable.

"""


def render_summary(
    stats: dict,
    profiles: dict,
    pool: pd.DataFrame,
    union_stats: dict | None = None,
    filter_stats: dict | None = None,
) -> str:
    sizes = stats["source_selected_counts"]
    checks = stats["checks"]
    total = stats["total_pool_size"]
    union_stats = union_stats or stats

    top_all_three = (
        pool.loc[pool["n_sources"] == 3]
        .sort_values("variance_rank")["gene"]
        .head(12)
        .tolist()
    )

    return f"""# Track 5 — curated gene pool: union summary

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")} by
`gene_pool_prep/build_curated_pool.py`.

Three sources, unioned on gene symbol: high-variance (Track 3) ∪
high-centrality (Track 4) ∪ KEGG cardiomyopathy membership (Track 2).
ClinGen is deliberately not a fourth category — it remains a post-fit
validation cross-check, and promoting it here would make that check
circular.

## Final pool size

**{total:,} genes** — the three-source union of
{union_stats['total_pool_size']:,}, minus the unannotated-gene filter below.
This is the file the QA-generation pipeline consumes.

{_render_filter_section(union_stats, filter_stats)}## Input selected sets

Counts are distinct *symbols*, after deduplicating the universe's
multi-column symbols. Tracks 3 and 4 each select 4,924 columns; those
collapse to fewer symbols because 1,534 symbols map to more than one column.
The final column is what survives into this pool after the filter.

| Source | Selected columns | Selected symbols | In final pool |
|---|---:|---:|---:|
| Variance (Track 3) | 4,924 | {union_stats['source_selected_counts']['variance']:,} | {sizes['variance']:,} |
| Centrality (Track 4) | 4,924 | {union_stats['source_selected_counts']['centrality']:,} | {sizes['centrality']:,} |
| KEGG (Track 2) | {sizes['kegg']:,} | {sizes['kegg']:,} | {sizes['kegg']:,} |
| **Naive sum** | | **{union_stats['naive_sum']:,}** | **{stats['naive_sum']:,}** |

## Contribution by category

All counts below describe the **final {total:,}-gene pool**.

Exactly-one-category genes — what each source uniquely brings to the pool:

| Qualifying through only… | Genes |
|---|---:|
| Variance only | {stats['unique_to_variance']:,} |
| Centrality only | {stats['unique_to_centrality']:,} |
| KEGG only | {stats['unique_to_kegg']:,} |

## Overlaps

Pairwise, counted inclusively (a gene in all three appears in all three rows):

| Pair | Genes |
|---|---:|
| Variance ∩ Centrality | {stats['overlap_variance_centrality']:,} |
| Variance ∩ KEGG | {stats['overlap_variance_kegg']:,} |
| Centrality ∩ KEGG | {stats['overlap_centrality_kegg']:,} |
| **All three** | **{stats['all_three']:,}** |

The seven disjoint Venn regions, which sum to the pool total:

| Region | Genes |
|---|---:|
| Variance only | {stats['unique_to_variance']:,} |
| Centrality only | {stats['unique_to_centrality']:,} |
| KEGG only | {stats['unique_to_kegg']:,} |
| Variance + Centrality, not KEGG | {stats['exactly_variance_and_centrality']:,} |
| Variance + KEGG, not Centrality | {stats['exactly_variance_and_kegg']:,} |
| Centrality + KEGG, not Variance | {stats['exactly_centrality_and_kegg']:,} |
| All three | {stats['all_three']:,} |
| **Total** | **{total:,}** |

By number of qualifying sources: {stats['n_sources_1']:,} genes through one
source, {stats['n_sources_2']:,} through two, {stats['n_sources_3']:,}
through all three. Overlap removed {stats['overlap_reduction']:,} genes
relative to the naive sum.

Genes qualifying through all three sources, by variance rank:
{', '.join(top_all_three)}{'…' if stats['all_three'] > len(top_all_three) else ''}

## Sanity checks (Task 6)

These validate the **union logic**, so they run on the unfiltered
{union_stats['total_pool_size']:,}-gene union — a downstream curation filter
must not be able to mask a broken join.

{_check_line(union_stats['checks']['smaller_than_naive_sum'],
             f"Union ({union_stats['total_pool_size']:,}) < sum of the three selected sets "
             f"({union_stats['naive_sum']:,}) — overlap reduces the union below the naive sum")}
{_check_line(union_stats['checks']['larger_than_largest_category'],
             f"Union ({union_stats['total_pool_size']:,}) > largest single category "
             f"({union_stats['largest_single_category']:,}) — a union can only grow or match")}
{_check_line(union_stats['checks']['venn_regions_sum_to_total'],
             "The seven disjoint Venn regions sum to the union total")}
{_check_line(union_stats['checks']['no_duplicate_genes'], "No duplicate gene symbols")}
{_check_line(union_stats['checks']['every_gene_has_a_source'],
             "Every pooled gene carries at least one source category")}

Re-run on the filtered {total:,}-gene pool, the same five checks still hold
({'passed' if stats['passed'] else 'FAILED'}): {total:,} < {stats['naive_sum']:,}
naive sum, {total:,} > {stats['largest_single_category']:,} largest category,
regions sum, no duplicates, every gene sourced.

**Result: {'all checks passed' if stats['passed'] and union_stats['passed']
           else 'FAILED — pool not saved'}.**

## Identifier consistency

All three sources key on the same ARCHS4 uppercase spelling from
`gene_symbols.npy`. Verified before unioning:

- Tracks 3 and 4 cover the identical universe — {profiles['variance']['n']:,}
  columns over {profiles['variance']['n_unique']:,} distinct symbols.
- All {profiles['kegg']['n']} KEGG members resolve into that universe (Track 2
  already case-normalised KEGG's HGNC spelling onto the universe's own).
- No Ensembl-vs-symbol mismatch across files, no bare numeric IDs, no stray
  casing or whitespace in any source.

{profiles['variance']['n_ensembl']:,} universe entries are bare `ENSG…`
identifiers — ARCHS4's fallback for genes with no assigned symbol. That is
one universe with a documented fallback, shared identically by all sources,
not a cross-file format mismatch. KEGG contributes none of them, since every
KEGG member has a symbol.

## Deduplication note

The pool is keyed by symbol, not by matrix column, because the
QA-generation pipeline asks questions about genes by name. A symbol enters a
set if **any** of its columns cleared that track's threshold, and carries
that column's best (lowest) rank — 198 symbols have columns that disagree on
variance selection and 63 on centrality, and taking the worst rank instead
would silently drop genes the upstream track did select. This follows the
explicit instruction in `gene_pool_prep/README.md` ("Track 5 should
deduplicate on symbol").

## Output

`curated_gene_pool.csv` — one row per pooled gene:
`gene`, `in_variance_set`, `in_centrality_set`, `in_kegg_set`, `n_sources`,
`variance`, `variance_rank`, `in_degree`, `centrality_rank`, `kegg_pathway`.

Metric columns are blank where the gene did not qualify through that source.
This file is the input to the QA-generation pipeline's Step 1.
"""


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run(indir: Path, pool_path: Path, drop_unannotated: bool = True) -> pd.DataFrame:
    variance, centrality, kegg = load_inputs(indir)
    profiles = check_identifiers(variance, centrality, kegg)

    union = build_pool(variance, centrality, kegg)
    source_sizes = {
        "variance": int(variance.loc[variance["selected"], "gene"].nunique()),
        "centrality": int(centrality.loc[centrality["selected"], "gene"].nunique()),
        "kegg": int(kegg["gene"].nunique()),
    }
    # The Task 6 checks validate the *union logic*, so they run on the
    # unfiltered union — the curation filter is a separate downstream step
    # and must not be able to mask a broken join.
    union_stats = summarize(union, source_sizes)
    assert_sane(union_stats)

    indir.mkdir(parents=True, exist_ok=True)
    unfiltered_path = indir / UNFILTERED_POOL_FILE
    union.to_csv(unfiltered_path, index=False)

    if drop_unannotated:
        pool, filter_stats = filter_unannotated(union)
    else:
        pool, filter_stats = union, None

    final_sizes = {
        "variance": int(pool["in_variance_set"].sum()),
        "centrality": int(pool["in_centrality_set"].sum()),
        "kegg": int(pool["in_kegg_set"].sum()),
    }
    final_stats = summarize(pool, final_sizes)
    assert_sane(final_stats)

    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(pool_path, index=False)
    summary_path = indir / SUMMARY_FILE
    summary_path.write_text(
        render_summary(final_stats, profiles, pool, union_stats, filter_stats)
    )

    logger.info("union: %d genes -> %s", len(union), unfiltered_path)
    logger.info("final pool: %d genes -> %s", len(pool), pool_path)
    logger.info("summary -> %s", summary_path)
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--indir",
        type=Path,
        default=Path("gene_pool_prep"),
        help="directory holding the three source CSVs (default: gene_pool_prep)",
    )
    parser.add_argument(
        "--pool-out",
        type=Path,
        default=Path(DEFAULT_POOL_FILE),
        help="path for curated_gene_pool.csv (default: repo-root curated_gene_pool.csv)",
    )
    parser.add_argument(
        "--no-drop-unannotated",
        action="store_false",
        dest="drop_unannotated",
        help="keep bare-ENSG genes (the raw union); the unfiltered union is "
        "written to gene_pool_prep/ either way",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.indir, args.pool_out, drop_unannotated=args.drop_unannotated)


if __name__ == "__main__":
    main()
