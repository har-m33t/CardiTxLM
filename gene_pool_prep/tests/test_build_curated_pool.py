"""Tests for Track 5's three-source union.

The guardrails that matter here: refuse to run on a missing source, refuse
to run on mismatched identifier conventions, and refuse to save a pool that
fails the size sanity checks. Plus the symbol-deduplication rule, which is
the one place this step can silently lose genes an upstream track selected.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gene_pool_prep import build_curated_pool as bcp


def _variance_frame(rows):
    return pd.DataFrame(rows, columns=["gene", "variance", "variance_rank", "selected"])


def _centrality_frame(rows):
    return pd.DataFrame(rows, columns=["gene", "in_degree", "centrality_rank", "selected"])


def _kegg_frame(rows):
    return pd.DataFrame(rows, columns=["gene", "kegg_symbol", "source_pathway", "matched_via"])


@pytest.fixture
def sources():
    """A toy universe of 6 symbols with a known, hand-checkable union.

    variance:   A, B, C          centrality: B, C, D        kegg: C, E
    union:      A, B, C, D, E    (5 genes; F is in neither)
    """
    variance = _variance_frame(
        [
            ("A", 9.0, 1, True),
            ("B", 8.0, 2, True),
            ("C", 7.0, 3, True),
            ("D", 6.0, 4, False),
            ("E", 5.0, 5, False),
            ("F", 4.0, 6, False),
        ]
    )
    centrality = _centrality_frame(
        [
            ("B", 300, 1, True),
            ("C", 200, 2, True),
            ("D", 100, 3, True),
            ("A", 50, 4, False),
            ("E", 40, 5, False),
            ("F", 30, 6, False),
        ]
    )
    kegg = _kegg_frame(
        [
            ("C", "C", "hsa05410", "primary_symbol"),
            ("E", "E", "hsa05414", "primary_symbol"),
        ]
    )
    return variance, centrality, kegg


def _write_sources(tmp_path, variance, centrality, kegg):
    variance.to_csv(tmp_path / bcp.VARIANCE_FILE, index=False)
    centrality.to_csv(tmp_path / bcp.CENTRALITY_FILE, index=False)
    with open(tmp_path / bcp.KEGG_FILE, "w") as handle:
        handle.write("# KEGG cardiomyopathy pathway genes\n")  # Track 2's preamble
        kegg.to_csv(handle, index=False)
    return tmp_path


# --------------------------------------------------------------------------
# Union correctness
# --------------------------------------------------------------------------


def test_union_membership_and_provenance(sources):
    pool = bcp.build_pool(*sources)

    assert list(pool["gene"]) == ["A", "B", "C", "D", "E"]
    assert "F" not in set(pool["gene"])  # selected by nothing

    by_gene = pool.set_index("gene")
    assert by_gene.loc["C", "n_sources"] == 3
    assert by_gene.loc["B", ["in_variance_set", "in_centrality_set", "in_kegg_set"]].tolist() == [
        True,
        True,
        False,
    ]
    assert by_gene.loc["E", ["in_variance_set", "in_centrality_set", "in_kegg_set"]].tolist() == [
        False,
        False,
        True,
    ]


def test_metrics_carried_forward_only_where_qualified(sources):
    by_gene = bcp.build_pool(*sources).set_index("gene")

    assert by_gene.loc["A", "variance_rank"] == 1
    assert pd.isna(by_gene.loc["A", "centrality_rank"])  # A did not clear centrality
    assert by_gene.loc["D", "in_degree"] == 100
    assert pd.isna(by_gene.loc["D", "variance_rank"])
    assert by_gene.loc["C", "kegg_pathway"] == "hsa05410"
    assert pd.isna(by_gene.loc["B", "kegg_pathway"])


def test_a_gene_enters_the_set_if_any_of_its_columns_was_selected(sources):
    """The real universe has 1,534 multi-column symbols, 198 of which
    disagree on variance selection. The selected column must win."""
    variance, centrality, kegg = sources
    variance = pd.concat(
        [variance, _variance_frame([("G", 3.0, 7, False), ("G", 20.0, 0, True)])],
        ignore_index=True,
    )
    centrality = pd.concat(
        [centrality, _centrality_frame([("G", 10, 7, False), ("G", 10, 8, False)])],
        ignore_index=True,
    )

    by_gene = bcp.build_pool(variance, centrality, kegg).set_index("gene")
    assert by_gene.loc["G", "in_variance_set"]
    assert by_gene.loc["G", "variance_rank"] == 0  # the best rank, not the worst
    assert (by_gene.index.value_counts() == 1).all()  # one row per symbol


def test_duplicate_kegg_pathways_merge_to_both(sources):
    variance, centrality, kegg = sources
    kegg = pd.concat(
        [kegg, _kegg_frame([("C", "C", "hsa05414", "primary_symbol")])], ignore_index=True
    )
    by_gene = bcp.build_pool(variance, centrality, kegg).set_index("gene")
    assert by_gene.loc["C", "kegg_pathway"] == "both"


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------


def test_summary_counts_match_the_hand_checked_toy_union(sources):
    pool = bcp.build_pool(*sources)
    stats = bcp.summarize(pool, {"variance": 3, "centrality": 3, "kegg": 2})

    assert stats["total_pool_size"] == 5
    assert stats["unique_to_variance"] == 1  # A
    assert stats["unique_to_centrality"] == 1  # D
    assert stats["unique_to_kegg"] == 1  # E
    assert stats["overlap_variance_centrality"] == 2  # B, C
    assert stats["overlap_variance_kegg"] == 1  # C
    assert stats["all_three"] == 1  # C
    assert stats["checks"]["venn_regions_sum_to_total"]
    assert stats["passed"]


# --------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing,expected_track",
    [(bcp.VARIANCE_FILE, "Track 3"), (bcp.CENTRALITY_FILE, "Track 4"), (bcp.KEGG_FILE, "Track 2")],
)
def test_missing_source_names_the_track_that_must_run_first(
    tmp_path, sources, missing, expected_track
):
    _write_sources(tmp_path, *sources)
    (tmp_path / missing).unlink()

    with pytest.raises(bcp.InputMissingError, match=expected_track):
        bcp.load_inputs(tmp_path)


def test_ensembl_ids_in_one_source_only_is_rejected(sources):
    """The exact silent-failure mode Track 5 is told to stop on."""
    variance, centrality, kegg = sources
    centrality["gene"] = ["ENSG0000000000" + str(i) for i in range(len(centrality))]

    with pytest.raises(bcp.IdentifierMismatchError, match="same universe"):
        bcp.check_identifiers(variance, centrality, kegg)


def test_kegg_genes_outside_the_ranked_universe_are_rejected(sources):
    variance, centrality, kegg = sources
    kegg.loc[len(kegg)] = ("ZZZ_NOT_IN_UNIVERSE", "Zzz", "hsa05410", "primary_symbol")

    with pytest.raises(bcp.IdentifierMismatchError, match="absent from the ranked universe"):
        bcp.check_identifiers(variance, centrality, kegg)


def test_lowercase_symbols_are_rejected_rather_than_case_folded(sources):
    variance, centrality, kegg = sources
    kegg.loc[0, "gene"] = "c"

    with pytest.raises(bcp.IdentifierMismatchError, match="uppercase convention"):
        bcp.check_identifiers(variance, centrality, kegg)


def test_consistent_sources_pass_identifier_check(sources):
    profiles = bcp.check_identifiers(*sources)
    assert profiles["variance"]["n_unique"] == 6
    assert profiles["kegg"]["n"] == 2


def test_pool_not_larger_than_naive_sum_check_fires():
    """A union equal to the naive sum means no overlap was detected — a
    symptom of a broken join, so it must not be saved."""
    pool = pd.DataFrame(
        {
            "gene": ["A", "B"],
            "in_variance_set": [True, False],
            "in_centrality_set": [False, True],
            "in_kegg_set": [False, False],
            "n_sources": [1, 1],
        }
    )
    stats = bcp.summarize(pool, {"variance": 1, "centrality": 1, "kegg": 0})

    assert not stats["checks"]["smaller_than_naive_sum"]
    with pytest.raises(bcp.SanityCheckError, match="smaller_than_naive_sum"):
        bcp.assert_sane(stats)


def test_unthresholded_input_is_rejected(tmp_path, sources):
    variance, centrality, kegg = sources
    variance["selected"] = False
    _write_sources(tmp_path, variance, centrality, kegg)

    with pytest.raises(bcp.InputMissingError, match="never applied its threshold"):
        bcp.load_inputs(tmp_path)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Unannotated-gene filter
# --------------------------------------------------------------------------


def _pool_with_ensg():
    return pd.DataFrame(
        {
            "gene": ["MYH7", "ENSG00000227948", "LINC01226", "TTN-AS1", "ENSG00000012048"],
            "in_variance_set": [True, False, False, True, True],
            "in_centrality_set": [True, True, True, False, True],
            "in_kegg_set": [False, False, False, False, False],
            "n_sources": [2, 1, 1, 1, 2],
            "variance_rank": [1, None, None, 40, 59],
        }
    )


def test_filter_drops_only_bare_ensg_ids():
    kept, stats = bcp.filter_unannotated(_pool_with_ensg())

    assert list(kept["gene"]) == ["MYH7", "LINC01226", "TTN-AS1"]
    assert stats["n_dropped"] == 2
    assert stats["n_kept"] == 3
    assert stats["dropped_multi_source"] == 1
    assert stats["best_variance_rank_dropped"] == 59


def test_filter_keeps_named_but_uncharacterised_features():
    """LINC…/…-AS1 have stable symbols and literature — a question about them
    is answerable, unlike a bare locus ID."""
    kept, _ = bcp.filter_unannotated(_pool_with_ensg())
    assert {"LINC01226", "TTN-AS1"} <= set(kept["gene"])


def test_filter_refuses_to_drop_a_kegg_member():
    pool = _pool_with_ensg()
    pool.loc[1, "in_kegg_set"] = True

    with pytest.raises(bcp.SanityCheckError, match="KEGG pathway members"):
        bcp.filter_unannotated(pool)


def test_run_preserves_the_unfiltered_union_on_disk(tmp_path, sources):
    variance, centrality, kegg = sources
    variance.loc[len(variance)] = ("ENSG00000999999", 99.0, 0, True)
    centrality.loc[len(centrality)] = ("ENSG00000999999", 1, 99, False)
    indir = _write_sources(tmp_path, variance, centrality, kegg)

    pool = bcp.run(indir, tmp_path / "curated_gene_pool.csv")

    assert "ENSG00000999999" not in set(pool["gene"])
    unfiltered = pd.read_csv(indir / bcp.UNFILTERED_POOL_FILE)
    assert "ENSG00000999999" in set(unfiltered["gene"])
    assert len(unfiltered) == len(pool) + 1


def test_filter_can_be_disabled(tmp_path, sources):
    variance, centrality, kegg = sources
    variance.loc[len(variance)] = ("ENSG00000999999", 99.0, 0, True)
    centrality.loc[len(centrality)] = ("ENSG00000999999", 1, 99, False)
    indir = _write_sources(tmp_path, variance, centrality, kegg)

    pool = bcp.run(indir, tmp_path / "pool.csv", drop_unannotated=False)
    assert "ENSG00000999999" in set(pool["gene"])


def test_run_writes_both_deliverables(tmp_path, sources):
    indir = _write_sources(tmp_path, *sources)
    pool_path = tmp_path / "curated_gene_pool.csv"

    pool = bcp.run(indir, pool_path)

    assert pool_path.is_file()
    written = pd.read_csv(pool_path)
    assert list(written["gene"]) == ["A", "B", "C", "D", "E"]
    assert list(written.columns) == [
        "gene",
        "in_variance_set",
        "in_centrality_set",
        "in_kegg_set",
        "n_sources",
        "variance",
        "variance_rank",
        "in_degree",
        "centrality_rank",
        "kegg_pathway",
    ]

    summary = (indir / bcp.SUMMARY_FILE).read_text()
    assert "**5 genes**" in summary
    assert "all checks passed" in summary
    assert "ClinGen" in summary  # scope note carried into the report
