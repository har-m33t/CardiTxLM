"""Tests for the QA ground-truth layer.

Expected values are derived through a path that does not touch the module under
test **or the artifacts it reads**: the `independent` fixture pulls raw counts
straight from the ARCHS4 H5, applies its own TPM -> log1p arithmetic (it does not
call `normalize_and_align`), and builds its own symbol -> vocabulary map from the
H5 gene axis. So it re-derives the answer from source data rather than checking
`bulkformer_expression.npy` against itself — a bug in either the materialization
or the lookup fails these tests.

The pinned literals below were produced that way. Tests assert against both the
literals and the live independent derivation, so regenerating source data fails
loudly instead of drifting silently.

Values are `log1p(TPM)` over BulkFormer's 20,010-gene vocabulary — what the model
actually receives — not the project's old log2(count + 1) matrix. See
`gene_pool_prep/correction_pass_report.md`.

Sample GSM1126620 (heart failure, GSE46224) is the main fixture.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from qa_generation import gt_functions as gt

# --- Pinned fixtures, independently derived (see module docstring) ----------

HF_SAMPLE = "GSM1126620"  # is_cvd_disease, cvd_subtype == heart_failure
UNRESOLVED_SAMPLE = "GSM1201748"  # is_cvd_disease, subtype unresolved
NOT_IN_MATRIX = "GSM1201751"  # is_cvd_disease but no expression row
NEG_HARD_SAMPLE = "GSM1009635"  # tissue_only_disease_unconfirmed
NON_CVD_SAMPLE = "GSM1000981"  # outside the CVD pool entirely

MYH7 = 9.3321
TTN = 4.7577
NPPA = 8.0595
PINX1 = 0.689  # occupies 2 vocabulary columns; value is their mean

TOP5 = ["MT-CO1", "MT-ATP6", "MT-CO2", "MT-ND4", "MT-ND6"]
TOP5_VALUES = [11.8344, 10.6558, 10.6432, 10.5836, 10.4416]
BOTTOM3 = ["ABCG5", "ACTL6B", "ADAD1"]
N_ABOVE_6 = 62
N_BELOW_1 = 1449

MYH7_PARTNERS = ["CSRP3", "CKM", "LMOD2", "TRIM63", "SMYD1"]
MYH7_PARTNER_VALUES = [7.3142, 7.988, 6.7348, 5.0355, 5.0457]

N_POOL_GENES = 5797  # BulkFormer-filtered pool, was 7,166
N_STABLE_GENES = 1142  # in-vocabulary stable drivers, was 1,234
N_STABLE_ROWS = 1291  # ranking rows at nonzero_frac == 1.0
N_DROPPED_OOV = 92
TOP_DRIVER = "FCGR2A"

#: Genes the audit confirmed are outside BulkFormer's vocabulary. MT-RNR2 and
#: RN7SL2 led the *old* implementation's top-10 for some samples.
OUT_OF_VOCAB = ["NEAT1", "RPL23AP42", "MALAT1", "MT-RNR2", "RN7SL2"]


@pytest.fixture(scope="module")
def independent():
    """Re-derive expression from raw H5 counts with self-contained arithmetic.

    Deliberately does NOT import or call anything from `gt_functions`, nor read
    `bulkformer_expression.npy` / `symbol_vocab_map.parquet`, nor call
    `linear_probe.extract.normalize_and_align`.
    """
    repo = Path(gt.REPO)
    labels = pd.read_parquet(gt.SAMPLE_LABELS)
    gene_info = pd.read_csv(
        repo / "bulkencoders/checkpoints/bulkformer/support/bulkformer_gene_info.csv"
    )
    lengths_df = pd.read_csv(
        repo / "bulkencoders/checkpoints/bulkformer/support/gene_length_df.csv"
    )
    vocab_pos = {g: j for j, g in enumerate(gene_info.ensg_id.astype(str))}
    lengths = dict(zip(lengths_df.ensg_id.astype(str), lengths_df.length.astype(int)))

    h5_path = repo / "eda/dataset/cvd_data/archs4/human_gene_v2.latest.h5"
    sample_index = int(
        labels.loc[labels.geo_accession == HF_SAMPLE, "sample_index"].iloc[0]
    )
    decode = lambda arr: [  # noqa: E731
        x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)
        for x in arr
    ]
    with h5py.File(h5_path, "r") as f:
        h5_ensg = decode(f["meta/genes/ensembl_gene"][:])
        h5_symbol = decode(f["meta/genes/symbol"][:])
        counts = f["data/expression"][:, [sample_index]].ravel().astype(np.float64)

    # TPM -> log1p, written out here rather than imported.
    kb = np.array([lengths.get(g, 1000) / 1000.0 for g in h5_ensg])
    rate = counts / kb
    values = np.log1p(rate / rate.sum() * 1e6)

    symbol_to_h5col: dict[str, list[int]] = {}
    for col, (sym, gid) in enumerate(zip(h5_symbol, h5_ensg)):
        if gid.split(".")[0] in vocab_pos:
            symbol_to_h5col.setdefault(sym, []).append(col)

    def value(gene: str) -> float:
        return float(np.mean([values[c] for c in symbol_to_h5col[gene]]))

    return {"value": value, "in_vocab": set(symbol_to_h5col)}


def test_fixtures_match_independent_derivation(independent):
    """The literals, and the module, both agree with raw-count re-derivation."""
    for gene, pinned in (("MYH7", MYH7), ("TTN", TTN), ("NPPA", NPPA), ("PINX1", PINX1)):
        assert round(independent["value"](gene), 4) == pinned, gene
        assert gt.direct_abundance_query(HF_SAMPLE, gene).payload["expression"] == pinned
    assert len(gt._pool_genes()) == N_POOL_GENES
    assert len(gt._expression_rows()) == 8553


# --- Corrections under test -------------------------------------------------


def test_reads_the_bulkformer_filtered_pool_not_the_original():
    assert gt.GENE_POOL.name == "curated_gene_pool_bulkformer_filtered.csv"
    assert gt.GENE_POOL.exists()
    assert len(gt._pool_genes()) == N_POOL_GENES


def test_expression_source_is_bulkformer_input_not_the_project_matrix():
    assert gt.EXPRESSION_NPY.name == "bulkformer_expression.npy"
    assert gt.EXPRESSION_UNITS == "log1p(TPM)"
    assert np.load(gt.EXPRESSION_NPY, mmap_mode="r").shape == (8553, 20010)


@pytest.mark.parametrize("gene", OUT_OF_VOCAB)
def test_out_of_vocabulary_genes_are_refused(gene, independent):
    """A gene the model never receives cannot be grounded, so it is not answered."""
    assert gene not in independent["in_vocab"]
    res = gt.direct_abundance_query(HF_SAMPLE, gene)
    assert res.status == gt.INSUFFICIENT_DATA
    assert res.reason == "gene_not_in_bulkformer_vocab"


def test_out_of_vocabulary_genes_are_absent_from_the_pool():
    pool = set(gt._pool_genes())
    assert not (set(OUT_OF_VOCAB) & pool)


def test_every_pool_gene_is_answerable():
    """The filtered pool and the vocabulary must not disagree anywhere."""
    vocab = set(gt._symbol_columns())
    assert not (set(gt._pool_genes()) - vocab)


# --- 1. direct_abundance_query ---------------------------------------------


def test_direct_abundance_matches_independent_value():
    res = gt.direct_abundance_query(HF_SAMPLE, "MYH7")
    assert res.ok
    assert res.payload["expression"] == MYH7
    assert res.payload["units"] == "log1p(TPM)"
    assert res.payload["n_vocab_columns"] == 1
    assert res.payload["in_curated_pool"] is True


@pytest.mark.parametrize("gene,expected", [("TTN", TTN), ("NPPA", NPPA)])
def test_direct_abundance_other_cardiac_genes(gene, expected):
    assert gt.direct_abundance_query(HF_SAMPLE, gene).payload["expression"] == expected


def test_direct_abundance_averages_multi_column_symbols(independent):
    """PINX1 occupies 2 vocabulary columns; the value is their mean."""
    res = gt.direct_abundance_query(HF_SAMPLE, "PINX1")
    assert res.payload["n_vocab_columns"] == 2
    assert res.payload["expression"] == PINX1
    assert res.payload["in_curated_pool"] is False  # answerable, just not sampled
    assert round(independent["value"]("PINX1"), 4) == PINX1


def test_direct_abundance_accepts_integer_sample_index():
    labels = pd.read_parquet(gt.SAMPLE_LABELS)
    sample_index = int(
        labels.loc[labels.geo_accession == HF_SAMPLE, "sample_index"].iloc[0]
    )
    res = gt.direct_abundance_query(sample_index, "MYH7")
    assert res.ok and res.sample_id == HF_SAMPLE


@pytest.mark.parametrize(
    "sample,gene,reason",
    [
        ("GSM_DOES_NOT_EXIST", "MYH7", "unknown_sample_id"),
        (HF_SAMPLE, "NOT_A_GENE", "gene_not_in_bulkformer_vocab"),
        (NOT_IN_MATRIX, "MYH7", "sample_not_in_expression_matrix"),
        (NON_CVD_SAMPLE, "MYH7", "sample_not_in_expression_matrix"),
    ],
)
def test_direct_abundance_insufficient_data(sample, gene, reason):
    res = gt.direct_abundance_query(sample, gene)
    assert res.status == gt.INSUFFICIENT_DATA
    assert res.reason == reason
    assert not res.ok


# --- 2. threshold_query -----------------------------------------------------


def test_threshold_above_counts_and_ordering():
    res = gt.threshold_query(HF_SAMPLE, 6.0, "above")
    assert res.ok
    assert res.payload["n_matching"] == N_ABOVE_6
    assert res.payload["n_universe"] == N_POOL_GENES
    assert res.payload["genes"][:5] == TOP5
    assert all(v > 6.0 for v in res.payload["expression"])
    assert res.payload["expression"] == sorted(res.payload["expression"], reverse=True)


def test_threshold_below_counts():
    res = gt.threshold_query(HF_SAMPLE, 1.0, "below")
    assert res.payload["n_matching"] == N_BELOW_1
    assert all(v < 1.0 for v in res.payload["expression"])


def test_threshold_is_strict_not_inclusive():
    """A gene sitting exactly on the threshold must not be counted.

    The threshold is the module's own stored value for MYH7, so this tests the
    comparison operator rather than float agreement between two derivations.
    """
    exact = gt._gene_value(gt._row(HF_SAMPLE), "MYH7")[0]
    res = gt.threshold_query(HF_SAMPLE, exact, "above")
    assert "MYH7" not in res.payload["genes"]
    assert "MYH7" in gt.threshold_query(HF_SAMPLE, exact - 1e-9, "above").payload["genes"]


def test_threshold_direction_aliases_agree():
    a = gt.threshold_query(HF_SAMPLE, 6.0, "above")
    b = gt.threshold_query(HF_SAMPLE, 6.0, "exceeds")
    assert a.payload["genes"] == b.payload["genes"]


def test_threshold_flags_degenerate_answers():
    assert gt.threshold_query(HF_SAMPLE, 1e9, "above").payload["degenerate"] is True
    assert gt.threshold_query(HF_SAMPLE, -1.0, "above").payload["degenerate"] is True
    assert gt.threshold_query(HF_SAMPLE, 6.0, "above").payload["degenerate"] is False


@pytest.mark.parametrize(
    "threshold,direction,reason",
    [
        (6.0, "sideways", "unrecognized_direction:sideways"),
        (float("nan"), "above", "threshold_not_finite"),
        ("not_a_number", "above", "threshold_not_numeric"),
    ],
)
def test_threshold_insufficient_data(threshold, direction, reason):
    res = gt.threshold_query(HF_SAMPLE, threshold, direction)
    assert res.status == gt.INSUFFICIENT_DATA and res.reason == reason


# --- 3. ranking_query -------------------------------------------------------


def test_ranking_top_n_matches_independent_derivation(independent):
    res = gt.ranking_query(HF_SAMPLE, 5, "top")
    assert res.payload["genes"] == TOP5
    assert res.payload["expression"] == TOP5_VALUES
    assert res.payload["mode"] == "count"
    # Re-rank the whole pool from raw counts and confirm the same head.
    pool = list(gt._pool_genes())
    vals = np.array([independent["value"](g) for g in pool])
    order = np.lexsort((np.array(pool, dtype=object).astype(str), -vals))
    assert [pool[i] for i in order[:5]] == TOP5


def test_ranking_bottom_n_is_lowest_valued():
    res = gt.ranking_query(HF_SAMPLE, 3, "bottom")
    assert res.payload["expression"] == [0.0, 0.0, 0.0]
    assert res.payload["genes"] == BOTTOM3 == sorted(res.payload["genes"])
    assert res.payload["boundary_tie"] is True


def test_ranking_percentile_string_and_fraction_agree():
    a = gt.ranking_query(HF_SAMPLE, "1%", "top")
    b = gt.ranking_query(HF_SAMPLE, 0.01, "top")
    assert a.payload["genes"] == b.payload["genes"]
    assert a.payload["mode"] == "percentile"
    assert a.payload["n"] == 58  # ceil(0.01 * 5797)
    assert a.payload["percentile"] == 1.0


def test_ranking_top_1_agrees_with_direct_lookup():
    top = gt.ranking_query(HF_SAMPLE, 1, "top")
    direct = gt.direct_abundance_query(HF_SAMPLE, top.payload["genes"][0])
    assert direct.payload["expression"] == top.payload["expression"][0]


@pytest.mark.parametrize("spec", [0, -5, N_POOL_GENES + 1, "abc", None, True])
def test_ranking_rejects_unusable_size_spec(spec):
    res = gt.ranking_query(HF_SAMPLE, spec, "top")
    assert res.status == gt.INSUFFICIENT_DATA
    assert res.reason.startswith("unusable_size_spec")


# --- 4. comparative_query ---------------------------------------------------


def test_comparative_reports_higher_gene_and_difference():
    res = gt.comparative_query(HF_SAMPLE, "MYH7", "TTN")
    assert res.payload["higher"] == "MYH7"
    assert res.payload["lower"] == "TTN"
    assert res.payload["equal"] is False
    assert res.payload["difference"] == round(MYH7 - TTN, 4)
    assert res.payload["signed_difference_a_minus_b"] == round(MYH7 - TTN, 4)


def test_comparative_log2_fold_change_converts_from_natural_log():
    """Values are ln(TPM+1), so a raw difference is NOT a log2 fold change."""
    res = gt.comparative_query(HF_SAMPLE, "MYH7", "TTN")
    assert res.payload["log2_fold_change_a_vs_b"] == round(
        (MYH7 - TTN) / math.log(2), 4
    )
    assert res.payload["log2_fold_change_a_vs_b"] != res.payload["difference"]


def test_comparative_is_order_symmetric():
    fwd = gt.comparative_query(HF_SAMPLE, "MYH7", "TTN")
    rev = gt.comparative_query(HF_SAMPLE, "TTN", "MYH7")
    assert fwd.payload["higher"] == rev.payload["higher"] == "MYH7"
    assert fwd.payload["difference"] == rev.payload["difference"]
    assert (
        fwd.payload["log2_fold_change_a_vs_b"]
        == -rev.payload["log2_fold_change_a_vs_b"]
    )


def test_comparative_detects_equality():
    res = gt.comparative_query(HF_SAMPLE, "MYH7", "MYH7")
    assert res.payload["equal"] is True
    assert res.payload["higher"] is None
    assert res.payload["difference"] == 0.0


def test_comparative_names_the_missing_gene():
    res = gt.comparative_query(HF_SAMPLE, "MYH7", "NEAT1")
    assert res.status == gt.INSUFFICIENT_DATA
    assert res.reason == "gene_not_in_bulkformer_vocab:NEAT1"


# --- 5. interaction_network_query -------------------------------------------


def test_interaction_partners_match_saved_edge_list():
    res = gt.interaction_network_query(HF_SAMPLE, "MYH7", n=5)
    assert res.ok
    assert [p["gene"] for p in res.payload["partners"]] == MYH7_PARTNERS
    assert [p["expression"] for p in res.payload["partners"]] == MYH7_PARTNER_VALUES
    assert [p["rank"] for p in res.payload["partners"]] == [1, 2, 3, 4, 5]


def test_interaction_partner_values_match_independent_derivation(independent):
    res = gt.interaction_network_query(HF_SAMPLE, "MYH7", n=5)
    for partner in res.payload["partners"]:
        assert round(independent["value"](partner["gene"]), 4) == partner["expression"]


def test_interaction_partner_values_agree_with_direct_lookup():
    res = gt.interaction_network_query(HF_SAMPLE, "MYH7", n=10)
    for partner in res.payload["partners"]:
        direct = gt.direct_abundance_query(HF_SAMPLE, partner["gene"])
        assert direct.payload["expression"] == partner["expression"]


def test_interaction_returns_exactly_the_requested_count_ordered_by_correlation():
    res = gt.interaction_network_query(HF_SAMPLE, "MYH7", 100)
    assert res.payload["n_partners"] == 100
    rs = [p["pearson_r"] for p in res.payload["partners"]]
    assert rs == sorted(rs, reverse=True)
    assert all(-1.001 <= r <= 1.001 for r in rs)


def test_every_partner_of_every_gene_is_in_vocabulary():
    """The whole point of the edge-list rebuild."""
    edges = pd.read_parquet(gt.COEXPRESSION_EDGES)
    vocab = set(gt._symbol_columns())
    assert not (set(edges.partner.astype(str)) - vocab)
    assert not (set(edges.gene.astype(str)) - set(gt._pool_genes()))
    assert edges.gene.nunique() == N_POOL_GENES


def test_interaction_never_returns_the_query_gene_as_its_own_partner():
    res = gt.interaction_network_query(HF_SAMPLE, "MYH7", 100)
    assert "MYH7" not in [p["gene"] for p in res.payload["partners"]]


def test_interaction_rejects_genes_outside_the_curated_pool():
    non_pool = next(g for g in gt._symbol_columns() if g not in set(gt._pool_genes()))
    res = gt.interaction_network_query(HF_SAMPLE, non_pool, 10)
    assert res.status == gt.INSUFFICIENT_DATA
    assert res.reason == "gene_not_in_coexpression_edges"


@pytest.mark.parametrize("n", [None, 0, -5, 101, "ten", 10.5, True])
def test_interaction_requires_an_explicitly_bound_valid_n(n):
    """N is asked for in the question text now, so GT never guesses it."""
    res = gt.interaction_network_query(HF_SAMPLE, "MYH7", n)
    assert res.status == gt.INSUFFICIENT_DATA
    assert res.reason == f"unusable_partner_count:{n!r}"


def test_interaction_n_is_mandatory_in_the_signature():
    params = inspect.signature(gt.interaction_network_query).parameters
    assert params["n"].default is inspect.Parameter.empty


def test_interaction_insufficient_for_sample_without_expression():
    res = gt.interaction_network_query(NOT_IN_MATRIX, "MYH7", 10)
    assert res.reason == "sample_not_in_expression_matrix"


# --- 6. disease_subtype_classification --------------------------------------


def test_subtype_returns_disease_confirmed_label():
    res = gt.disease_subtype_classification(HF_SAMPLE)
    assert res.ok
    assert res.payload["subtype"] == "heart_failure"
    assert res.payload["in_expression_matrix"] is True


@pytest.mark.parametrize(
    "sample,subtype",
    [
        ("GSM1085736", "coronary_artery_disease"),
        ("GSM2309837", "hypertension"),
        ("GSM3908451", "arrhythmia_afib"),
        ("GSM1841266", "cardiomyopathy_other"),
    ],
)
def test_subtype_covers_every_resolved_class(sample, subtype):
    assert gt.disease_subtype_classification(sample).payload["subtype"] == subtype


def test_subtype_insufficient_for_unresolved_bucket():
    res = gt.disease_subtype_classification(UNRESOLVED_SAMPLE)
    assert res.status == gt.INSUFFICIENT_DATA
    assert res.reason == "subtype_unresolved:disease_matched_subtype_unresolved"


def test_subtype_insufficient_for_tissue_only_and_non_cvd():
    for sample in (NEG_HARD_SAMPLE, NON_CVD_SAMPLE):
        res = gt.disease_subtype_classification(sample)
        assert res.status == gt.INSUFFICIENT_DATA
        assert res.reason == "not_disease_confirmed"


# --- 7. comparative_differential_reasoning ----------------------------------


def test_differential_accepts_neg_hard():
    res = gt.comparative_differential_reasoning(HF_SAMPLE, "neg_hard")
    assert res.ok
    assert res.payload["comparison_group"] == "neg_hard"
    assert res.payload["n_comparison"] == 22307
    assert res.payload["n_positive"] == 8725
    assert res.payload["separability"]["roc_auc_mean"] == pytest.approx(0.7806, abs=1e-4)


def test_differential_accepts_the_subtype_string_alias():
    a = gt.comparative_differential_reasoning(HF_SAMPLE, "neg_hard")
    b = gt.comparative_differential_reasoning(
        HF_SAMPLE, "tissue_only_disease_unconfirmed"
    )
    assert b.ok and b.payload == a.payload


@pytest.mark.parametrize(
    "group",
    ["neg_whole_corpus", "random_tissue", "random_bulk_tissue", "healthy_controls", ""],
)
def test_differential_rejects_every_other_comparison_group(group):
    res = gt.comparative_differential_reasoning(HF_SAMPLE, group)
    assert res.status == gt.INSUFFICIENT_DATA
    assert res.reason == f"comparison_group_not_permitted:{group}"
    assert res.payload == {}


def test_differential_rejects_non_positive_samples():
    for sample in (NEG_HARD_SAMPLE, NON_CVD_SAMPLE):
        res = gt.comparative_differential_reasoning(sample, "neg_hard")
        assert res.status == gt.INSUFFICIENT_DATA
        assert res.reason == "sample_not_a_probe_positive"


def test_differential_declares_no_per_gene_contrast():
    payload = gt.comparative_differential_reasoning(HF_SAMPLE, "neg_hard").payload
    assert payload["per_gene_differential"] is None
    assert "no expression matrix" in payload["per_gene_differential_reason"]


def test_differential_carries_the_confound_contrast():
    ctx = gt.comparative_differential_reasoning(HF_SAMPLE, "neg_hard").payload[
        "confound_context"
    ]
    assert ctx["random_tissue_roc_auc"] > 0.9
    assert "not valid ground truth" in ctx["note"]


# --- 8. gene_driver_reasoning -----------------------------------------------


def test_gene_driver_returns_the_in_vocabulary_stable_subset():
    res = gt.gene_driver_reasoning(HF_SAMPLE, N_STABLE_GENES)
    assert res.ok
    assert res.payload["n_returned"] == N_STABLE_GENES
    assert res.payload["n_stable_rows_before_symbol_dedup"] == N_STABLE_ROWS
    assert res.payload["n_dropped_out_of_bulkformer_vocab"] == N_DROPPED_OOV
    assert res.payload["stability_criterion"] == "nonzero_frac == 1.0 (all 5 outer folds)"


def test_gene_driver_excludes_out_of_vocabulary_drivers(independent):
    """RPL23AP42 (rank 8) and NEAT1 (rank 9) must be gone."""
    genes = {g["gene"] for g in gt.gene_driver_reasoning(HF_SAMPLE, N_STABLE_GENES).payload["genes"]}
    assert "RPL23AP42" not in genes
    assert "NEAT1" not in genes
    assert not (genes - independent["in_vocab"])


def test_gene_driver_ranking_matches_the_csv_filtered_to_vocab():
    """Re-derived from the ranking CSV, not from the module's cached frame."""
    rank = pd.read_csv(gt.GENE_RANKING)
    vocab = set(gt._symbol_columns())
    expected = (
        rank[rank.nonzero_frac == 1.0]
        .sort_values("abs_mean_coef", ascending=False)
        .drop_duplicates("gene_symbol")
        .loc[lambda d: d.gene_symbol.astype(str).isin(vocab)]
        .gene_symbol.head(5)
        .tolist()
    )
    genes = [g["gene"] for g in gt.gene_driver_reasoning(HF_SAMPLE, 5).payload["genes"]]
    assert genes == expected
    assert genes[0] == TOP_DRIVER


def test_gene_driver_filter_did_not_rerun_the_elastic_net():
    """Coefficients must be the CSV's originals, untouched by the filter."""
    rank = (
        pd.read_csv(gt.GENE_RANKING)
        .sort_values("abs_mean_coef", ascending=False)
        .drop_duplicates("gene_symbol")
        .set_index("gene_symbol")
    )
    for g in gt.gene_driver_reasoning(HF_SAMPLE, 20).payload["genes"]:
        assert g["mean_coef"] == round(float(rank.loc[g["gene"], "mean_coef"]), 5)
        assert rank.loc[g["gene"], "nonzero_frac"] == 1.0


def test_gene_driver_reports_direction_and_is_ordered():
    genes = gt.gene_driver_reasoning(HF_SAMPLE, 50).payload["genes"]
    assert len(genes) == 50
    coefs = [g["abs_mean_coef"] for g in genes]
    assert coefs == sorted(coefs, reverse=True)
    for g in genes:
        expected = "up_in_cvd" if g["mean_coef"] > 0 else "down_in_cvd"
        assert g["direction"] == expected


def test_gene_driver_is_identical_across_subtypes():
    answers = [
        gt.gene_driver_reasoning(s, 25).payload["genes"]
        for s in (HF_SAMPLE, "GSM1085736", "GSM2309837", "GSM3908451")
    ]
    assert all(a == answers[0] for a in answers)


def test_gene_driver_gate_is_disease_confirmation():
    assert gt.gene_driver_reasoning(NOT_IN_MATRIX, 10).ok
    for sample in (NEG_HARD_SAMPLE, NON_CVD_SAMPLE):
        res = gt.gene_driver_reasoning(sample, 10)
        assert res.status == gt.INSUFFICIENT_DATA
        assert res.reason == "not_disease_confirmed"


@pytest.mark.parametrize("top_n", [None, 0, -5, N_STABLE_GENES + 1, "ten", 10.5, True])
def test_gene_driver_requires_an_explicitly_bound_valid_top_n(top_n):
    """Unbound top_n used to mean "all 1,142", which no template asks for."""
    res = gt.gene_driver_reasoning(HF_SAMPLE, top_n)
    assert res.status == gt.INSUFFICIENT_DATA
    assert res.reason == f"unusable_top_n:{top_n!r}"


def test_gene_driver_top_n_is_mandatory_in_the_signature():
    params = inspect.signature(gt.gene_driver_reasoning).parameters
    assert params["top_n"].default is inspect.Parameter.empty


@pytest.mark.parametrize("top_n", [10, 15, 20])
def test_gene_driver_returns_exactly_the_requested_count(top_n):
    res = gt.gene_driver_reasoning(HF_SAMPLE, top_n)
    assert res.ok
    assert res.payload["n_returned"] == top_n
    assert len(res.payload["genes"]) == top_n


def test_gene_driver_counts_are_nested_prefixes():
    """A larger bound must extend the smaller one, not reorder it."""
    small = [g["gene"] for g in gt.gene_driver_reasoning(HF_SAMPLE, 10).payload["genes"]]
    large = [g["gene"] for g in gt.gene_driver_reasoning(HF_SAMPLE, 20).payload["genes"]]
    assert large[:10] == small


def test_gene_driver_declares_its_broad_scope():
    payload = gt.gene_driver_reasoning(HF_SAMPLE, 1).payload
    assert payload["scope"] == "broad_cardiovascular_disease"
    assert payload["not_subtype_specific"] is True
    assert payload["sample_role"] == "eligibility_gate_only"


# --- Guardrails -------------------------------------------------------------


def test_gene_driver_exposes_no_subtype_parameter():
    params = set(inspect.signature(gt.gene_driver_reasoning).parameters)
    assert params == {"sample_id", "top_n"}
    assert not any(
        tok in p for p in params for tok in ("subtype", "condition", "disease")
    )


def test_only_neg_hard_is_a_permitted_comparison_group():
    assert gt.VALID_COMPARISON_GROUPS == {
        "neg_hard",
        "tissue_only_disease_unconfirmed",
    }


def test_registry_exposes_exactly_the_eight_categories():
    assert set(gt.GT_FUNCTIONS) == {
        "direct_abundance_query",
        "threshold_query",
        "ranking_ordering_query",
        "comparative_query",
        "interaction_network_query",
        "disease_subtype_classification",
        "comparative_differential_reasoning",
        "gene_driver_reasoning",
    }


def test_every_function_reports_insufficient_data_for_an_unknown_sample():
    calls = {
        "direct_abundance_query": ("MYH7",),
        "threshold_query": (6.0,),
        "ranking_ordering_query": (5,),
        "comparative_query": ("MYH7", "TTN"),
        "interaction_network_query": ("MYH7", 10),
        "disease_subtype_classification": (),
        "comparative_differential_reasoning": ("neg_hard",),
        "gene_driver_reasoning": (10,),
    }
    for name, args in calls.items():
        res = gt.GT_FUNCTIONS[name]("GSM_DOES_NOT_EXIST", *args)
        assert res.status == gt.INSUFFICIENT_DATA, name
        assert res.reason == "unknown_sample_id", name
        assert res.payload == {}, name


def test_missing_artifact_is_not_reported_as_insufficient_data(monkeypatch):
    monkeypatch.setattr(gt, "COEXPRESSION_EDGES", Path("/nonexistent/edges.parquet"))
    gt._coexpression_edges.cache_clear()
    try:
        with pytest.raises(gt.MissingArtifactError):
            gt.interaction_network_query(HF_SAMPLE, "MYH7", 10)
    finally:
        gt._coexpression_edges.cache_clear()
