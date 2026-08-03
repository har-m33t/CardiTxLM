"""End-to-end toy-H5 tests for the CVD-only matrix materialisation.

Runs the whole Track 1 path against a synthetic ARCHS4-shaped H5 so the
64GB real file is only ever touched once the logic is known-good.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from eda.dataset import io as archs4_io
from eda.dataset.make_toy_data import make_toy_h5
from gene_pool_prep import materialize_cvd_matrix as mcm

N_GENES = 300
N_SAMPLES = 600
N_KEPT_GENES = 120


@pytest.fixture(scope="module")
def toy_env(tmp_path_factory):
    """A toy H5 + labels parquet + QC gene universe, wired like the real run."""
    root = tmp_path_factory.mktemp("gene_pool_prep")
    h5_path = make_toy_h5(root / "toy.h5", n_genes=N_GENES, n_samples=N_SAMPLES)

    rng = np.random.default_rng(20260802)
    # Disease-confirmed is a strict subset of the union pool, mirroring the
    # real parquet where is_cvd_pool == is_cvd_disease | is_cvd_tissue.
    is_disease = np.zeros(N_SAMPLES, dtype=bool)
    is_disease[rng.choice(N_SAMPLES, size=90, replace=False)] = True
    is_tissue = np.zeros(N_SAMPLES, dtype=bool)
    is_tissue[rng.choice(N_SAMPLES, size=200, replace=False)] = True

    labels = pd.DataFrame({
        "sample_index": np.arange(N_SAMPLES, dtype=np.int64),
        "geo_accession": [f"GSM{i}" for i in range(N_SAMPLES)],
        "is_cvd_disease": is_disease,
        "is_cvd_tissue": is_tissue,
        "is_cvd_pool": is_disease | is_tissue,
    })
    labels_parquet = root / "sample_labels.parquet"
    labels.to_parquet(labels_parquet, index=False)

    expression_dir = root / "expression"
    expression_dir.mkdir()
    keep_mask = np.zeros(N_GENES, dtype=bool)
    keep_mask[rng.choice(N_GENES, size=N_KEPT_GENES, replace=False)] = True
    with archs4_io.open_h5(h5_path) as h5:
        symbols = archs4_io.gene_symbols(h5)
    np.save(expression_dir / "kept_gene_mask.npy", keep_mask)
    np.save(expression_dir / "gene_symbols.npy", symbols[keep_mask])

    return {
        "root": root,
        "h5_path": h5_path,
        "labels_parquet": labels_parquet,
        "expression_dir": expression_dir,
        "keep_mask": keep_mask,
        "is_disease": is_disease,
        "is_pool": is_disease | is_tissue,
    }


def _run(toy_env, outdir_name, **kwargs):
    outdir = toy_env["root"] / outdir_name
    # The toy H5's counts are far below the real 1e5 library-size floor, so
    # the filter is off unless a test is specifically exercising it.
    kwargs.setdefault("min_library_size", 0)
    mcm.run(
        h5_path=toy_env["h5_path"],
        labels_parquet=toy_env["labels_parquet"],
        expression_dir=toy_env["expression_dir"],
        outdir=outdir,
        chunk_size=64,
        **kwargs,
    )
    x = np.load(outdir / "cvd_only_expression.npy")
    manifest = json.loads((outdir / "cvd_only_matrix_manifest.json").read_text())
    idx = np.load(outdir / "cvd_only_sample_index.npy")
    return x, idx, manifest


def test_disease_confirmed_pool_shape_and_values(toy_env):
    x, idx, manifest = _run(
        toy_env, "disease", pool_definition="disease_confirmed",
        singlecell_filter=False,
    )
    assert x.shape == (int(toy_env["is_disease"].sum()), N_KEPT_GENES)
    assert x.dtype == np.float32
    assert np.isfinite(x).all()
    assert x.min() >= 0.0
    # Rows are selected by is_cvd_disease, sorted ascending.
    np.testing.assert_array_equal(idx, np.flatnonzero(toy_env["is_disease"]))
    assert manifest["pool_definition_column"] == "is_cvd_disease"
    assert manifest["sanity_checks"]["passed"] is True


def test_values_match_direct_log2_of_raw_counts(toy_env):
    """The transform is exactly log2(raw + 1) on the QC-kept genes."""
    x, idx, _ = _run(
        toy_env, "transform", pool_definition="disease_confirmed",
        singlecell_filter=False,
    )
    with archs4_io.open_h5(toy_env["h5_path"]) as h5:
        raw = archs4_io.read_samples_by_index(h5, idx)
    expected = np.log2(raw[toy_env["keep_mask"], :].astype(np.float32) + 1.0).T
    np.testing.assert_allclose(x, expected, rtol=0, atol=0)


def test_disease_confirmed_is_subset_of_union_pool(toy_env):
    _, disease_idx, _ = _run(
        toy_env, "sub_disease", pool_definition="disease_confirmed",
        singlecell_filter=False,
    )
    _, union_idx, union_manifest = _run(
        toy_env, "sub_union", pool_definition="union_pool",
        singlecell_filter=False,
    )
    assert set(disease_idx).issubset(set(union_idx))
    assert len(disease_idx) < len(union_idx)
    assert union_manifest["pool_definition_column"] == "is_cvd_pool"


def test_singlecell_filter_drops_samples_and_records_stats(toy_env):
    _, unfiltered_idx, _ = _run(
        toy_env, "sc_off", pool_definition="disease_confirmed",
        singlecell_filter=False,
    )
    _, filtered_idx, manifest = _run(
        toy_env, "sc_on", pool_definition="disease_confirmed",
        singlecell_filter=True,
    )
    assert set(filtered_idx).issubset(set(unfiltered_idx))
    assert manifest["singlecell_filter_applied"] is True
    assert manifest["singlecell_filter_stats"]["threshold"] == 0.5
    with archs4_io.open_h5(toy_env["h5_path"]) as h5:
        sc = archs4_io.read_sample_field(h5, "singlecellprobability")
    assert (sc[filtered_idx] <= 0.5).all()


def test_manifest_records_full_provenance(toy_env):
    _, idx, manifest = _run(
        toy_env, "provenance", pool_definition="disease_confirmed",
    )
    assert manifest["n_samples"] == len(idx)
    assert manifest["n_genes"] == N_KEPT_GENES
    assert manifest["log2_pseudocount"] == 1.0
    assert manifest["sources"]["labels_parquet"].endswith("sample_labels.parquet")
    assert manifest["sources"]["archs4_h5"]["n_genes"] == N_GENES
    assert manifest["sources"]["archs4_h5"]["size_bytes"] > 0
    assert "X.npy" in manifest["not_derived_from"]
    for key in ("started", "finished", "label_stats", "sanity_checks"):
        assert manifest[key]


def test_gene_mask_is_reused_not_recomputed(toy_env):
    """The saved mask/symbols must be untouched by a run."""
    mask_path = toy_env["expression_dir"] / "kept_gene_mask.npy"
    before = mask_path.read_bytes()
    _run(toy_env, "no_mutate", pool_definition="disease_confirmed")
    assert mask_path.read_bytes() == before


def test_rejects_unknown_pool_definition(toy_env):
    with pytest.raises(ValueError, match="unknown pool definition"):
        mcm.select_pool_indices(toy_env["labels_parquet"], "everything")


def test_rejects_mask_built_against_a_different_h5(toy_env, tmp_path):
    bad_dir = tmp_path / "bad_expression"
    bad_dir.mkdir()
    np.save(bad_dir / "kept_gene_mask.npy", np.ones(N_GENES + 7, dtype=bool))
    np.save(bad_dir / "gene_symbols.npy", np.array(["A"] * (N_GENES + 7)))
    with archs4_io.open_h5(toy_env["h5_path"]) as h5:
        with pytest.raises(mcm.SanityCheckError, match="different H5"):
            mcm.load_gene_universe(bad_dir, h5)


def test_sanity_check_failure_blocks_save(toy_env):
    x = np.full((10, N_KEPT_GENES), np.nan, dtype=np.float32)
    with pytest.raises(mcm.SanityCheckError, match="NaN"):
        mcm.run_sanity_checks(
            x, np.arange(10), N_KEPT_GENES, "disease_confirmed"
        )


# --------------------------------------------------------------------------
# Library-size filter (added after Track 5's confounder screen found 172
# failed runs, the smallest with a library size of 2, in the CVD pool)
# --------------------------------------------------------------------------


def test_library_size_filter_drops_empty_libraries_and_records_stats(toy_env):
    """Threshold set between the toy libraries so the split is predictable."""
    with archs4_io.open_h5(toy_env["h5_path"]) as h5:
        pool_idx = np.flatnonzero(toy_env["is_disease"]).astype(np.int64)
        counts = archs4_io.read_samples_by_index(h5, pool_idx)
        lib = counts.sum(axis=0)
        threshold = float(np.median(lib))
        kept, stats = mcm.apply_library_size_filter(h5, pool_idx, threshold)

    expected = pool_idx[lib >= threshold]
    np.testing.assert_array_equal(kept, expected)
    assert stats["n_before"] == pool_idx.size
    assert stats["n_kept"] == expected.size
    assert stats["n_dropped"] == pool_idx.size - expected.size
    assert stats["min_library_size"] == threshold


def test_library_size_filter_is_off_at_zero(toy_env):
    with archs4_io.open_h5(toy_env["h5_path"]) as h5:
        pool_idx = np.flatnonzero(toy_env["is_disease"]).astype(np.int64)
        kept, stats = mcm.apply_library_size_filter(h5, pool_idx, 0)
    np.testing.assert_array_equal(kept, pool_idx)
    assert stats["n_dropped"] == 0


def test_emptying_the_pool_refuses_to_write(toy_env):
    """A threshold above every library must fail loudly, not save an empty
    matrix that downstream tracks would silently rank."""
    with pytest.raises(mcm.SanityCheckError, match="library-size filter removed every"):
        _run(
            toy_env, "empty", pool_definition="disease_confirmed",
            singlecell_filter=False, min_library_size=1e12,
        )


def test_manifest_records_the_library_size_filter(toy_env):
    _, _, manifest = _run(
        toy_env, "libstats", pool_definition="disease_confirmed",
        singlecell_filter=False, min_library_size=0,
    )
    assert manifest["library_size_filter_applied"] is False
    assert manifest["library_size_filter_stats"] is None
