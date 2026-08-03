"""Tests for Track 3's variance ranking.

Includes the guardrail that matters most here: the module must refuse to
run against the elastic net's training matrix, which is the wrong
population and the original bug this whole effort exists to fix.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gene_pool_prep import compute_variance_genes as cvg

N_SAMPLES = 200
N_GENES = 60


def _write_matrix(dirpath, x, *, pool="disease_confirmed", n_genes=None, passed=True):
    dirpath.mkdir(parents=True, exist_ok=True)
    np.save(dirpath / "cvd_only_expression.npy", x.astype(np.float32))
    manifest = {
        "pool_definition": pool,
        "pool_definition_column": "is_cvd_disease",
        "n_samples": int(x.shape[0]),
        "n_genes": int(n_genes if n_genes is not None else x.shape[1]),
        "singlecell_filter_applied": True,
        "sanity_checks": {"passed": passed},
    }
    (dirpath / "cvd_only_matrix_manifest.json").write_text(json.dumps(manifest))
    return dirpath


def _write_universe(dirpath, symbols, n_total_genes=None):
    dirpath.mkdir(parents=True, exist_ok=True)
    n_total = n_total_genes if n_total_genes is not None else len(symbols)
    mask = np.zeros(n_total, dtype=bool)
    mask[: len(symbols)] = True
    np.save(dirpath / "gene_symbols.npy", np.array(symbols))
    np.save(dirpath / "kept_gene_mask.npy", mask)
    return dirpath


@pytest.fixture
def env(tmp_path):
    rng = np.random.default_rng(20260802)
    # Gene i gets spread proportional to i, so the true variance ranking is
    # known exactly: gene 59 highest, gene 0 lowest.
    # Spread grows with gene index. No abs()/clip here: folding at zero is
    # non-monotonic in scale and would scramble the very ranking this
    # fixture exists to make predictable.
    scales = np.linspace(0.5, 20.0, N_GENES)
    x = rng.normal(loc=8.0, scale=scales, size=(N_SAMPLES, N_GENES))
    symbols = [f"GENE{i:03d}" for i in range(N_GENES)]
    return {
        "matrix_dir": _write_matrix(tmp_path / "matrix", x),
        "expression_dir": _write_universe(tmp_path / "expression", symbols),
        "outdir": tmp_path / "out",
        "x": x,
        "symbols": symbols,
    }


def _run(env, top_percent=10.0):
    cvg.run(
        matrix_dir=env["matrix_dir"],
        expression_dir=env["expression_dir"],
        outdir=env["outdir"],
        top_percent=top_percent,
        chunk_size=32,
    )
    df = pd.read_csv(env["outdir"] / "high_variance_genes.csv")
    manifest = json.loads((env["outdir"] / "variance_manifest.json").read_text())
    return df, manifest


def test_variance_matches_numpy_exactly(env):
    v = cvg.compute_variance(np.load(env["matrix_dir"] / "cvd_only_expression.npy",
                                     mmap_mode="r"), chunk_size=32)
    expected = np.var(env["x"].astype(np.float32), axis=0, ddof=1, dtype=np.float64)
    np.testing.assert_allclose(v, expected, rtol=1e-10)


def test_chunk_size_does_not_change_result(env):
    x = np.load(env["matrix_dir"] / "cvd_only_expression.npy", mmap_mode="r")
    a = cvg.compute_variance(x, chunk_size=7)
    b = cvg.compute_variance(x, chunk_size=N_SAMPLES)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_full_universe_written_with_selected_flag(env):
    df, manifest = _run(env, top_percent=10.0)
    # Every gene present, not just the selected ones.
    assert len(df) == N_GENES
    assert set(df["gene"]) == set(env["symbols"])
    assert list(df.columns) == ["gene", "variance", "variance_rank", "selected"]
    assert df["selected"].sum() == manifest["n_selected"] == 6  # ceil(60 * 0.10)


def test_ranking_is_descending_and_complete(env):
    df, _ = _run(env)
    assert (df["variance"].diff().dropna() <= 1e-12).all()
    assert df["variance_rank"].tolist() == list(range(1, N_GENES + 1))
    # Spread grows with gene index, so rank must track index almost
    # perfectly. Adjacent scales differ by ~1.7% while the sampling error on
    # a variance from 200 samples is ~10%, so neighbours legitimately swap —
    # assert the monotone trend, not any single gene's exact position.
    gene_idx = df["gene"].str.removeprefix("GENE").astype(int)
    assert gene_idx.corr(df["variance_rank"], method="spearman") < -0.99


def test_selected_marks_exactly_the_top_ranks(env):
    df, manifest = _run(env, top_percent=25.0)
    n = manifest["n_selected"]
    assert df.loc[df["selected"], "variance_rank"].tolist() == list(range(1, n + 1))


def test_refuses_elasticnet_training_matrix(tmp_path):
    """The core guardrail: wrong population must hard-fail."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, N_GENES))
    matrix_dir = _write_matrix(tmp_path / "m", x, pool="elasticnet_training_pool")
    _write_universe(tmp_path / "e", [f"G{i}" for i in range(N_GENES)])
    with pytest.raises(cvg.InputValidationError, match="pool_definition"):
        cvg.load_inputs(matrix_dir, tmp_path / "e")


def test_refuses_matrix_with_elasticnet_sample_count(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, N_GENES))
    d = tmp_path / "m"
    d.mkdir()
    np.save(d / "cvd_only_expression.npy", x.astype(np.float32))
    (d / "cvd_only_matrix_manifest.json").write_text(json.dumps({
        "pool_definition": "disease_confirmed",
        "n_samples": 34900, "n_genes": N_GENES,
        "sanity_checks": {"passed": True},
    }))
    _write_universe(tmp_path / "e", [f"G{i}" for i in range(N_GENES)])
    with pytest.raises(cvg.InputValidationError, match="not a CVD-only sample count"):
        cvg.load_inputs(d, tmp_path / "e")


def test_refuses_when_track1_sanity_checks_failed(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, N_GENES))
    matrix_dir = _write_matrix(tmp_path / "m", x, passed=False)
    _write_universe(tmp_path / "e", [f"G{i}" for i in range(N_GENES)])
    with pytest.raises(cvg.InputValidationError, match="failing sanity checks"):
        cvg.load_inputs(matrix_dir, tmp_path / "e")


def test_reports_missing_track1_output(tmp_path):
    _write_universe(tmp_path / "e", ["A"])
    with pytest.raises(cvg.InputValidationError, match="Track 1"):
        cvg.load_inputs(tmp_path / "nonexistent", tmp_path / "e")


def test_rejects_gene_axis_mismatch(tmp_path):
    rng = np.random.default_rng(1)
    matrix_dir = _write_matrix(tmp_path / "m", rng.normal(size=(50, N_GENES)))
    _write_universe(tmp_path / "e", [f"G{i}" for i in range(N_GENES - 3)])
    with pytest.raises(cvg.InputValidationError, match="gene-axis mismatch"):
        cvg.load_inputs(matrix_dir, tmp_path / "e")


def test_rejects_out_of_range_threshold(env):
    df = cvg.rank_genes(np.array(env["symbols"]), np.arange(N_GENES, dtype=float))
    for bad in (0.0, -5.0, 100.1):
        with pytest.raises(ValueError, match="top_percent"):
            cvg.select_top_percent(df, bad)


def test_manifest_records_population_and_provenance(env):
    _, manifest = _run(env)
    assert manifest["source_population"]["pool_definition"] == "disease_confirmed"
    assert manifest["variance_ddof"] == 1
    assert "X.npy" in manifest["not_derived_from"]
    assert manifest["sanity_checks"]["passed"] is True
    assert manifest["sanity_checks"]["no_nan_or_inf"] is True


def test_sanity_check_catches_nan_variance(env):
    df = cvg.rank_genes(np.array(env["symbols"]),
                        np.full(N_GENES, np.nan, dtype=float))
    with pytest.raises(cvg.SanityCheckError, match="NaN"):
        cvg.run_sanity_checks(df, np.full(N_GENES, np.nan), 6, 10.0)


def test_spot_check_reports_absent_genes(env):
    df, _ = _run(env)
    # None of the cardiac genes exist in this toy universe.
    result = cvg.spot_check_cardiac_genes(df, 6)
    assert all(r["present"] is False for r in result.values())
