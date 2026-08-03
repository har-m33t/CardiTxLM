"""Tests for the pool confounder screen.

The screen's whole value is that its two effect sizes are trustworthy, so
the core test is agreement with scipy/hand-computed references plus recovery
of planted confounding. If these drift, the arm contrast means nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from gene_pool_prep import screen_pool_confounders as spc

N_SAMPLES = 400
N_GENES = 25


@pytest.fixture
def planted(tmp_path):
    """A matrix with one depth-driven gene, one batch-driven gene, rest null."""
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(N_SAMPLES, N_GENES)).astype(np.float32)
    log_lib = rng.normal(size=N_SAMPLES)
    series = rng.choice(["A", "B", "C", "D"], size=N_SAMPLES)

    matrix[:, 0] += 3 * log_lib.astype(np.float32)
    for i, name in enumerate(np.unique(series)):
        matrix[series == name, 1] += 4 * i

    path = tmp_path / "m.npy"
    np.save(path, matrix)
    return path, matrix.astype(np.float64), log_lib, series


def test_depth_r2_matches_scipy(planted):
    path, matrix, log_lib, series = planted
    out = spc.screen(path, log_lib, series, np.arange(N_GENES))

    reference = np.array(
        [scipy_stats.pearsonr(matrix[:, j], log_lib)[0] ** 2 for j in range(N_GENES)]
    )
    np.testing.assert_allclose(out["depth_r2"], reference, atol=1e-12)


def test_series_eta2_matches_hand_computed_anova(planted):
    path, matrix, log_lib, series = planted

    def eta2(y):
        grand = y.mean()
        total = ((y - grand) ** 2).sum()
        between = sum(
            (y[series == s]).size * ((y[series == s]).mean() - grand) ** 2
            for s in np.unique(series)
        )
        return between / total

    out = spc.screen(path, log_lib, series, np.arange(N_GENES))
    reference = np.array([eta2(matrix[:, j]) for j in range(N_GENES)])
    np.testing.assert_allclose(out["series_eta2"], reference, atol=1e-12)


def test_planted_confounding_is_recovered_and_null_genes_stay_low(planted):
    path, matrix, log_lib, series = planted
    out = spc.screen(path, log_lib, series, np.arange(N_GENES))

    assert out.loc[0, "depth_r2"] > 0.8  # planted depth gene
    assert out.loc[1, "series_eta2"] > 0.9  # planted batch gene

    null = out.drop(index=[0, 1])
    assert null["depth_r2"].max() < 0.1
    assert null["series_eta2"].max() < 0.15


def test_keep_mask_restricts_the_sample_set(planted):
    path, matrix, log_lib, series = planted
    keep = np.zeros(N_SAMPLES, dtype=bool)
    keep[:200] = True

    out = spc.screen(path, log_lib[keep], series[keep], np.arange(N_GENES), keep=keep)
    reference = scipy_stats.pearsonr(matrix[:200, 0], log_lib[:200])[0] ** 2
    assert out.loc[0, "depth_r2"] == pytest.approx(reference, abs=1e-12)


def test_mismatched_confounder_length_is_rejected(planted):
    path, _, log_lib, series = planted
    with pytest.raises(RuntimeError, match="!= kept samples"):
        spc.screen(path, log_lib[:10], series[:10], np.arange(N_GENES))


def test_matched_contrast_only_reports_populated_strata():
    """Strata with too few genes on either side must be dropped, not reported
    as a comparison on a handful of genes."""
    per_gene = pd.DataFrame(
        {
            "arm": ["centrality_only"] * 50 + ["random_universe"] * 5,
            "detection": [0.9] * 55,
            "depth_r2": [0.5] * 55,
            "series_eta2": [0.8] * 55,
        }
    )
    assert spc.matched_contrast(per_gene).empty
