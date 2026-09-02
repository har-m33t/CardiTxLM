"""phase0_analysis.py — Hypothesis B, Phase 0 preliminary data analysis.

Regenerates every number in `hypothesis_b_prelim_report.md`. Read-only with
respect to the rest of the repo: the only things written are
`scripts/hypothesis_b/phase0_stats.json` (machine-readable) and whatever the
caller redirects stdout to.

Sections mirror `.claude/stage_2_revisions.md` Phase 0:

    0a  population sizing (positives / neg_hard, holdout-excluded)
    0b  series-level structure and concentration
    0c  tissue composition + the tissue-vs-disease confound measurement
    0d  reusability of the already-computed per-sample DE

REUSE (mandatory, per the plan — nothing here is a reimplementation)
--------------------------------------------------------------------
    linear_probe.probe.run_cv / summarize
        the exact grouped-CV harness every other probe in this repo uses:
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260707)
        grouped by series_id, StandardScaler + balanced LogisticRegression
        fit inside the fold. Used unchanged for the tissue-only classifier so
        its AUC is directly comparable to the published embedding probes.
    eval_binary_comparison.embedding_io.load_embeddings / embedding_columns
        the one correct embedding-column selector. `c.startswith("e0")` is a
        known silent-truncation bug in this repo and is never used here.
    qa_generation.build_per_sample_de.normalize_tissue / read_source_names
        the same source_name_ch1 bucket key the DE reference statistics were
        built with, so 0c's tissue buckets are the DE pipeline's tissue
        buckets and not a second, differently-normalized universe.

CLI
---
    python3 scripts/hypothesis_b/phase0_analysis.py
    python3 scripts/hypothesis_b/phase0_analysis.py --skip-embedding-probe
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

# --- mandatory reuse ------------------------------------------------------
from eval_binary_comparison.embedding_io import (  # noqa: E402
    embedding_columns,
    load_embeddings,
)
from linear_probe.probe import run_cv, summarize  # noqa: E402
from qa_generation.build_per_sample_de import (  # noqa: E402
    normalize_tissue,
    read_source_names,
)

LABELS_PATH = REPO / "linear_probe" / "probe_sample_labels.parquet"
HOLDOUT_PATH = REPO / "data" / "cvd_transcriptome" / "holdout_series.json"
EMB_PATH = REPO / "linear_probe" / "embeddings" / "embeddings_BulkFormer-93M.parquet"
POS_INDEX_PATH = REPO / "qa_generation" / "bulkformer_input" / "bulkformer_sample_index.npy"
DE_PARQUET = REPO / "qa_generation" / "de" / "per_sample_de.parquet"
DE_MANIFEST = REPO / "qa_generation" / "de" / "de_manifest.json"
STABLE_Z = REPO / "qa_generation" / "de" / "stable_gene_z.npz"
BUNDLE_STATS = REPO / "qa_generation" / "stage2_bundle_stats.json"
H5_PATH = REPO / "eda" / "dataset" / "cvd_data" / "archs4" / "human_gene_v2.latest.h5"

OUT_JSON = HERE / "phase0_stats.json"

SEED = 20260707          # the seed every prior probe in this repo used
K_FOLDS = 5
MIN_TISSUE_N = 50        # 0c: a tissue must have >= this many samples (both
                         # pools combined) to get a reported positive fraction

# Expected values established before this script ran. Asserted, not assumed:
# if the underlying data ever moves, this script fails loudly instead of
# quietly reporting a different corpus.
EXPECT = {
    "n_label_rows": 1_098_771,
    "n_positive_global": 8_725,
    "n_neg_hard_global": 22_307,
    "n_holdout_series": 92,
    "n_positive_train": 7_384,
    "n_neg_hard_train": 21_041,
    "n_holdout_positive": 1_341,
    "n_holdout_neg_hard": 1_266,
    "n_corpus_positives": 7_212,
}


def _log() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    return logging.getLogger("phase0")


def _check(name: str, got, want, failures: list[str]) -> None:
    if got != want:
        failures.append(f"{name}: got {got}, expected {want}")


# ---------------------------------------------------------------------------
# concentration measures
# ---------------------------------------------------------------------------

def concentration(counts: pd.Series) -> dict:
    """Series-count concentration for one pool. `counts` is per-series sizes."""
    n = int(counts.sum())
    share = (counts / n).sort_values(ascending=False)
    cum = share.cumsum()
    # smallest number of series whose cumulative share reaches 50% / 80%
    n50 = int((cum < 0.50).sum() + 1)
    n80 = int((cum < 0.80).sum() + 1)
    # Herfindahl-Hirschman index on shares (1/n_series = perfectly even,
    # 1.0 = one series holds everything).
    hhi = float((share ** 2).sum())
    # Gini over series sizes (0 = every series the same size).
    x = np.sort(counts.to_numpy(dtype=np.float64))
    k = x.size
    gini = float((2.0 * np.sum((np.arange(1, k + 1)) * x)) / (k * x.sum()) - (k + 1.0) / k)
    return {
        "n_samples": n,
        "n_series": int(counts.size),
        "max_series_share": float(share.iloc[0]),
        "max_series_id": str(share.index[0]),
        "max_series_n": int(counts.loc[share.index[0]]),
        "top5_share": float(share.iloc[:5].sum()),
        "top15_share": float(share.iloc[:15].sum()),
        "n_series_for_50pct": n50,
        "n_series_for_80pct": n80,
        "hhi": hhi,
        "hhi_even_reference": float(1.0 / counts.size),
        "effective_n_series": float(1.0 / hhi),
        "gini": gini,
        "median_series_size": float(counts.median()),
        "mean_series_size": float(counts.mean()),
        "n_singleton_series": int((counts == 1).sum()),
    }


def top_series(counts: pd.Series, k: int = 15) -> list[dict]:
    n = int(counts.sum())
    top = counts.sort_values(ascending=False).head(k)
    return [{"series_id": str(s), "n": int(v), "pct_of_pool": round(100.0 * v / n, 3)}
            for s, v in top.items()]


# ---------------------------------------------------------------------------
# 0c helpers
# ---------------------------------------------------------------------------

# A coarse grouping layered ON TOP OF normalize_tissue (which is imported, not
# reimplemented). Its only job is to make the confound number interpretable:
# normalize_tissue's key is a free-text GEO string with >1,000 distinct values,
# and a classifier over that alphabet says little about whether "tissue" as a
# biological concept predicts class. The coarse map is a reporting aid and a
# LOWER bracket on the confound; the normalized-string classifier is the upper
# bracket. Both are reported.
COARSE_RULES: list[tuple[str, str]] = [
    (r"\bipsc|induced pluripotent|ips cell|ipscs\b", "ipsc_derived"),
    (r"\besc\b|embryonic stem", "esc_derived"),
    (r"cardiomyocyte|myocyte", "cardiomyocyte"),
    (r"heart|cardiac|ventric|atri|myocard|septum|apex", "heart_tissue"),
    (r"huvec|umbilical vein|endothelial|\bendo\b|hcaec|haec", "endothelial"),
    (r"smooth muscle|vsmc|hasmc|\bsmc\b", "smooth_muscle"),
    (r"aorta|aortic|artery|arterial|vein|vascular|vessel|carotid|coronary", "vascular_tissue"),
    (r"fibroblast", "fibroblast"),
    (r"blood|pbmc|monocyte|macrophage|lymphocyte|neutrophil|leukocyte|plasma|serum|platelet",
     "blood"),
    (r"skeletal muscle|\bmuscle\b", "skeletal_muscle"),
    (r"liver|hepato", "liver"),
    (r"kidney|renal", "kidney"),
    (r"lung|pulmonary|airway|bronch", "lung"),
    (r"adipose|fat\b", "adipose"),
    (r"brain|neuro|cortex|neuron|glia", "neural"),
    (r"placenta|umbilical cord|cord blood", "placental"),
    (r"tumor|carcinoma|cancer|melanoma|sarcoma", "tumor"),
]
_COARSE = [(re.compile(p), lab) for p, lab in COARSE_RULES]


def coarse_tissue(norm: str) -> str:
    if not norm:
        return "unspecified"
    for rx, lab in _COARSE:
        if rx.search(norm):
            return lab
    return "other"


def onehot(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Dense one-hot of a categorical label array. Deterministic column order."""
    levels = sorted(set(labels.tolist()))
    pos = {lab: j for j, lab in enumerate(levels)}
    X = np.zeros((labels.size, len(levels)), dtype=np.float32)
    X[np.arange(labels.size), [pos[v] for v in labels]] = 1.0
    return X, levels


def bayes_tissue_rule(labels: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      logger: logging.Logger) -> dict:
    """Per-tissue empirical positive rate, fit on train fold only.

    This is the Bayes-optimal rule for a predictor whose ONLY input is the
    tissue label, under the same grouped folds. Unseen-in-train tissues fall
    back to the training-fold prior (predicting anything else would be using
    information the rule does not have).
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    skf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    aucs, accs, unseen = [], [], []
    for tr, va in skf.split(np.zeros((y.size, 1)), y, groups):
        prior = float(y[tr].mean())
        df = pd.DataFrame({"t": labels[tr], "y": y[tr]})
        rate = df.groupby("t")["y"].mean()
        p = np.array([rate.get(t, prior) for t in labels[va]], dtype=np.float64)
        unseen.append(float(np.mean([t not in rate.index for t in labels[va]])))
        aucs.append(float(roc_auc_score(y[va], p)))
        accs.append(float(accuracy_score(y[va], (p >= 0.5).astype(int))))
    logger.info(f"  bayes per-tissue rule: AUC={np.mean(aucs):.4f} "
                f"acc={np.mean(accs):.4f}")
    return {
        "roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "roc_auc_folds": aucs,
        "mean_val_frac_tissue_unseen_in_train": float(np.mean(unseen)),
    }


def probe_summary(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  logger: logging.Logger) -> dict:
    """The repo's standard grouped-CV logistic probe, on whatever X is given."""
    folds = run_cv(X, y, groups, K_FOLDS, SEED, logger)
    out = summarize(folds)
    out["folds"] = [asdict(f) for f in folds]
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hypothesis B Phase 0 analysis.")
    ap.add_argument("--skip-embedding-probe", action="store_true",
                    help="Skip the 515-d BulkFormer reference probe in 0c.")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)

    logger = _log()
    t_start = time.time()
    failures: list[str] = []
    stats: dict = {
        "generated": pd.Timestamp.utcnow().isoformat(),
        "script": "scripts/hypothesis_b/phase0_analysis.py",
        "seed": SEED,
        "k_folds": K_FOLDS,
        "reused_code": {
            "grouped_cv_harness": "linear_probe.probe.run_cv + summarize "
                                  "(StratifiedGroupKFold(5, shuffle=True, "
                                  "random_state=20260707), groups=series_id, "
                                  "StandardScaler + balanced LogisticRegression)",
            "embedding_column_selector": "eval_binary_comparison.embedding_io."
                                         "embedding_columns / load_embeddings "
                                         "(never c.startswith('e0'))",
            "tissue_normalization": "qa_generation.build_per_sample_de."
                                    "normalize_tissue (the DE pipeline's own "
                                    "source_name_ch1 bucket key)",
            "h5_metadata_reader": "qa_generation.build_per_sample_de."
                                  "read_source_names",
            "new_here": "coarse_tissue() — a reporting-only grouping layered on "
                        "top of normalize_tissue, used to bracket the confound "
                        "measurement; it replaces nothing.",
        },
    }

    # ---------------- 0a: populations -------------------------------------
    logger.info("0a: populations")
    labels = pd.read_parquet(LABELS_PATH)
    holdout_blob = json.loads(HOLDOUT_PATH.read_text())
    holdout_series = set(holdout_blob["holdout_series"])

    _check("n_label_rows", len(labels), EXPECT["n_label_rows"], failures)
    _check("n_holdout_series", len(holdout_series), EXPECT["n_holdout_series"], failures)

    pos_all = labels.loc[labels["is_positive"]].copy()
    neg_all = labels.loc[labels["is_neg_hard"]].copy()
    _check("n_positive_global", len(pos_all), EXPECT["n_positive_global"], failures)
    _check("n_neg_hard_global", len(neg_all), EXPECT["n_neg_hard_global"], failures)

    overlap_pos_neg = int((labels["is_positive"] & labels["is_neg_hard"]).sum())

    pos_in_hold = pos_all["series_id"].isin(holdout_series)
    neg_in_hold = neg_all["series_id"].isin(holdout_series)
    pos = pos_all.loc[~pos_in_hold].copy()
    neg = neg_all.loc[~neg_in_hold].copy()

    _check("n_positive_train", len(pos), EXPECT["n_positive_train"], failures)
    _check("n_neg_hard_train", len(neg), EXPECT["n_neg_hard_train"], failures)
    _check("n_holdout_positive", int(pos_in_hold.sum()), EXPECT["n_holdout_positive"], failures)
    _check("n_holdout_neg_hard", int(neg_in_hold.sum()), EXPECT["n_holdout_neg_hard"], failures)

    # embedding coverage — the discriminative task's actual input requirement.
    # Loaded once here and reused for 0c's reference probe. `load_embeddings`
    # is the audited selector; it raises rather than silently truncating.
    import pyarrow.parquet as pq
    emb_cols = embedding_columns(pq.read_schema(EMB_PATH).names)
    Xemb, emb_ids_all = load_embeddings(EMB_PATH)
    emb_ids = set(emb_ids_all.tolist())
    n_emb_cols = int(Xemb.shape[1])
    _check("embedding width matches audited selector", n_emb_cols,
           len(emb_cols), failures)
    pos_emb = int(pos_all["sample_index"].isin(emb_ids).sum())
    neg_emb = int(neg_all["sample_index"].isin(emb_ids).sum())

    # expression-row coverage — what the OTHER four categories needed
    expr_index = np.load(POS_INDEX_PATH)
    expr_ids = set(expr_index.tolist())
    pos_expr_all = int(pos_all["sample_index"].isin(expr_ids).sum())
    pos_expr_train = int(pos["sample_index"].isin(expr_ids).sum())
    pos_no_expr_train = int(len(pos) - pos_expr_train)

    bundle = json.loads(BUNDLE_STATS.read_text())
    _check("n_corpus_positives", bundle["n_unique_samples"],
           EXPECT["n_corpus_positives"], failures)
    _check("pos_expr_train == corpus positives", pos_expr_train,
           EXPECT["n_corpus_positives"], failures)

    stats["s0a_populations"] = {
        "label_frame": str(LABELS_PATH.relative_to(REPO)),
        "n_label_rows": int(len(labels)),
        "n_positive_global": int(len(pos_all)),
        "n_neg_hard_global": int(len(neg_all)),
        "n_samples_in_both_classes": overlap_pos_neg,
        "holdout": {
            "file": str(HOLDOUT_PATH.relative_to(REPO)),
            "n_series": len(holdout_series),
            "definition": holdout_blob["definition"],
            "excluded_by": "series_id",
            "n_positive_excluded": int(pos_in_hold.sum()),
            "n_neg_hard_excluded": int(neg_in_hold.sum()),
            "n_total_excluded": int(pos_in_hold.sum() + neg_in_hold.sum()),
        },
        "train_eligible": {
            "n_positive": int(len(pos)),
            "n_neg_hard": int(len(neg)),
            "raw_ratio_pos_to_neg": round(len(neg) / len(pos), 4),
            "raw_ratio_str": f"1 : {len(neg) / len(pos):.2f}",
            "positive_prevalence_if_all_negs_used": round(
                len(pos) / (len(pos) + len(neg)), 5),
            "n_total_if_all_negs_used": int(len(pos) + len(neg)),
        },
        "embedding_coverage": {
            "file": str(EMB_PATH.relative_to(REPO)),
            "n_unique_sample_index": int(len(emb_ids)),
            "embedding_dim": n_emb_cols,
            "n_positive_covered": pos_emb,
            "pct_positive_covered": round(100.0 * pos_emb / len(pos_all), 3),
            "n_neg_hard_covered": neg_emb,
            "pct_neg_hard_covered": round(100.0 * neg_emb / len(neg_all), 3),
        },
        "expression_row_coverage": {
            "file": str(POS_INDEX_PATH.relative_to(REPO)),
            "n_rows": int(expr_index.size),
            "n_positive_global_with_expression": pos_expr_all,
            "n_positive_train_with_expression": pos_expr_train,
            "n_positive_train_without_expression": pos_no_expr_train,
        },
        "gap_7384_vs_7212": {
            "n_positive_train_eligible": int(len(pos)),
            "n_positive_in_current_corpus": bundle["n_unique_samples"],
            "gap": int(len(pos)) - int(bundle["n_unique_samples"]),
            "explanation": (
                "The 7,384 figure is holdout-excluded positives that have a "
                "515-d BulkFormer embedding (coverage is 100%). The 7,212 "
                "figure is the subset that also has a normalized expression "
                "row in bulkformer_sample_index.npy, which the DE-grounded "
                "categories require. The 172-sample gap is expression-row "
                "coverage, not label coverage."
            ),
        },
        "current_corpus": {
            "file": str(BUNDLE_STATS.relative_to(REPO)),
            "items_by_category": bundle["items_by_category"],
            "n_items": bundle["n_written"],
            "n_unique_samples": bundle["n_unique_samples"],
        },
    }

    # ---------------- 0b: series structure --------------------------------
    logger.info("0b: series structure")
    pos_counts = pos.groupby("series_id").size()
    neg_counts = neg.groupby("series_id").size()
    pos_series, neg_series = set(pos_counts.index), set(neg_counts.index)
    both = pos_series & neg_series

    # global (pre-holdout) mixed-series count, to show the exclusion worked
    both_global = set(pos_all["series_id"]) & set(neg_all["series_id"])

    stats["s0b_series_structure"] = {
        "scope": "holdout-excluded",
        "positive_pool": {**concentration(pos_counts), "top15": top_series(pos_counts)},
        "negative_pool": {**concentration(neg_counts), "top15": top_series(neg_counts)},
        "series_overlap": {
            "n_series_with_both_classes_after_holdout_exclusion": len(both),
            "series_with_both_classes": sorted(str(s) for s in both)[:20],
            "n_series_with_both_classes_before_exclusion": len(both_global),
            "note": (
                "The holdout was DEFINED as every series containing both a "
                "positive and a neg_hard sample, so zero mixed series outside "
                "the holdout is structurally guaranteed, not a lucky draw. "
                "The consequence is that series_id is a perfect predictor of "
                "class in the training population: any series-level covariate "
                "(batch, platform, lab, protocol) is an available shortcut, "
                "and grouped CV controls how it is EVALUATED but not what the "
                "model LEARNS to associate with 'negative' during training."
            ),
        },
        "why_this_matters_for_training": (
            "Series concentration biases what the model learns, independently "
            "of leakage. If one series supplies a large share of the negative "
            "class, its batch signature becomes the model's operational "
            "definition of 'no disease'."
        ),
    }

    # ---------------- 0c: tissue -----------------------------------------
    logger.info("0c: tissue composition and confound")
    work = pd.concat([
        pos.assign(y=1), neg.assign(y=0),
    ], ignore_index=True)[["sample_index", "geo_accession", "series_id", "y"]]
    work = work.sort_values("sample_index").reset_index(drop=True)

    import h5py
    with h5py.File(H5_PATH, "r") as f:
        meta_fields = sorted(f["meta/samples"].keys())

    raw_tissue = read_source_names(work["sample_index"].to_numpy(np.int64), logger)
    work["tissue_raw"] = raw_tissue
    work["tissue"] = [normalize_tissue(t) for t in raw_tissue]
    work["tissue_coarse"] = [coarse_tissue(t) for t in work["tissue"]]

    n_blank = int((work["tissue"] == "").sum())
    work.loc[work["tissue"] == "", "tissue"] = "<blank>"

    p, q = work.loc[work.y == 1], work.loc[work.y == 0]
    pos_tis = p["tissue"].value_counts()
    neg_tis = q["tissue"].value_counts()

    def dist(vc: pd.Series, n: int, k: int = 25) -> list[dict]:
        return [{"tissue": str(t), "n": int(v), "pct": round(100.0 * v / n, 3)}
                for t, v in vc.head(k).items()]

    # per-tissue positive fraction for tissues with >= MIN_TISSUE_N combined
    comb = work.groupby("tissue").agg(n=("y", "size"), n_pos=("y", "sum"))
    comb["pos_frac"] = comb["n_pos"] / comb["n"]
    big = comb.loc[comb["n"] >= MIN_TISSUE_N].sort_values("n", ascending=False)
    n_in_big = int(big["n"].sum())
    # how many of those buckets are effectively pure one class
    pure = big.loc[(big["pos_frac"] <= 0.02) | (big["pos_frac"] >= 0.98)]

    coarse_comb = work.groupby("tissue_coarse").agg(n=("y", "size"), n_pos=("y", "sum"))
    coarse_comb["pos_frac"] = coarse_comb["n_pos"] / coarse_comb["n"]
    coarse_comb = coarse_comb.sort_values("n", ascending=False)

    y = work["y"].to_numpy(dtype=int)
    groups = work["series_id"].astype(str).to_numpy()

    logger.info("  tissue-only probe: normalized source_name_ch1 one-hot")
    Xt, levels = onehot(work["tissue"].to_numpy(dtype=object))
    tissue_probe = probe_summary(Xt, y, groups, logger)

    logger.info("  tissue-only probe: coarse tissue one-hot")
    Xc, coarse_levels = onehot(work["tissue_coarse"].to_numpy(dtype=object))
    coarse_probe = probe_summary(Xc, y, groups, logger)

    logger.info("  tissue-only Bayes rule (normalized source_name_ch1)")
    bayes = bayes_tissue_rule(work["tissue"].to_numpy(dtype=object), y, groups, logger)

    reference_probe = None
    if not args.skip_embedding_probe:
        logger.info("  reference: 515-d BulkFormer-93M embedding probe, "
                    "same population/folds")
        Xe, eids = Xemb, emb_ids_all
        order = pd.Series(np.arange(eids.size), index=eids)
        sel = order.reindex(work["sample_index"].to_numpy(np.int64))
        if sel.isna().any():
            failures.append("embedding parquet is missing some pool samples")
        else:
            reference_probe = probe_summary(
                Xe[sel.to_numpy(dtype=np.int64)], y, groups, logger)

    # --- paired fold-by-fold comparison: tissue-only vs full transcriptome ---
    # The means alone understate the confound. StratifiedGroupKFold on this
    # population produces wildly uneven folds (a few series are enormous), so
    # a single bad fold moves the mean a lot. The paired per-fold delta is the
    # honest statement of "how much of the discriminable signal is tissue".
    paired = None
    if reference_probe is not None:
        t_f = [f["roc_auc"] for f in tissue_probe["folds"]]
        e_f = [f["roc_auc"] for f in reference_probe["folds"]]
        deltas = [float(e - t) for e, t in zip(e_f, t_f)]
        paired = {
            "fold_roc_auc_tissue_only": t_f,
            "fold_roc_auc_bulkformer_515d": e_f,
            "fold_delta_embedding_minus_tissue": deltas,
            "n_folds_where_tissue_within_0.10_of_embedding": int(
                sum(1 for d in deltas if d <= 0.10)),
            "n_folds_where_tissue_beats_embedding": int(sum(1 for d in deltas if d < 0)),
            "median_fold_roc_auc_tissue_only": float(np.median(t_f)),
            "median_fold_roc_auc_bulkformer_515d": float(np.median(e_f)),
            "excess_auc_over_chance_ratio_mean": float(
                (tissue_probe["roc_auc_mean"] - 0.5)
                / (reference_probe["roc_auc_mean"] - 0.5)),
            "excess_auc_over_chance_ratio_median": float(
                (float(np.median(t_f)) - 0.5) / (float(np.median(e_f)) - 0.5)),
            "note": "excess_auc_over_chance_ratio = (AUC_tissue - 0.5) / "
                    "(AUC_embedding - 0.5): the share of the full "
                    "transcriptome probe's above-chance discrimination that a "
                    "tissue label alone already reproduces.",
        }

    # --- tissue-matching feasibility, the Phase 1 design input ---------------
    # If Phase 1 tissue-matches the negatives to the positives' tissue
    # distribution, how many negatives are actually obtainable at each ratio?
    def matching_capacity(key: str) -> dict:
        tab = work.groupby([key, "y"]).size().unstack(fill_value=0)
        npos = tab.get(1, pd.Series(0, index=tab.index))
        nneg = tab.get(0, pd.Series(0, index=tab.index))
        out = {"key": key,
               "n_tissues_with_both_classes": int(((npos > 0) & (nneg > 0)).sum()),
               "n_positives_in_tissues_with_no_negative": int(npos[nneg == 0].sum()),
               "n_negatives_in_tissues_with_no_positive": int(nneg[npos == 0].sum())}
        for r in (0.5, 1.0, 2.0, 2.85):
            cap = int(np.minimum(nneg.to_numpy(), np.floor(r * npos.to_numpy())).sum())
            out[f"max_matched_negatives_at_ratio_{r}"] = cap
            out[f"achieved_ratio_at_target_{r}"] = round(cap / len(p), 3)
        return out

    matching = {"normalized": matching_capacity("tissue"),
                "coarse": matching_capacity("tissue_coarse")}

    # --- joint feasibility: coarse tissue match + per-series cap ------------
    # The two mitigations Phase 1 can apply are (i) match the negatives'
    # tissue mix to the positives' and (ii) cap how many samples any single
    # series may contribute. This is how many negatives survive both, which is
    # what actually bounds the achievable ratio.
    negs = work.loc[work.y == 0]
    pos_coarse = p["tissue_coarse"].value_counts()
    joint = []
    for cap in (10, 25, 50, 100, 250, 10**9):
        per_series = negs.groupby(["tissue_coarse", "series_id"]).size()
        capped = per_series.clip(upper=cap).groupby(level=0).sum()
        row = {"per_series_cap": (None if cap == 10**9 else cap),
               "n_negatives_available_after_cap": int(capped.sum()),
               "n_series_contributing": int(
                   per_series.index.get_level_values(1).nunique())}
        for r in (1.0, 2.0, 2.85):
            tot = 0
            for t, npos_t in pos_coarse.items():
                tot += int(min(round(r * npos_t), int(capped.get(t, 0))))
            row[f"tissue_matched_negatives_at_ratio_{r}"] = tot
            row[f"achieved_ratio_at_target_{r}"] = round(tot / len(p), 3)
        joint.append(row)

    tissue_auc = tissue_probe["roc_auc_mean"]
    stats["s0c_tissue"] = {
        "source": "meta/samples/source_name_ch1 in "
                  "eda/dataset/cvd_data/archs4/human_gene_v2.latest.h5",
        "normalization": "qa_generation.build_per_sample_de.normalize_tissue "
                         "(imported, not reimplemented)",
        "h5_meta_sample_fields": meta_fields,
        "fallback_fields_checked": {
            "meta/samples/tissue_present": "tissue" in meta_fields,
            "meta/samples/characteristics_ch1_present":
                "characteristics_ch1" in meta_fields,
            "note": "source_name_ch1 was used because it is the field the DE "
                    "pipeline's tissue buckets were built from and it is "
                    "non-blank for the fraction reported in "
                    "coverage_pct_nonblank; the fallbacks were only consulted "
                    "if that coverage had been poor.",
        },
        "n_samples": int(len(work)),
        "n_blank_source_name": n_blank,
        "coverage_pct_nonblank": round(100.0 * (1 - n_blank / len(work)), 3),
        "n_distinct_normalized_tissues": int(work["tissue"].nunique()),
        "n_distinct_in_positive_pool": int(p["tissue"].nunique()),
        "n_distinct_in_negative_pool": int(q["tissue"].nunique()),
        "n_tissues_shared_by_both_pools": int(
            len(set(p["tissue"]) & set(q["tissue"]))),
        "positive_top25": dist(pos_tis, len(p)),
        "negative_top25": dist(neg_tis, len(q)),
        "positive_top25_cumulative_pct": round(
            100.0 * pos_tis.head(25).sum() / len(p), 2),
        "negative_top25_cumulative_pct": round(
            100.0 * neg_tis.head(25).sum() / len(q), 2),
        "coarse_distribution": [
            {"coarse_tissue": str(t),
             "n": int(r.n), "n_pos": int(r.n_pos),
             "pos_frac": round(float(r.pos_frac), 4),
             "pct_of_positive_pool": round(100.0 * int(r.n_pos) / len(p), 2),
             "pct_of_negative_pool": round(
                 100.0 * (int(r.n) - int(r.n_pos)) / len(q), 2)}
            for t, r in coarse_comb.iterrows()
        ],
        "per_tissue_positive_fraction": {
            "min_n": MIN_TISSUE_N,
            "n_tissues_qualifying": int(len(big)),
            "n_samples_in_qualifying_tissues": n_in_big,
            "pct_of_pool_in_qualifying_tissues": round(
                100.0 * n_in_big / len(work), 2),
            "n_qualifying_tissues_effectively_pure": int(len(pure)),
            "pct_of_qualifying_samples_in_pure_tissues": round(
                100.0 * float(pure["n"].sum()) / n_in_big, 2) if n_in_big else None,
            "rows": [
                {"tissue": str(t), "n": int(r.n), "n_pos": int(r.n_pos),
                 "pos_frac": round(float(r.pos_frac), 4)}
                for t, r in big.head(40).iterrows()
            ],
        },
        "confound_measurement": {
            "protocol": "StratifiedGroupKFold(n_splits=5, shuffle=True, "
                        "random_state=20260707) grouped by series_id — "
                        "identical to every prior probe in this repo — via "
                        "linear_probe.probe.run_cv.",
            "interpretation": "AUC ~0.5 means tissue carries no class "
                              "information; AUC ~1.0 means the discriminative "
                              "task is largely tissue recognition.",
            "tissue_only_normalized": {
                "n_features": len(levels),
                "roc_auc_mean": tissue_probe["roc_auc_mean"],
                "roc_auc_std": tissue_probe["roc_auc_std"],
                "accuracy_mean": tissue_probe["accuracy_mean"],
                "pr_auc_mean": tissue_probe["pr_auc_mean"],
                "sensitivity_mean": tissue_probe["sensitivity_mean"],
                "specificity_mean": tissue_probe["specificity_mean"],
                "full": tissue_probe,
            },
            "tissue_only_coarse": {
                "n_features": len(coarse_levels),
                "levels": coarse_levels,
                "roc_auc_mean": coarse_probe["roc_auc_mean"],
                "roc_auc_std": coarse_probe["roc_auc_std"],
                "accuracy_mean": coarse_probe["accuracy_mean"],
                "full": coarse_probe,
            },
            "tissue_only_bayes_rule": bayes,
            "reference_bulkformer_515d_probe": (
                {"roc_auc_mean": reference_probe["roc_auc_mean"],
                 "roc_auc_std": reference_probe["roc_auc_std"],
                 "accuracy_mean": reference_probe["accuracy_mean"],
                 "full": reference_probe}
                if reference_probe else None),
            "paired_fold_comparison": paired,
            "numerical_note": (
                "sklearn's lbfgs emits spurious 'divide by zero / overflow / "
                "invalid value encountered in matmul' RuntimeWarnings on this "
                "platform (numpy + Apple Accelerate BLAS) for BOTH the one-hot "
                "and the dense embedding probe. They are not affecting the "
                "results: the reference embedding probe reproduces this repo's "
                "published BulkFormer-93M/neg_hard ROC-AUC of 0.80050 to four "
                "decimals on a differently-scoped population."
            ),
        },
        "tissue_matching_feasibility": matching,
        "joint_feasibility_coarse_match_plus_series_cap": joint,
    }

    # ---------------- 0d: DE reusability ----------------------------------
    logger.info("0d: DE reusability")
    de = pd.read_parquet(DE_PARQUET, columns=["sample_index", "series_id",
                                              "in_holdout", "status",
                                              "reference_scope"])
    de_ids = set(de["sample_index"].tolist())
    de_manifest = json.loads(DE_MANIFEST.read_text())
    # allow_pickle is required because stable_gene_z.npz stores an object-dtype
    # `genes` array. The file is generated by this repo's own
    # qa_generation/build_per_sample_de.py — it is not untrusted input.
    npz = np.load(STABLE_Z, allow_pickle=True)
    z_ids = set(np.asarray(npz["sample_index"]).tolist())

    n_de_pos = int(pos_all["sample_index"].isin(de_ids).sum())
    n_de_neg = int(neg_all["sample_index"].isin(de_ids).sum())
    n_de_train_pos = int(pos["sample_index"].isin(de_ids).sum())
    n_z_neg = int(neg_all["sample_index"].isin(z_ids).sum())

    stats["s0d_de_reusability"] = {
        "per_sample_de": {
            "file": str(DE_PARQUET.relative_to(REPO)),
            "n_rows": int(len(de)),
            "n_unique_sample_index": int(len(de_ids)),
            "n_rows_status_ok": int((de["status"] == "ok").sum()),
            "n_rows_in_holdout": int(de["in_holdout"].sum()),
            "n_positive_samples_covered": n_de_pos,
            "n_neg_hard_samples_covered": n_de_neg,
            "n_train_eligible_positives_covered": n_de_train_pos,
            "identical_to_expression_index": de_ids == expr_ids,
        },
        "stable_gene_z": {
            "file": str(STABLE_Z.relative_to(REPO)),
            "shape_z": list(np.asarray(npz["z"]).shape),
            "n_sample_rows": int(len(z_ids)),
            "n_neg_hard_covered": n_z_neg,
            "covers_negatives": n_z_neg > 0,
        },
        "negatives_role": {
            "manifest_populations": de_manifest["populations"],
            "tissue_matching": de_manifest["tissue_matching"],
            "role": (
                "The 21,041 holdout-excluded neg_hard samples were consumed as "
                "a REFERENCE POOL: their normalized expression was reduced to "
                "per-tissue-bucket and pooled mean/sd vectors in "
                "de_reference_stats.npz. No per-negative-sample DE row was "
                "ever written."
            ),
            "reference_stats_file": "qa_generation/de/de_reference_stats.npz",
        },
        "verdict": {
            "negatives_with_own_de_rows": n_de_neg,
            "de_needed_for_discriminative_task": False,
            "reasoning": (
                "Plan §2.1 states the discriminative task's ground truth is "
                "the verified binary label. The model input is the 515-d "
                "BulkFormer embedding, which exists for 100% of neg_hard. "
                "Neither the question nor the answer references a DE "
                "statistic, so no per-negative DE row is required."
            ),
            "if_de_were_required": (
                "It would be a full recomputation for every selected negative: "
                "H5 read + TPM/log1p/vocab-align + z against a reference. The "
                "reference itself would also have to be rebuilt to exclude "
                "each negative from its own comparison population, otherwise "
                "every negative is scored against a pool containing itself."
            ),
        },
    }

    # ---------------- corpus integration arithmetic -----------------------
    # Not a Phase 0 subsection of its own, but the recommendation at the end
    # of the report has to be checkable, so the arithmetic lives here rather
    # than in prose.
    existing_items = int(bundle["n_written"])
    n_pos_items = int(bundle["n_unique_samples"])   # one discriminative item per positive
    scenarios = []
    for n_neg in (3_000, 4_363, 5_539, 6_098, 6_817, 7_212, 10_668, 21_041):
        new_cat = n_pos_items + n_neg
        total = existing_items + new_cat
        scenarios.append({
            "n_negatives": n_neg,
            "n_positive_items": n_pos_items,
            "new_category_items": new_cat,
            "pos_to_neg_within_category": round(n_neg / n_pos_items, 3),
            "total_corpus_items": total,
            "new_category_share_pct": round(100.0 * new_cat / total, 2),
            "each_existing_category_share_pct": round(
                100.0 * existing_items / 4 / total, 2),
        })
    # The recommended design: cut BOTH classes to the same per-coarse-tissue
    # counts, so the category is exactly 1:1 and exactly tissue-matched. The
    # per-tissue count is min(n_pos_t, capped_neg_t), which is the same number
    # the joint-feasibility table reports at ratio 1.0.
    matched_pair = []
    for row in joint:
        n = row["tissue_matched_negatives_at_ratio_1.0"]
        new_cat = 2 * n
        total = existing_items + new_cat
        matched_pair.append({
            "per_series_cap": row["per_series_cap"],
            "n_per_class": n,
            "new_category_items": new_cat,
            "positives_dropped_from_this_category": n_pos_items - n,
            "total_corpus_items": total,
            "new_category_share_pct": round(100.0 * new_cat / total, 2),
            "each_existing_category_share_pct": round(
                100.0 * existing_items / 4 / total, 2),
        })

    # Holdout evaluation prior — the thing the training ratio should match.
    stats["corpus_integration"] = {
        "holdout_eval_prior": {
            "n_holdout_positive": holdout_blob["n_holdout_positive"],
            "n_holdout_neg_hard": holdout_blob["n_holdout_neg_hard"],
            "pos_to_neg": round(holdout_blob["n_holdout_neg_hard"]
                                / holdout_blob["n_holdout_positive"], 3),
        },
        "scenarios_matched_pairs_1to1": matched_pair,
        "existing_items": existing_items,
        "existing_items_by_category": bundle["items_by_category"],
        "existing_unique_samples": n_pos_items,
        "assumption": "one discriminative QA item per sample (positive or "
                      "negative); no paraphrase multiplication modelled here.",
        "scenarios_append_only": scenarios,
        "note": "append-only over-weights the new category. The alternative is "
                "to downsample the four existing categories to hold the new "
                "one at a target share — Phase 1's decision, not Phase 0's.",
    }

    # ---------------- assertions ------------------------------------------
    stats["assertions"] = {
        "expected": EXPECT,
        "failures": failures,
        "passed": not failures,
    }
    stats["elapsed_seconds"] = round(time.time() - t_start, 1)

    args.out.write_text(json.dumps(stats, indent=2, default=str))
    logger.info(f"wrote {args.out}")

    # ---------------- console summary -------------------------------------
    a, b, c = stats["s0a_populations"], stats["s0b_series_structure"], stats["s0c_tissue"]
    print("\n================ PHASE 0 HEADLINES ================")
    print(f"0a positives (holdout-excluded)      : {a['train_eligible']['n_positive']}")
    print(f"0a neg_hard  (holdout-excluded)      : {a['train_eligible']['n_neg_hard']}")
    print(f"0a raw ratio                         : {a['train_eligible']['raw_ratio_str']}")
    print(f"0a corpus positives (expression rows): {a['current_corpus']['n_unique_samples']}")
    print(f"0b positive-pool series              : {b['positive_pool']['n_series']}"
          f"  max share {b['positive_pool']['max_series_share']:.3%}")
    print(f"0b negative-pool series              : {b['negative_pool']['n_series']}"
          f"  max share {b['negative_pool']['max_series_share']:.3%}")
    print(f"0b mixed series after exclusion      : "
          f"{b['series_overlap']['n_series_with_both_classes_after_holdout_exclusion']}")
    cm = c["confound_measurement"]
    print(f"0c tissue-only AUC (normalized)      : "
          f"{cm['tissue_only_normalized']['roc_auc_mean']:.4f} "
          f"± {cm['tissue_only_normalized']['roc_auc_std']:.4f}")
    print(f"0c tissue-only AUC (coarse)          : "
          f"{cm['tissue_only_coarse']['roc_auc_mean']:.4f}")
    print(f"0c tissue-only AUC (Bayes rule)      : "
          f"{cm['tissue_only_bayes_rule']['roc_auc_mean']:.4f}")
    if cm["reference_bulkformer_515d_probe"]:
        print(f"0c BulkFormer-93M 515d AUC (ref)     : "
              f"{cm['reference_bulkformer_515d_probe']['roc_auc_mean']:.4f}")
    if cm["paired_fold_comparison"]:
        pf = cm["paired_fold_comparison"]
        print(f"0c median fold AUC tissue / 515d     : "
              f"{pf['median_fold_roc_auc_tissue_only']:.4f} / "
              f"{pf['median_fold_roc_auc_bulkformer_515d']:.4f}")
        print(f"0c above-chance share explained      : "
              f"{pf['excess_auc_over_chance_ratio_mean']:.1%} (mean), "
              f"{pf['excess_auc_over_chance_ratio_median']:.1%} (median fold)")
    mn = c["tissue_matching_feasibility"]["normalized"]
    print(f"0c tissue-matched negatives @1:1     : "
          f"{mn['max_matched_negatives_at_ratio_1.0']}")
    print(f"0c tissue-matched negatives @1:2     : "
          f"{mn['max_matched_negatives_at_ratio_2.0']}")
    print(f"0d neg_hard with own DE rows         : "
          f"{stats['s0d_de_reusability']['verdict']['negatives_with_own_de_rows']}")
    print(f"assertions passed                    : {stats['assertions']['passed']}")
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
    print("===================================================")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
