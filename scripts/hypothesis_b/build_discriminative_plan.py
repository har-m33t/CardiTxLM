"""Phase 1/2 — select the samples for the new binary discriminative category.

WHAT PHASE 0 FOUND, AND WHY THIS SCRIPT LOOKS THE WAY IT DOES
-------------------------------------------------------------
Two structural facts drive every choice below (scripts/hypothesis_b/
phase0_stats.json, and hypothesis_b_prelim_report.md in prose):

1. TISSUE IS MOST OF THE SIGNAL. A classifier given ONLY the normalized
   `source_name_ch1` string scores ROC-AUC 0.691 +/- 0.168 under the repo's
   standard grouped 5-fold CV, against 0.800 for the full 515-d BulkFormer
   embedding on the same folds. A tissue string alone reproduces ~64% of the
   above-chance signal, and beats the transcriptome outright on fold 0. Train a
   discriminative task on an unmatched pool and most of what the model can
   learn is tissue recognition.

   MITIGATION: exact per-coarse-tissue-bucket 1:1 matching. Not "approximately
   balanced" — within every bucket the counts are equal by construction, so
   tissue carries EXACTLY zero information about the label in this category.
   That is checkable, and `--assert-tissue-uninformative` checks it.

2. SERIES IS A PERFECT PREDICTOR, AND CANNOT BE FIXED HERE. Zero non-holdout
   series contain both classes (388 positive-only, 1,174 negative-only, no
   overlap). This is structural: the holdout was DEFINED as every mixed-class
   series, so the mixed ones are all on the other side of the wall. A model can
   therefore score well by recognizing batch signature and learning no biology.

   NO SAMPLING SCHEME CAN REMOVE THIS. It is a property of the corpus. What
   this script does is make the shortcut expensive rather than free: a
   per-series cap forces the label to be spread over many series, so
   memorizing signature means memorizing hundreds of them. The shortcut is
   DETECTED, not prevented, and detection happens at evaluation time via
   within-series AUC on the 92 mixed holdout series
   (`eval_binary_comparison/run_binary_cvd_eval.py`). Read that number before
   believing any headline from this category.

RATIO: 1:1, not the raw 1:2.85.
   - The holdout evaluation prior is 1,341:1,266 ~= 1:0.94, and Phase 4's
     forced-choice log-probability extraction is prior-sensitive; a training
     prior far from the evaluation prior shifts the operating point for
     reasons that have nothing to do with representation quality.
   - 1:2.85 is a curation artifact (how many samples happened to be labelled
     tissue-only-unconfirmed), not a disease base rate. It is not "more
     realistic" — it is differently arbitrary.
   - A true CVD base rate (~5-10%) would make "always answer no" nearly
     optimal, reintroducing the degenerate-answer failure the last
     regeneration existed to remove.

PER-SERIES CAP: 50, applied to BOTH classes.
   Phase 0 measured the knee: cap 50 cuts GSE262419 from 4,395 negatives to 50
   while costing only 719 matched negatives against no cap (6,098 vs 6,817);
   cap 10 costs 28%. Phase 0 evaluated the cap on negatives only, but the
   positive pool is comparably concentrated (top series 15.85%, effective
   series count 28.4 against a nominal 388), so it is applied symmetrically —
   an uncapped positive pool would leave exactly the same shortcut on the other
   side of the label.

Output: `scripts/hypothesis_b/discriminative_plan.json` (the selection) and
`discriminative_plan_stats.json` (the audit).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import logging  # noqa: E402

from scripts.hypothesis_b.phase0_analysis import coarse_tissue  # noqa: E402
from qa_generation.build_per_sample_de import (  # noqa: E402
    normalize_tissue,
    read_source_names,
)

LABELS = REPO / "linear_probe/probe_sample_labels.parquet"
HOLDOUT = REPO / "data/cvd_transcriptome/holdout_series.json"
EXPR_INDEX = REPO / "qa_generation/bulkformer_input/bulkformer_sample_index.npy"
CACHE = REPO / "data/cvd_transcriptome/embeddings_encoded"
OUT = REPO / "scripts/hypothesis_b/discriminative_plan.json"
STATS = REPO / "scripts/hypothesis_b/discriminative_plan_stats.json"

SEED = 20260901
PER_SERIES_CAP = 50


def cap_by_series(df: pd.DataFrame, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    """At most `cap` samples from any one GEO series, chosen at random within it."""
    keep = []
    for _, g in df.groupby("series_id", sort=True):
        idx = g.index.to_numpy()
        if idx.size > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.append(idx)
    return df.loc[np.sort(np.concatenate(keep))]


def series_spread_sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Take `n` rows, maximizing the number of distinct series represented.

    Round-robin across series rather than a flat random draw: a flat draw
    reproduces the pool's own concentration, which is the thing being diluted.
    Series order and within-series order are both shuffled so the choice is not
    an artifact of accession sort order.
    """
    if n >= len(df):
        return df
    by_series = defaultdict(list)
    for i, s in zip(df.index.to_numpy(), df["series_id"].to_numpy()):
        by_series[s].append(i)
    order = list(by_series)
    rng.shuffle(order)
    for s in order:
        rng.shuffle(by_series[s])

    picked, r = [], 0
    while len(picked) < n:
        progressed = False
        for s in order:
            if r < len(by_series[s]):
                picked.append(by_series[s][r])
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break
        r += 1
    return df.loc[np.sort(np.asarray(picked))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=PER_SERIES_CAP)
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="negatives per positive; 1.0 == exact per-bucket balance")
    ap.add_argument("--positives", choices=["with-expression", "all-eligible"],
                    default="with-expression",
                    help="'with-expression' (7,212) keeps the sample set aligned "
                         "with the other four categories; 'all-eligible' (7,384) "
                         "adds 172 positives that have no DE row")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    lab = pd.read_parquet(LABELS)
    hold = json.load(open(HOLDOUT))
    hold_series = set(hold["holdout_series"])

    # --- candidate pools ---------------------------------------------------
    lab = lab[~lab.series_id.isin(hold_series)]
    pos = lab[lab.is_positive].copy()
    neg = lab[lab.is_neg_hard].copy()
    if args.positives == "with-expression":
        expr = set(np.load(EXPR_INDEX).tolist())
        pos = pos[pos.sample_index.isin(expr)]
    n_pos0, n_neg0 = len(pos), len(neg)

    # --- tissue ------------------------------------------------------------
    both = pd.concat([pos, neg])
    # Same field, same normalizer, same coarse grouping Phase 0 measured the
    # confound with — reused by import so the matching cannot drift from the
    # measurement that justified it.
    raw = read_source_names(both.sample_index.to_numpy(np.int64),
                            logging.getLogger("discrim_plan"))
    both["tissue_coarse"] = [coarse_tissue(normalize_tissue(t)) for t in raw]
    pos = both[both.is_positive]
    neg = both[both.is_neg_hard]

    # --- per-series cap, applied to BOTH classes ---------------------------
    pos_c = cap_by_series(pos, args.cap, rng)
    neg_c = cap_by_series(neg, args.cap, rng)

    # --- exact per-bucket matching ----------------------------------------
    rows, buckets = [], []
    for b in sorted(set(pos_c.tissue_coarse) | set(neg_c.tissue_coarse)):
        p = pos_c[pos_c.tissue_coarse == b]
        q = neg_c[neg_c.tissue_coarse == b]
        take_p = min(len(p), int(round(len(q) / args.ratio))) if len(q) else 0
        take_n = min(len(q), int(round(take_p * args.ratio)))
        take_p = min(take_p, int(round(take_n / args.ratio))) if take_n else 0
        buckets.append({"tissue_coarse": b, "n_pos_available": len(p),
                        "n_neg_available": len(q), "n_pos_taken": take_p,
                        "n_neg_taken": take_n,
                        "dropped_reason": None if take_p else
                        ("no negatives in bucket" if not len(q) else "no positives in bucket")})
        if not take_p:
            continue
        for sub, lb in ((series_spread_sample(p, take_p, rng), 1),
                        (series_spread_sample(q, take_n, rng), 0)):
            for r in sub.itertuples():
                rows.append({"sample_index": int(r.sample_index),
                             "geo_accession": r.geo_accession,
                             "series_id": r.series_id,
                             "tissue_coarse": r.tissue_coarse,
                             "label": lb})

    sel = pd.DataFrame(rows)

    # --- assertions: each guards a way this could be silently wrong --------
    a = {}
    a["no_holdout_series"] = bool(not set(sel.series_id) & hold_series)
    a["no_duplicate_samples"] = bool(sel.sample_index.is_unique)
    a["all_have_encoded_vector"] = bool(
        all((CACHE / f"{g}.npy").exists() for g in sel.geo_accession))
    a["series_cap_respected"] = bool(sel.series_id.value_counts().max() <= args.cap)
    # The one that matters most: after exact matching, the positive rate must be
    # identical in every bucket, so tissue carries zero information about the
    # label. If this is false the confound survived the matching.
    rate = sel.groupby("tissue_coarse")["label"].mean()
    a["tissue_uninformative"] = bool(np.allclose(rate.to_numpy(),
                                                 1.0 / (1.0 + args.ratio), atol=1e-9))
    for k, v in a.items():
        assert v, f"ASSERTION FAILED: {k}"

    # Empirical confirmation, measured rather than asserted from construction.
    #
    # TWO measurements, because one alone is misleading here. Grouped folds put
    # whole series on one side of the split, and every non-holdout series is
    # single-class — so a validation fold can receive the positive-series of one
    # tissue bucket and the negative-series of another, while the training fold
    # saw the complementary pattern. The rule then learns per-bucket rates that
    # are systematically INVERTED on validation, which drives AUC below 0.5
    # without any residual tissue confound existing. Ungrouped folds break that
    # interaction and isolate the tissue effect proper, which exact per-bucket
    # matching should pin at ~0.5.
    #
    # Series-only AUC is reported alongside as the honest upper bound on what a
    # shortcut-taking model could exploit. It is expected to be ~1.0 and is NOT
    # fixable by sampling (see the module docstring).
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    def onehot_auc(keys, grouped: bool) -> float:
        lv = sorted(set(keys))
        Xh = np.zeros((len(sel), len(lv)), dtype=np.float32)
        Xh[np.arange(len(sel)), [lv.index(t) for t in keys]] = 1.0
        yv = sel.label.to_numpy()
        splitter = (StratifiedGroupKFold(5, shuffle=True, random_state=20260707)
                    if grouped else StratifiedKFold(5, shuffle=True, random_state=20260707))
        args_ = (Xh, yv, sel.series_id.to_numpy()) if grouped else (Xh, yv)
        out = []
        for tr, va in splitter.split(*args_):
            if len(set(yv[va])) < 2:
                continue
            m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xh[tr], yv[tr])
            out.append(roc_auc_score(yv[va], m.predict_proba(Xh[va])[:, 1]))
        return float(np.mean(out)) if out else float("nan")

    tissue_auc = onehot_auc(sel.tissue_coarse.to_numpy(), grouped=True)
    tissue_auc_ungrouped = onehot_auc(sel.tissue_coarse.to_numpy(), grouped=False)
    series_auc_ungrouped = onehot_auc(sel.series_id.to_numpy(), grouped=False)

    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED, "per_series_cap": args.cap, "ratio_neg_per_pos": args.ratio,
        "positive_pool": args.positives,
        "n_positive": int((sel.label == 1).sum()),
        "n_negative": int((sel.label == 0).sum()),
        "samples": sel.to_dict("records"),
    }, indent=2) + "\n")

    stats = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED, "per_series_cap": args.cap, "ratio_neg_per_pos": args.ratio,
        "candidate_pools": {"n_positive": n_pos0, "n_negative": n_neg0,
                            "positive_pool_definition": args.positives},
        "after_series_cap": {"n_positive": len(pos_c), "n_negative": len(neg_c)},
        "selected": {"n_positive": int((sel.label == 1).sum()),
                     "n_negative": int((sel.label == 0).sum()),
                     "n_total": len(sel),
                     "n_series": int(sel.series_id.nunique()),
                     "n_series_positive": int(sel[sel.label == 1].series_id.nunique()),
                     "n_series_negative": int(sel[sel.label == 0].series_id.nunique()),
                     "max_series_share_pct": round(
                         100 * sel.series_id.value_counts().iloc[0] / len(sel), 3),
                     "n_tissue_buckets": int(sel.tissue_coarse.nunique())},
        "buckets": buckets,
        "tissue_confound": {
            "tissue_only_auc_before_matching_grouped": 0.6914189068779029,
            "tissue_only_auc_after_matching_grouped": tissue_auc,
            "tissue_only_auc_after_matching_ungrouped": tissue_auc_ungrouped,
            "per_bucket_positive_rate_is_constant": a["tissue_uninformative"],
            "note": ("Ungrouped is the measurement that answers 'did matching "
                     "remove the tissue shortcut' — ~0.5 means yes. The grouped "
                     "number can fall BELOW 0.5 without any residual confound: "
                     "single-class series + whole-series folds make per-bucket "
                     "rates invert between train and validation. Neither is a "
                     "defect; they answer different questions."),
        },
        "series_shortcut_upper_bound": {
            "series_only_auc_ungrouped": series_auc_ungrouped,
            "note": ("What a model could score using series identity alone. "
                     "Expected ~1.0 and irreducible: no non-holdout series is "
                     "mixed. Measured ungrouped on purpose — grouped folds hide "
                     "the shortcut by construction, which is exactly why the "
                     "real check is within-series AUC on the mixed holdout."),
        },
        "series_shortcut": {
            "n_mixed_series_in_selection": int(sum(
                sel.groupby("series_id")["label"].nunique() > 1)),
            "note": ("Expected to be 0 and NOT a defect of this selection: no "
                     "non-holdout series contains both classes, because the "
                     "holdout was defined as every mixed series. Series identity "
                     "therefore remains a perfect predictor within training data. "
                     "This is detected at eval time by within-series AUC on the 92 "
                     "mixed holdout series, never claimed to be fixed here."),
        },
        "assertions": a,
    }
    STATS.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"selected {len(sel):,} = {(sel.label==1).sum():,} pos + "
          f"{(sel.label==0).sum():,} neg over {sel.series_id.nunique():,} series, "
          f"{sel.tissue_coarse.nunique()} tissue buckets")
    print(f"max single-series share: {stats['selected']['max_series_share_pct']}%")
    print(f"tissue-only AUC: {0.6914:.4f} (pool) -> {tissue_auc:.4f} (selection)")
    print("assertions: " + ", ".join(f"{k}={v}" for k, v in a.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
