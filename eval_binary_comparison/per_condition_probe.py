"""per_condition_probe.py — per-condition AUROC for the linear-probe baselines.

Replicates the per-condition table structure: for each named cardiovascular
condition, score THAT condition's positives against the SAME negative pool the
binary probe used, with the same folds, same model and same metric code. Macro
AUROC is the unweighted mean across conditions.

Why this structure: the existing probe pools all 8,725 positives into one
CVD-vs-control run. Splitting those positives by `cvd_subtype` and scoring each
against the same negatives yields per-condition columns whose mean lands near the
pooled binary figure — which is the relationship the target table shows.

Everything that can be reused IS reused, so the baseline rows are produced by the
probe's own code path:
  * `linear_probe.probe.run_cv`      — folds, per-fold skip floor, pipeline, fit
  * `linear_probe.probe._fold_metrics` — metrics
  * StratifiedGroupKFold(5, shuffle=True, random_state=seed) grouped by series_id

Run:
    python -m eval_binary_comparison.per_condition_probe \
        --variants BulkFormer-37M BulkFormer-50M BulkFormer-93M \
        --pool neg_hard --out eval_binary_comparison/per_condition_probe.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

CONDITIONS = [
    ("heart_failure",          "Heart failure"),
    ("arrhythmia_afib",        "Arrhythmia / AFib"),
    ("coronary_artery_disease", "Coronary artery dis."),
    ("cardiomyopathy_other",   "Cardiomyo. (other)"),
    ("hypertension",           "Hypertension"),
]
UNRESOLVED = "disease_matched_subtype_unresolved"


def load_variant(path: Path):
    import re
    import pyarrow.parquet as pq
    t = pq.read_table(path).to_pydict()
    # Match the full e<digits> name, NOT startswith("e0"): the latter silently
    # truncates a 4096-d feature block to e0000-e0999. Sort by index, not by
    # string, so widths above 10000 stay ordered.
    ecols = sorted((c for c in t if re.fullmatch(r"e\d+", c)),
                   key=lambda c: int(c[1:]))
    n = len(t["geo_accession"])
    X = np.empty((n, len(ecols)), dtype=np.float32)
    for j, c in enumerate(ecols):
        X[:, j] = np.asarray(t[c], dtype=np.float32)
    meta = {k: np.asarray(t[k]) for k in
            ("geo_accession", "series_id", "cvd_subtype", "is_positive",
             "is_neg_hard", "is_neg_whole_corpus")}
    return X, meta


def run_condition(X, meta, cond_key, pool, k_folds, seed, logger):
    """Positives = samples whose cvd_subtype is this condition.
    Negatives = the same pool the binary probe used."""
    neg_flag = "is_neg_hard" if pool == "neg_hard" else "is_neg_whole_corpus"
    is_pos = meta["is_positive"].astype(bool) & (meta["cvd_subtype"] == cond_key)
    is_neg = meta[neg_flag].astype(bool)
    sel = np.where(is_pos | is_neg)[0]
    y = is_pos[sel].astype(int)
    groups = meta["series_id"][sel].astype(str)

    from linear_probe.probe import run_cv
    folds = run_cv(X[sel], y, groups, k_folds, seed, logger)
    ran = [f for f in folds if not f.skipped]
    agg = {}
    for key in ("roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity", "f1", "brier"):
        vals = [getattr(f, key) for f in ran if getattr(f, key) is not None]
        agg[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        agg[f"{key}_std"] = float(np.std(vals)) if vals else None
    agg.update(n_samples=int(len(sel)), n_positive=int(y.sum()),
               n_negative=int((y == 0).sum()),
               n_series=int(np.unique(groups).size),
               n_folds_ran=len(ran), n_folds_skipped=len(folds) - len(ran))
    return agg


def run_pooled_breakdown(X, meta, pool, k_folds, seed):
    """ONE probe trained on ALL positives vs the negative pool (exactly the
    existing binary run), then AUROC computed separately per condition on each
    validation fold: that condition's positives in the fold + all negatives in
    the fold. This is a per-condition BREAKDOWN of a single model, not five
    models. Its macro lands near the pooled binary AUROC, which is the
    relationship the target table shows."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from linear_probe.probe import _fold_metrics

    neg_flag = "is_neg_hard" if pool == "neg_hard" else "is_neg_whole_corpus"
    is_pos = meta["is_positive"].astype(bool)
    is_neg = meta[neg_flag].astype(bool)
    sel = np.where(is_pos | is_neg)[0]
    X_s, y = X[sel], is_pos[sel].astype(int)
    groups = meta["series_id"][sel].astype(str)
    subtype = meta["cvd_subtype"][sel]

    skf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    per_cond = {k: [] for k, _ in CONDITIONS}
    pooled = []
    for tr, va in skf.split(X_s, y, groups):
        pipe = Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, solver="lbfgs",
                                                    class_weight="balanced",
                                                    random_state=seed))])
        pipe.fit(X_s[tr], y[tr])
        p = pipe.predict_proba(X_s[va])[:, 1]
        yv, sv = y[va], subtype[va]
        if len(set(yv.tolist())) == 2:
            pooled.append(_fold_metrics(yv, p))
        for key, _ in CONDITIONS:
            m = (yv == 0) | (sv == key)          # all negatives + this condition
            if m.sum() == 0:
                continue
            yy, pp = yv[m], p[m]
            if len(set(yy.tolist())) < 2 or yy.sum() < 5:
                continue
            per_cond[key].append(_fold_metrics(yy, pp))

    def agg(rows):
        if not rows:
            return {"roc_auc_mean": None, "roc_auc_std": None, "n_folds": 0}
        return {"roc_auc_mean": float(np.mean([r["roc_auc"] for r in rows])),
                "roc_auc_std": float(np.std([r["roc_auc"] for r in rows])),
                "pr_auc_mean": float(np.mean([r["pr_auc"] for r in rows])),
                "n_folds": len(rows)}

    out = {k: agg(v) for k, v in per_cond.items()}
    aur = [v["roc_auc_mean"] for v in out.values() if v["roc_auc_mean"] is not None]
    return out, (float(np.mean(aur)) if aur else None), agg(pooled)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["BulkFormer-37M", "BulkFormer-50M", "BulkFormer-93M"])
    ap.add_argument("--emb-dir", default="linear_probe/embeddings")
    ap.add_argument("--pool", default="neg_hard", choices=["neg_hard", "neg_whole_corpus"])
    ap.add_argument("--k-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--out", default="eval_binary_comparison/per_condition_probe.json")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger = logging.getLogger("per_condition")

    out = {"pool": args.pool, "seed": args.seed, "k_folds": args.k_folds,
           "conditions": [c for c, _ in CONDITIONS], "variants": {}}

    for variant in args.variants:
        p = Path(args.emb_dir) / f"embeddings_{variant}.parquet"
        if not p.exists():
            print(f"[skip] {variant}: {p} not present locally")
            continue
        print(f"\n=== {variant} ===", flush=True)
        X, meta = load_variant(p)
        per_cond, aurocs = {}, []
        for key, label in CONDITIONS:
            agg = run_condition(X, meta, key, args.pool, args.k_folds, args.seed, logger)
            per_cond[key] = agg
            a = agg["roc_auc_mean"]
            if a is not None:
                aurocs.append(a)
            print(f"  {label:22s} n_pos={agg['n_positive']:5d} folds={agg['n_folds_ran']}/{args.k_folds} "
                  f"ROC-AUC={a:.4f}±{agg['roc_auc_std']:.4f}" if a is not None
                  else f"  {label:22s} n_pos={agg['n_positive']:5d} ALL FOLDS SKIPPED")
        macro = float(np.mean(aurocs)) if aurocs else None
        out["variants"][variant] = {"per_condition": per_cond, "macro_auroc": macro,
                                    "n_conditions_scored": len(aurocs)}
        print(f"  {'MACRO AUROC':22s} {macro:.4f}  ({len(aurocs)}/{len(CONDITIONS)} conditions)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
