"""Phase 1c / 4c — broad multi-label probing, before and after the fix.

Fits a probe on one feature set against every label in
`linear_probe/multilabel_labels.parquet` and writes one row per (feature set,
label). Run it once on the PRE-fix LLM latents and once on the retrained ones;
the two tables are directly comparable because folds, estimator and label set
are identical.

This measures general representation quality rather than CVD-subtype accuracy
specifically. The question it answers is whether correcting the Stage-2 data
changed what the LLM's latent space encodes at all, not just whether it got
better at the one task it was tuned on.

TWO LABELS ARE NOT SCIENTIFIC RESULTS. `platform` and `instrument` are
technical controls: they measure how much pure sequencing-batch signal the
representation carries. A probe that separates them well is not a good result,
and the output marks them `technical_control` so they cannot be read as one.
`is_bulk` is skipped entirely — the probe population was already bulk-filtered
upstream, so it has a single class and nothing to predict.

Multiclass labels use macro-averaged one-vs-rest ROC-AUC. Grouping is by
series_id, as everywhere else in this project: without it a probe can memorise
a study rather than learn the label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_binary_comparison.embedding_io import load_embeddings
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "linear_probe/embeddings"
LABELS = REPO / "linear_probe/multilabel_labels.parquet"
MANIFEST = REPO / "linear_probe/multilabel_labels_manifest.json"

SEED = 20260707
K_FOLDS = 5

#: PCA width applied INSIDE the CV pipeline before the probe.
#:
#: Not an optimisation — a correctness fix. A multinomial logistic regression
#: over 28 tissue classes on 4096 raw dimensions does not converge at
#: max_iter=1000 (measured: sustained ConvergenceWarning), and a non-converged
#: fit gives an arbitrary number, not a conservative one. It is also
#: prohibitively slow: the 1000-d version of this run took ~50 minutes, and
#: 4096-d would be several hours per side.
#:
#: The reduction is applied IDENTICALLY to the before and after feature sets, so
#: the comparison this file exists to make stays fair — both sides are probed
#: through the same bottleneck. It does mean the absolute numbers describe the
#: top-256 principal subspace rather than the full representation, which the
#: report states.
PCA_COMPONENTS = 256
#: A class needs at least this many members per fold to be scorable at all.
MIN_CLASS_SUPPORT = K_FOLDS * 2


def load_features(path: Path):
    """Delegates to embedding_io, which asserts no dimension was dropped.

    Do NOT inline a `startswith("e0")` filter here — that matches only
    e0000..e0999 and silently truncates a 4096-d latent to 1000. See
    eval_binary_comparison/embedding_io.py.
    """
    return load_embeddings(path)


def probe_label(X, y_raw, groups, name: str) -> dict | None:
    classes, counts = np.unique(y_raw, return_counts=True)
    keep_classes = set(classes[counts >= MIN_CLASS_SUPPORT])
    if len(keep_classes) < 2:
        return {"label": name, "skipped": "fewer than two classes with support"}

    mask = np.isin(y_raw, list(keep_classes))
    X, y_raw, groups = X[mask], y_raw[mask], groups[mask]
    classes = np.unique(y_raw)
    y = np.searchsorted(classes, y_raw)

    skf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    aucs = []
    for tr, va in skf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        # PCA is a pipeline step, so it is fitted on the training fold only and
        # never sees validation data.
        k = min(PCA_COMPONENTS, X.shape[1], len(tr) - 1)
        est = make_pipeline(
            StandardScaler(),
            PCA(n_components=k, random_state=SEED),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=SEED),
        )
        est.fit(X[tr], y[tr])
        p = est.predict_proba(X[va])
        present = np.unique(y[va])
        try:
            if len(classes) == 2:
                auc = roc_auc_score(y[va], p[:, 1])
            else:
                # Restrict to classes present in this validation fold; scoring
                # against an absent class is undefined, not zero.
                cols = [list(est.classes_).index(c) for c in present]
                pp = p[:, cols]
                pp = pp / pp.sum(axis=1, keepdims=True)
                auc = roc_auc_score(y[va], pp, multi_class="ovr",
                                    average="macro", labels=present)
            aucs.append(float(auc))
        except ValueError:
            continue

    if not aucs:
        return {"label": name, "skipped": "no scorable folds"}
    return {
        "label": name,
        "n": int(len(y)),
        "n_classes": int(len(classes)),
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "n_folds_scored": len(aucs),
        "pca_components": int(min(PCA_COMPONENTS, X.shape[1])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--name", required=True,
                    help="feature-set name, e.g. LLM-latent-prefix or LLM-latent-regen")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    controls = set(man.get("technical_controls", ["platform", "instrument"]))

    X, idx = load_features(args.features)
    lab = pd.read_parquet(LABELS).set_index("sample_index").reindex(idx)
    groups = lab.series_id.astype(str).to_numpy()

    label_cols = [c for c in ("tissue", "disease_category", "cvd_subtype",
                              "platform", "instrument") if c in lab.columns]

    rows = []
    for name in label_cols:
        y_raw = lab[name].astype(str).to_numpy()
        print(f"  probing {name} ({len(np.unique(y_raw))} raw classes)...")
        r = probe_label(X, y_raw, groups, name)
        r["feature_set"] = args.name
        r["kind"] = "technical_control" if name in controls else "scientific"
        rows.append(r)
        if "roc_auc_mean" in r:
            print(f"    {name:18s} macro ROC-AUC {r['roc_auc_mean']:.4f} "
                  f"+/- {r['roc_auc_std']:.4f}  ({r['kind']})")
        else:
            print(f"    {name:18s} skipped: {r.get('skipped')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out.relative_to(REPO)}")
    print("NOTE: platform/instrument are technical controls — a high score "
          "there measures batch signal, not representation quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
