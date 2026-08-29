"""Phase 4 — representation quality on the clean holdout.

Three feature sets, one methodology, three-way comparison:

    BulkFormer-93M        515-d   the frozen encoder, the baseline to beat
    LLM latent (linear)  4096-d   a linear probe on the retrained LLM's states
    LLM latent (MLP)     4096-d   a small MLP on the same states

The MLP is the new instrument (KRONOS reported a plain MLP competitive with a
graph model, suggesting signal a linear probe cannot reach). It is reported
ALONGSIDE the linear probe, never instead of it — the gap between them is
itself the result: a large gap means the LLM's space holds non-linearly
separable structure, a small one means it does not.

POPULATION. The primary evaluation is restricted to the 92 mixed holdout
series. Every one contains both a positive and a neg_hard negative, so the task
cannot be solved by batch signature — which is exactly how the previous
session's best-looking number (0.9343) turned out to be an artifact: its 15
series contained no negatives at all, so the classifier was separating studies,
not disease (comparison_report.md section 3). The retrained model never saw
these samples in Stage 2, so this is the first uncontaminated measurement at
this stage.

The full-population number is ALSO computed and reported, explicitly labelled
contaminated, so the result stays comparable to the prior session's table
rather than silently changing the population under the reader.

Folds, metric and estimator come from `linear_probe/probe.py` — imported, not
reimplemented, so "same methodology" is a fact rather than a claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_binary_comparison.embedding_io import load_embeddings
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from linear_probe.probe import _fold_metrics

REPO = Path(__file__).resolve().parent.parent
EMB = REPO / "linear_probe/embeddings"
HOLDOUT = REPO / "data/cvd_transcriptome/holdout_series.json"
PROBE_LABELS = REPO / "linear_probe/probe_sample_labels.parquet"

#: Identical to every prior probe in this project. Changing either would make
#: the before/after comparison meaningless.
SEED = 20260707
K_FOLDS = 5


def load_features(path: Path):
    """Delegates to embedding_io, which asserts no dimension was dropped.

    Do NOT inline a `startswith("e0")` filter here — that matches only
    e0000..e0999 and silently truncates a 4096-d latent to 1000. See
    eval_binary_comparison/embedding_io.py.
    """
    return load_embeddings(path)


def linear_est():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )


def pca_matched_est(n_components: int):
    """Linear probe on the LLM latents reduced to the encoder's width.

    The headline three-way comparison is not dimensionality-matched: 4096-d LLM
    latents against a 515-d encoder, on 2,607 holdout samples. In that regime a
    linear probe's regulariser, rather than the representation, can be the
    binding constraint — so a tie is weaker evidence against the LLM than it
    looks. Reducing the latents to exactly the encoder's width removes that
    asymmetry and makes the two directly comparable.

    Fitted INSIDE the CV loop (it is a pipeline step), so the PCA never sees the
    validation fold. Fitting it once over the whole population would leak.
    """
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=SEED),
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           random_state=SEED),
    )


def mlp_est():
    # Start simple, per the plan: 512 -> 256 -> out, tuned only if the first
    # pass is clearly under/overfitting. early_stopping guards the latter.
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(512, 256), activation="relu", alpha=1e-4,
            batch_size=256, learning_rate_init=1e-3, max_iter=300,
            early_stopping=True, n_iter_no_change=15, validation_fraction=0.1,
            random_state=SEED,
        ),
    )


def run_cv(X, y, groups, make_est, name: str) -> dict:
    skf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for k, (tr, va) in enumerate(skf.split(X, y, groups)):
        est = make_est()
        est.fit(X[tr], y[tr])
        p = est.predict_proba(X[va])[:, 1]
        m = _fold_metrics(y[va], p)
        m["fold"] = k
        rows.append(m)
        print(f"    [{name}] fold {k}: roc_auc={m.get('roc_auc'):.4f}")
    df = pd.DataFrame(rows)
    out = {"n_folds": len(rows), "per_fold": rows}
    for metric in ("roc_auc", "pr_auc"):
        if metric in df:
            out[f"{metric}_mean"] = float(df[metric].mean())
            out[f"{metric}_std"] = float(df[metric].std())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-latents", type=Path,
                    default=EMB / "embeddings_LLM-latent-regen-imgtok.parquet")
    ap.add_argument("--encoder", type=Path,
                    default=EMB / "embeddings_BulkFormer-93M.parquet")
    ap.add_argument("--out", type=Path,
                    default=REPO / "stage2_regen_report/tables/probe_three_way.json")
    args = ap.parse_args()

    held_series = set(json.loads(HOLDOUT.read_text())["holdout_series"])
    labels = pd.read_parquet(PROBE_LABELS).set_index("sample_index")

    feats = {}
    Xl, idx_l = load_features(args.llm_latents)
    feats["LLM-latent-imgtok"] = (Xl, idx_l)
    Xe, idx_e = load_features(args.encoder)
    feats["BulkFormer-93M"] = (Xe, idx_e)
    encoder_dim = Xe.shape[1]

    results: dict = {"seed": SEED, "k_folds": K_FOLDS, "populations": {}}

    for pop_name, restrict in (("holdout_clean", True), ("full_contaminated", False)):
        print(f"\n=== population: {pop_name} ===")
        results["populations"][pop_name] = {}
        for fname, (X, idx) in feats.items():
            meta = labels.reindex(idx)
            # .copy(): pandas can hand back a read-only view here, and the
            # in-place &= below then raises "output array is read-only".
            keep = (meta.is_positive | meta.is_neg_hard).to_numpy().copy()
            if restrict:
                keep &= meta.series_id.isin(held_series).to_numpy()
            Xs = X[keep]
            y = meta.is_positive.to_numpy()[keep].astype(int)
            groups = meta.series_id.to_numpy()[keep]
            n_series = len(set(groups))
            print(f"  {fname}: n={Xs.shape[0]:,} dim={Xs.shape[1]} "
                  f"pos={int(y.sum()):,} neg={int((1-y).sum()):,} series={n_series}")
            if y.sum() < K_FOLDS or (1 - y).sum() < K_FOLDS or n_series < K_FOLDS:
                results["populations"][pop_name][fname] = {
                    "skipped": "too few samples/series for grouped CV"}
                continue

            entry = {"n": int(Xs.shape[0]), "dim": int(Xs.shape[1]),
                     "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
                     "n_series": n_series}
            entry["linear"] = run_cv(Xs, y, groups, linear_est, f"{fname}/linear")
            if fname.startswith("LLM"):
                entry["mlp"] = run_cv(Xs, y, groups, mlp_est, f"{fname}/mlp")
                # Dimensionality-matched control: same estimator, same folds,
                # reduced to the encoder's width so the comparison is like for
                # like. n_components cannot exceed n_samples in the smallest
                # training fold, so it is clamped.
                k = min(encoder_dim, Xs.shape[1], int(Xs.shape[0] * 0.6))
                entry["linear_pca_matched"] = run_cv(
                    Xs, y, groups, lambda: pca_matched_est(k),
                    f"{fname}/linear-pca{k}")
                entry["pca_components"] = int(k)
            results["populations"][pop_name][fname] = entry

    # The guard that makes the holdout number trustworthy, asserted not assumed.
    counts = (labels[labels.series_id.isin(held_series)]
              .groupby("series_id")[["is_positive", "is_neg_hard"]].sum())
    bad = counts[(counts.is_positive == 0) | (counts.is_neg_hard == 0)]
    results["holdout_guard"] = {
        "n_series": int(len(counts)),
        "n_series_missing_a_class": int(len(bad)),
        "every_series_has_both_classes": bool(len(bad) == 0),
        "note": ("a series with only one class can be separated on batch "
                 "signature alone; comparison_report.md section 3 is what "
                 "happens when this is not checked"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.out.relative_to(REPO)}")
    print(f"holdout guard: every series has both classes = "
          f"{results['holdout_guard']['every_series_has_both_classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
