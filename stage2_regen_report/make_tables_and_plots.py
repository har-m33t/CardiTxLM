"""Phase 5b/5c — consolidated tables and the three-way comparison plots.

Reads the JSON/CSV that Phase 4 produced and emits:

  tables/probe_comparison.csv        encoder vs linear probe vs MLP, mean +/- sd
  tables/multilabel_before_after.csv one row per label, before and after
  plots/roc_pr_three_way.png         one overlay, not three separate figures
  plots/multilabel_before_after.png  paired bars per label

Two presentation rules are enforced here rather than left to the reader:

  * `platform` and `instrument` are TECHNICAL CONTROLS. They are plotted in a
    muted colour and labelled as controls, because a high score there measures
    sequencing-batch signal, not representation quality. Reporting them as a
    win would repeat the mistake comparison_report.md documents.
  * The MLP is never shown without the linear probe beside it. The gap between
    them is the result — it says whether the LLM's latent space holds structure
    a linear probe cannot reach.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
REPORT = REPO / "stage2_regen_report"
TABLES = REPORT / "tables"
PLOTS = REPORT / "plots"

# Brand-neutral, colour-blind-safe, and distinguishable in greyscale.
C_ENCODER = "#4C6EF5"
C_LINEAR = "#12B886"
C_MLP = "#E8590C"
C_CONTROL = "#ADB5BD"


def probe_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"skip probe table: {path} not found")
        return None
    d = json.loads(path.read_text())
    rows = []
    for pop, feats in d.get("populations", {}).items():
        for fname, entry in feats.items():
            if "skipped" in entry:
                continue
            for est in ("linear", "mlp"):
                if est not in entry:
                    continue
                e = entry[est]
                rows.append({
                    "population": pop,
                    "feature_set": fname,
                    "estimator": est,
                    "n": entry.get("n"),
                    "dim": entry.get("dim"),
                    "n_series": entry.get("n_series"),
                    "roc_auc_mean": e.get("roc_auc_mean"),
                    "roc_auc_std": e.get("roc_auc_std"),
                    "pr_auc_mean": e.get("pr_auc_mean"),
                    "pr_auc_std": e.get("pr_auc_std"),
                })
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(
        ["population", "roc_auc_mean"], ascending=[True, False])
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "probe_comparison.csv", index=False)
    print(f"wrote tables/probe_comparison.csv ({len(df)} rows)")
    return df


def probe_plot(df: pd.DataFrame) -> None:
    sub = df[df.population == "holdout_clean"]
    if sub.empty:
        print("skip probe plot: no clean-holdout rows")
        return
    labels, means, errs, colors = [], [], [], []
    for _, r in sub.iterrows():
        est = "linear probe" if r.estimator == "linear" else "MLP probe"
        name = "BulkFormer-93M" if r.feature_set.startswith("BulkFormer") else "LLM latent"
        labels.append(f"{name}\n{est}")
        means.append(r.roc_auc_mean)
        errs.append(r.roc_auc_std or 0)
        colors.append(C_ENCODER if name.startswith("BulkFormer")
                      else (C_MLP if r.estimator == "mlp" else C_LINEAR))

    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=4, color=colors, width=0.6)
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.012, f"{m:.4f}", ha="center", fontsize=8.5)
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.text(len(labels) - 0.4, 0.508, "chance", fontsize=7.5, color="0.5", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("ROC-AUC (grouped 5-fold, holdout only)")
    ax.set_title("Representation quality on the clean holdout")
    ax.set_ylim(0.4, min(1.0, max(means) + 0.09))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "probe_three_way.png", dpi=160)
    plt.close(fig)
    print("wrote plots/probe_three_way.png")


def multilabel(before: Path, after: Path) -> None:
    if not (before.exists() and after.exists()):
        print("skip multi-label: need both before and after CSVs")
        return
    b = pd.read_csv(before).set_index("label")
    a = pd.read_csv(after).set_index("label")
    labels = [l for l in b.index if l in a.index
              and "roc_auc_mean" in b.columns and pd.notna(b.loc[l, "roc_auc_mean"])]
    if not labels:
        print("skip multi-label: no comparable labels")
        return

    rows = []
    for l in labels:
        rows.append({
            "label": l,
            "kind": b.loc[l].get("kind", "scientific"),
            "n_classes": b.loc[l].get("n_classes"),
            "before_roc_auc": b.loc[l, "roc_auc_mean"],
            "before_std": b.loc[l].get("roc_auc_std"),
            "after_roc_auc": a.loc[l, "roc_auc_mean"],
            "after_std": a.loc[l].get("roc_auc_std"),
            "delta": a.loc[l, "roc_auc_mean"] - b.loc[l, "roc_auc_mean"],
            # Carried because it is load-bearing for interpretation: a
            # many-class label under series-grouped CV can lose most folds to
            # "this validation fold contains only one class", and a mean over 2
            # folds is not comparable to a mean over 5. Measured: `tissue` (28
            # classes) scored 2/5. Without this column that difference is
            # invisible and the delta looks more solid than it is.
            "before_folds_scored": b.loc[l].get("n_folds_scored"),
            "after_folds_scored": a.loc[l].get("n_folds_scored"),
        })
    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "multilabel_before_after.csv", index=False)
    print(f"wrote tables/multilabel_before_after.csv ({len(df)} rows)")

    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(df))
    w = 0.38
    ctrl = (df.kind == "technical_control").to_numpy()
    cb = np.where(ctrl, C_CONTROL, C_ENCODER)
    ca = np.where(ctrl, "#868E96", C_LINEAR)
    ax.bar(x - w/2, df.before_roc_auc, w, yerr=df.before_std, capsize=3,
           color=cb, label="before fix")
    ax.bar(x + w/2, df.after_roc_auc, w, yerr=df.after_std, capsize=3,
           color=ca, label="after fix")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.label}\n({r.kind.replace('_',' ')})"
                        for r in df.itertuples()], fontsize=8)
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.set_ylabel("macro ROC-AUC")
    ax.set_title("Broad multi-label probing, before vs after the data fix")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.5, -0.30,
            "Grey pairs are TECHNICAL CONTROLS (sequencing platform / instrument): "
            "a high score there measures\nbatch signal, not representation quality, "
            "and is not a result.",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout()
    fig.savefig(PLOTS / "multilabel_before_after.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote plots/multilabel_before_after.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, default=TABLES / "probe_three_way.json")
    ap.add_argument("--ml-before", type=Path,
                    default=TABLES / "multilabel_probe_before.csv")
    ap.add_argument("--ml-after", type=Path,
                    default=TABLES / "multilabel_probe_after.csv")
    args = ap.parse_args()

    df = probe_table(args.probe)
    if df is not None:
        probe_plot(df)
    multilabel(args.ml_before, args.ml_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
