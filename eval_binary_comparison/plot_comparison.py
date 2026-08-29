"""plot_comparison.py — per-condition AUROC: frozen encoder vs LLM latents.

Regenerates per_condition_auroc.png from the saved JSON, so the figure is
reproducible instead of being a one-off inline render.

All rows come from ONE probe per feature set (run_pooled_breakdown): a single
CVD-vs-control model, with AUROC computed separately per condition on each
validation fold. Identical folds, model and metric across rows, so the bars are
comparable by construction.

Run:  python -m eval_binary_comparison.plot_comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_binary_comparison.per_condition_probe import CONDITIONS

# Okabe-Ito: colourblind-safe by construction, fixed order, never cycled.
#
# EVERY row must come from the SAME sklearn build. StratifiedGroupKFold's split
# algorithm changed between 1.6.1 (local) and 1.9.0 (pod): the same seed yields
# different folds, moving macro AUROC by 1-3 points and changing how many folds
# clear the per-condition floor. An earlier version of this figure mixed the two
# and was therefore not a like-for-like comparison. All four rows below were
# produced on the pod under sklearn 1.9.0.
SERIES = [
    ("BulkFormer-37M",    "bf_37_50_pod.json",     "#56B4E9"),
    ("BulkFormer-50M",    "bf_37_50_pod.json",     "#0072B2"),
    ("BulkFormer-93M",    "llm_latent_probe.json", "#009E73"),
    ("LLM-latent-imgtok", "llm_latent_probe.json", "#D55E00"),
]
SLIDE_TARGET = 0.81
BASE = Path("eval_binary_comparison")


def load_rows():
    cache, rows = {}, []
    for name, fname, colour in SERIES:
        if fname not in cache:
            cache[fname] = json.load(open(BASE / fname))
        blob = cache[fname]
        rec = blob[name] if name in blob else blob["variants"][name]
        per = rec["per_condition"]
        vals = [per[k]["roc_auc_mean"] for k, _ in CONDITIONS]
        errs = [per[k].get("roc_auc_std") or 0.0 for k, _ in CONDITIONS]
        rows.append((name, colour, vals, errs, rec["macro_auroc"]))
    return rows


def main():
    rows = load_rows()
    labels = [lab for _, lab in CONDITIONS]
    x = np.arange(len(labels))
    w = 0.8 / len(rows)

    fig, (ax, axm) = plt.subplots(
        1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [3.1, 1]})

    for i, (name, colour, vals, errs, _) in enumerate(rows):
        off = (i - (len(rows) - 1) / 2) * w
        ax.bar(x + off, vals, w * 0.9, label=name, color=colour,
               yerr=errs, error_kw=dict(ecolor="#9a9a9a", lw=1, capsize=2),
               edgecolor="white", linewidth=1.2, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="#b0b0b0", lw=1, ls=":", zorder=1)
    ax.set_title("Per-condition AUROC — one pooled CVD-vs-control probe per feature set",
                 fontsize=11, loc="left")
    ax.grid(axis="y", color="#e6e6e6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")

    # ---- macro panel: the number the slide reports ----
    names = [r[0] for r in rows]
    macros = [r[4] for r in rows]
    cols = [r[1] for r in rows]
    xm = np.arange(len(names))
    axm.bar(xm, macros, 0.62, color=cols, edgecolor="white", linewidth=1.2, zorder=3)
    axm.axhline(SLIDE_TARGET, color="#333333", lw=1.4, ls="--", zorder=4)
    # Label sits BELOW the line on the left, clear of every bar top.
    axm.text(-0.45, SLIDE_TARGET - 0.012, f"slide target {SLIDE_TARGET:.2f}",
             ha="left", va="top", fontsize=8.5, color="#333333")
    for xi, m in zip(xm, macros):
        axm.text(xi, m + 0.006, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    axm.set_xticks(xm)
    axm.set_xticklabels([n.replace("BulkFormer-", "BF-").replace("LLM-latent-", "LLM\n")
                         for n in names], fontsize=8.5)
    axm.set_ylim(0.5, 1.0)
    axm.set_ylabel("macro ROC-AUC")
    axm.set_title("Macro AUROC", fontsize=11, loc="left")
    axm.grid(axis="y", color="#e6e6e6", lw=0.8, zorder=0)
    axm.set_axisbelow(True)
    for s in ("top", "right"):
        axm.spines[s].set_visible(False)

    fig.text(0.005, 0.015,
             "Error bars: SD across folds. StratifiedGroupKFold(5) grouped by series_id, seed 20260707, neg_hard pool. "
             "LLM row carries a train/eval overlap caveat (98% of eval positives were in stage-2 training); "
             "BulkFormer rows do not — the encoder was frozen and never saw a label.",
             fontsize=7.4, color="#666666")

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = BASE / "per_condition_auroc.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
