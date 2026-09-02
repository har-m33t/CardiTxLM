"""Phase 4 output — one table showing all three training conditions.

    1. original   Stage-2 corpus whose answers were 86.4% one fixed string
    2. data-fix   per-sample DE targets, four categories
    3. discrim    data-fix plus the binary disease-vs-control category (Hyp. B)

Conditions are read from whatever artifacts exist; a missing one is reported as
absent rather than interpolated. Every metric is mean +/- std across the same
five grouped folds (StratifiedGroupKFold(5, seed 20260707) by series_id) on the
same 92-series clean holdout, so the columns are comparable.

HOW TO READ THIS TABLE — two traps, both easy to fall into
-----------------------------------------------------------
1. POOLED AUC IS NOT THE HEADLINE. Zero non-holdout series contains both
   classes, so a model can score on pooled AUC by recognizing batch signature
   and learning no biology. Within-series AUC is the number that cannot be
   reached that way. The frozen encoder measures 0.765 within-series against
   0.643 pooled, so the bar this project has to clear is 0.765 — the pooled
   0.668 previously quoted UNDERSTATES the baseline.

2. THE LOSS FLOOR ANSWERS NOTHING HERE. Condition 3 adds an easy one-bit
   target, so its floor should fall below condition 2 from mixture arithmetic
   alone. That is not evidence for or against Hypothesis B.

The verdict this table supports is narrow and should stay narrow: did adding
discriminative supervision move the LLM's representation relative to the frozen
encoder it sits on? Everything else is context.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TABLES = REPO / "stage2_regen_report/tables"
LOSS = REPO / "stage2_regen_report/loss_curves/stage2_loss_floors.json"

#: (condition, three-way probe json, binary eval json). None == not produced.
CONDITIONS = [
    ("original",  None,                          None),
    ("data-fix",  "probe_three_way.json",        None),
    ("discrim",   "probe_three_way_hypb.json",   "binary_cvd_eval_hypb.json"),
]

FEATURES = ["LLM-latent-imgtok", "BulkFormer-93M"]


def agg(block: dict, model: str) -> tuple[float, float] | None:
    """mean/std of roc_auc across folds for one probe head."""
    b = block.get(model)
    if not isinstance(b, dict) or "per_fold" not in b:
        return None
    v = [f["roc_auc"] for f in b["per_fold"] if "roc_auc" in f]
    return (float(np.mean(v)), float(np.std(v))) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", type=Path, default=TABLES / "three_way_conditions.csv")
    ap.add_argument("--out-json", type=Path, default=TABLES / "three_way_conditions.json")
    args = ap.parse_args()

    floors = json.loads(LOSS.read_text()) if LOSS.exists() else {}
    rows, summary = [], {}

    for cond, probe_file, binary_file in CONDITIONS:
        rec: dict = {"condition": cond}

        if probe_file and (TABLES / probe_file).exists():
            d = json.loads((TABLES / probe_file).read_text())
            clean = d.get("populations", {}).get("holdout_clean", {})
            for feat in FEATURES:
                blk = clean.get(feat, {})
                for head in ("linear", "mlp", "pca_matched"):
                    a = agg(blk, head)
                    if a:
                        rec[f"{feat}:{head}"] = a
            rec["n_holdout"] = clean.get(FEATURES[0], {}).get("n")
        else:
            rec["probe"] = "not produced"

        if binary_file and (TABLES / binary_file).exists():
            b = json.loads((TABLES / binary_file).read_text())
            rec["binary"] = b
        elif binary_file:
            rec["binary"] = "not produced"

        for k, v in floors.items():
            if cond.split("-")[0] in k.lower() or (cond == "discrim" and "discrim" in k.lower()):
                rec["loss_floor"] = v.get("final_10pct_mean")

        summary[cond] = rec
        rows.append(rec)

    args.out_json.write_text(json.dumps({
        "purpose": "all three Stage-2 training conditions on one page",
        "read_me_first": {
            "pooled_auc_is_not_the_headline": (
                "Zero non-holdout series contains both classes, so pooled AUC "
                "can be reached by recognizing batch signature. Within-series "
                "AUC cannot."),
            "encoder_bar_within_series": 0.765,
            "encoder_bar_pooled": 0.643,
            "loss_floor_is_not_evidence": (
                "condition 3 adds an easy one-bit target, so a lower floor is "
                "mixture arithmetic, not representation quality"),
        },
        "conditions": summary,
    }, indent=2) + "\n")

    cols = sorted({k for r in rows for k in r if isinstance(r.get(k), tuple)})
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition"] + [f"{c} (mean)" for c in cols]
                   + [f"{c} (std)" for c in cols] + ["loss_floor", "n_holdout"])
        for r in rows:
            w.writerow([r["condition"]]
                       + [f"{r[c][0]:.4f}" if c in r else "" for c in cols]
                       + [f"{r[c][1]:.4f}" if c in r else "" for c in cols]
                       + [r.get("loss_floor", ""), r.get("n_holdout", "")])

    print(f"wrote {args.out_csv.relative_to(REPO)}")
    print(f"wrote {args.out_json.relative_to(REPO)}")
    for r in rows:
        got = {k: v for k, v in r.items() if isinstance(v, tuple)}
        if got:
            print(f"\n{r['condition']}:")
            for k, (m, sd) in sorted(got.items()):
                print(f"  {k:36s} {m:.4f} +/- {sd:.4f}")
        else:
            print(f"\n{r['condition']}: {r.get('probe', 'no probe artifact')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
