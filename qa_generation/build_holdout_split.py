"""Build the clean Stage-2 evaluation holdout (regen plan, Phase 2f).

The holdout is every GEO series that contains BOTH a probe positive and a
`neg_hard` negative — the "mixed" series. Reserving these entirely from Stage-2
training is what makes the post-retrain evaluation meaningful, and it must
happen BEFORE the retrain, not after.

Why mixed series specifically
-----------------------------
`eval_binary_comparison/comparison_report.md` §3 documents the failure this
prevents. The previous session's best-looking number (0.9343 on 172 held-out
positives) was an artifact: those positives spanned 15 series, and *none of
those series contained a single negative*. Under grouping by `series_id` the
classifier was separating 15 specific studies from 22,307 samples drawn from
entirely different studies — a batch-signature task, not a disease task.

A series containing both classes cannot be separated by its batch signature,
because the batch signature is shared by the positives and the negatives inside
it. That is the whole point of the split, so it is asserted here rather than
assumed.

Outputs `data/cvd_transcriptome/holdout_series.json`. Every downstream script —
DE reference construction, template filling, the training-bundle builder, and
every probe — reads the split from that file. Nothing else defines it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PROBE_LABELS = REPO / "linear_probe/probe_sample_labels.parquet"
EXPRESSION_SAMPLE_INDEX = REPO / "qa_generation/bulkformer_input/bulkformer_sample_index.npy"
OUT = REPO / "data/cvd_transcriptome/holdout_series.json"


def build() -> dict:
    probe = pd.read_parquet(PROBE_LABELS)
    expr_rows = set(np.load(EXPRESSION_SAMPLE_INDEX, allow_pickle=True).tolist())

    # The two classes the evaluation is scored on, exactly as linear_probe/probe.py
    # defines them for the neg_hard pool.
    pool = probe[probe.is_positive | probe.is_neg_hard]

    counts = pool.groupby("series_id").agg(
        pos=("is_positive", "sum"), neg=("is_neg_hard", "sum")
    )
    mixed = sorted(counts.index[(counts.pos > 0) & (counts.neg > 0)].tolist())

    held = pool[pool.series_id.isin(mixed)]
    held_pos = held[held.is_positive]
    held_neg = held[held.is_neg_hard]

    # Only positives with an expression row can appear in Stage-2 training at
    # all, so that is the population the training-side exclusion acts on.
    held_pos_expr = held_pos[held_pos.sample_index.isin(expr_rows)]
    train_pos_expr = probe[
        probe.is_positive & probe.sample_index.isin(expr_rows)
        & ~probe.series_id.isin(mixed)
    ]

    # --- assertions: the properties the split exists to guarantee -------------
    failures = []
    for sid in mixed:
        row = counts.loc[sid]
        if row.pos == 0 or row.neg == 0:
            failures.append(f"series {sid} is not mixed: pos={row.pos} neg={row.neg}")
    if set(held_pos.sample_index) & set(train_pos_expr.sample_index):
        failures.append("holdout and training positives overlap")
    if len(mixed) != len(set(mixed)):
        failures.append("duplicate series ids in holdout")

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Stage-2 evaluation holdout. Every series here contains both a probe "
            "positive and a neg_hard negative, so separating its samples cannot "
            "be done on batch signature alone."
        ),
        "definition": (
            "series_id values where (is_positive AND is_neg_hard) both occur, over "
            "linear_probe/probe_sample_labels.parquet"
        ),
        "n_series": len(mixed),
        "n_holdout_positive": int(len(held_pos)),
        "n_holdout_neg_hard": int(len(held_neg)),
        "n_holdout_total": int(len(held)),
        "n_holdout_positive_with_expression": int(len(held_pos_expr)),
        "n_training_positive_with_expression": int(len(train_pos_expr)),
        "n_positive_with_expression_total": int(len(expr_rows)),
        "assertions": {
            "every_series_contains_both_classes": True,
            "no_overlap_with_training_positives": True,
            "passed": not failures,
            "failures": failures,
        },
        "holdout_series": mixed,
        "holdout_sample_index": sorted(int(i) for i in held.sample_index),
        "holdout_geo_accession": sorted(held.geo_accession.astype(str)),
    }


def main() -> int:
    payload = build()
    if not payload["assertions"]["passed"]:
        for f in payload["assertions"]["failures"]:
            print(f"FAIL: {f}")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"holdout: {payload['n_series']} mixed series")
    print(f"  positives      {payload['n_holdout_positive']:,} "
          f"({payload['n_holdout_positive_with_expression']:,} with an expression row)")
    print(f"  neg_hard       {payload['n_holdout_neg_hard']:,}")
    print(f"  stage-2 training positives remaining: "
          f"{payload['n_training_positive_with_expression']:,}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
