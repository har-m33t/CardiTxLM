"""Retune threshold_query's bound thresholds to bounded answer sizes.

Step 3 validation found threshold_query answers running to 500-2,500 genes
(median 4,165 characters, max 48,807) — true answers, but data dumps rather than
learnable ones, and 23,428 of 40,000 exceeded Step 4's configured `max_tokens`.

Target: **5-30 matched genes per answer.** Grounded in two things rather than
picked round: at ~18 characters per "GENE (value)" entry, 30 genes renders to
roughly 600 characters (~150 tokens), which sits inside `max_tokens: 512` with
headroom and alongside the other list categories already in the corpus
(`ranking_ordering_query` median 495 chars, `interaction_network_query` 264). The
lower bound of 5 keeps the answer a genuine list rather than a near-singleton.

Thresholds are derived per sample from that sample's own expression
distribution — a fixed absolute cutoff cannot hold a match count steady across
samples whose distributions differ. For a target of m genes, the threshold is
placed midway between the m-th and (m+1)-th largest values, so exactly m genes
sit strictly above it.

**Direction:** every assignment becomes `above`. `below` is structurally
incapable of hitting this range: a median of 422 pool genes (max 5,534) sit at
exactly 0.0 in a given sample, so any threshold above zero already matches that
entire mass, and any threshold at or below zero matches nothing. Only 1 sample in
200 has 30 or fewer genes at the minimum. The 15,120 `below` assignments are
therefore rebound to `above` and to an above-worded template — see the report;
this is the "adjust the direction if it caused the problem" case, not a silent
semantics change.

Only threshold_query assignments are touched. Every other category is copied
through untouched and verified byte-identical by checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from qa_generation import gt_functions as gt

REPO = Path(__file__).resolve().parent.parent
QA = REPO / "qa_generation"
PLAN = QA / "generation_plan.json"

SEED = 20260803
TARGET_MIN = 5
TARGET_MAX = 30

#: Template indices whose wording implies "above". Mirrors
#: build_generation_plan.THRESHOLD_TEMPLATE_DIRECTION; index 7 states no
#: direction and stays excluded.
ABOVE_TEMPLATES = [0, 1, 2, 3, 4, 8]
BELOW_TEMPLATES = [5, 6]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def category_checksum(assignments: list[dict]) -> dict[str, str]:
    """Per-category checksum over the exact serialized entries, in order."""
    buckets: dict[str, hashlib._Hash] = {}
    for a in assignments:
        buckets.setdefault(a["category"], hashlib.sha256())
        buckets[a["category"]].update(
            json.dumps(a, sort_keys=True).encode() + b"\n"
        )
    return {k: v.hexdigest() for k, v in buckets.items()}


def descending_values(patient: str) -> np.ndarray:
    matrix = np.load(gt.EXPRESSION_NPY, mmap_mode="r")
    row = np.asarray(matrix[gt._expression_rows()[patient]], dtype=np.float64)
    return np.sort(gt._pool_vector(row))[::-1]


def threshold_for_target(values: np.ndarray, m: int) -> tuple[float, int] | None:
    """Threshold placing exactly ~m genes strictly above it, or None if tied.

    `values` is sorted descending. Exactly m genes exceed t when
    values[m] <= t < values[m-1]; the midpoint is used, then rounded to 2 dp and
    re-verified, since rounding can move the count.
    """
    if m < 1 or m >= len(values):
        return None
    hi, lo = float(values[m - 1]), float(values[m])
    if hi <= lo:
        return None  # tie across the cut — no threshold separates them
    threshold = round((hi + lo) / 2.0, 2)
    if not (lo <= threshold < hi):
        return None  # rounding left the open interval
    count = int((values > threshold).sum())
    if TARGET_MIN <= count <= TARGET_MAX:
        return threshold, count
    return None


def retune_patient(
    values: np.ndarray, k: int, rng: random.Random
) -> list[dict] | None:
    """k distinct in-range (threshold, count) bindings for one patient."""
    targets = list(range(TARGET_MIN, TARGET_MAX + 1))
    rng.shuffle(targets)
    out: list[dict] = []
    used: set[float] = set()
    for m in targets:
        if len(out) == k:
            break
        found = threshold_for_target(values, m)
        if found is None:
            continue
        threshold, count = found
        if threshold in used:
            continue
        used.add(threshold)
        out.append(
            {
                "threshold": threshold,
                "direction": "above",
                "n_matching": count,
                "target_rank": m,
            }
        )
    return out if len(out) == k else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    rng = random.Random(SEED)
    plan = json.loads(PLAN.read_text())
    assignments = plan["assignments"]

    before_checksums = category_checksum(assignments)
    threshold_idx = [
        i for i, a in enumerate(assignments) if a["category"] == "threshold_query"
    ]
    log(f"{len(threshold_idx)} threshold_query assignments")

    # --- Baseline: what do the current thresholds actually match? ----------
    by_patient: dict[str, list[int]] = defaultdict(list)
    for i in threshold_idx:
        by_patient[assignments[i]["patient"]].append(i)

    cache: dict[str, np.ndarray] = {}

    def values_for(patient: str) -> np.ndarray:
        if patient not in cache:
            cache[patient] = descending_values(patient)
        return cache[patient]

    log("measuring baseline match counts")
    baseline: list[int] = []
    baseline_by_direction: dict[str, list[int]] = defaultdict(list)
    for patient, idxs in by_patient.items():
        values = values_for(patient)
        for i in idxs:
            e = assignments[i]["entities"]
            if e["direction"] == "above":
                count = int((values > e["threshold"]).sum())
            else:
                count = int((values < e["threshold"]).sum())
            baseline.append(count)
            baseline_by_direction[e["direction"]].append(count)
    base = np.array(baseline)
    stats: dict = {
        "target_range": [TARGET_MIN, TARGET_MAX],
        "before": {
            "n": len(base),
            "in_range": int(((base >= TARGET_MIN) & (base <= TARGET_MAX)).sum()),
            "out_of_range": int(((base < TARGET_MIN) | (base > TARGET_MAX)).sum()),
            "min": int(base.min()),
            "median": int(np.median(base)),
            "max": int(base.max()),
            "by_direction": {
                k: {
                    "n": len(v),
                    "median": int(np.median(v)),
                    "max": int(max(v)),
                    "in_range": int(
                        sum(1 for x in v if TARGET_MIN <= x <= TARGET_MAX)
                    ),
                }
                for k, v in baseline_by_direction.items()
            },
        },
    }
    log(f"  before: median {stats['before']['median']}, max {stats['before']['max']}, "
        f"out of range {stats['before']['out_of_range']}")

    # --- Retune ------------------------------------------------------------
    log("retuning")
    infeasible: list[dict] = []
    retuned = 0
    direction_changed = 0
    for patient, idxs in by_patient.items():
        values = values_for(patient)
        bindings = retune_patient(values, len(idxs), rng)
        if bindings is None:
            for i in idxs:
                infeasible.append(
                    {
                        "assignment_index": i,
                        "patient": patient,
                        "reason": "no_distinct_in_range_threshold",
                    }
                )
            continue
        # Templates must match the (now uniformly "above") direction, and stay
        # distinct within the patient so DIVERSITY_RULE still holds.
        templates = rng.sample(ABOVE_TEMPLATES, min(len(idxs), len(ABOVE_TEMPLATES)))
        while len(templates) < len(idxs):
            templates.append(rng.choice(ABOVE_TEMPLATES))
        for i, binding, t_idx in zip(idxs, bindings, templates):
            if assignments[i]["entities"]["direction"] == "below":
                direction_changed += 1
            assignments[i]["entities"] = binding
            assignments[i]["template_index"] = t_idx
            retuned += 1

    stats["retuned"] = retuned
    stats["direction_changed_below_to_above"] = direction_changed
    stats["infeasible"] = len(infeasible)

    after = np.array(
        [assignments[i]["entities"]["n_matching"] for i in threshold_idx
         if "n_matching" in assignments[i]["entities"]]
    )
    stats["after_planned"] = {
        "n": len(after),
        "min": int(after.min()),
        "median": int(np.median(after)),
        "max": int(after.max()),
        "in_range": int(((after >= TARGET_MIN) & (after <= TARGET_MAX)).sum()),
    }
    log(f"  after (planned): median {stats['after_planned']['median']}, "
        f"range {stats['after_planned']['min']}-{stats['after_planned']['max']}")

    # --- Checksums: everything else must be byte-identical -----------------
    after_checksums = category_checksum(assignments)
    unchanged = {
        k: (before_checksums[k] == after_checksums[k])
        for k in before_checksums
        if k != "threshold_query"
    }
    stats["other_categories_unchanged"] = unchanged
    stats["all_other_categories_byte_identical"] = all(unchanged.values())
    stats["threshold_query_changed"] = (
        before_checksums["threshold_query"] != after_checksums["threshold_query"]
    )
    stats["checksums_before"] = before_checksums
    stats["checksums_after"] = after_checksums

    if not stats["all_other_categories_byte_identical"]:
        log("STOP: a non-threshold category changed")
        return 1

    if infeasible:
        (QA / "threshold_retune_infeasible.json").write_text(
            json.dumps({"count": len(infeasible), "assignments": infeasible}, indent=2)
        )

    if not args.dry_run:
        plan["assignments"] = assignments
        with open(PLAN, "w") as fh:
            json.dump(plan, fh)
        log(f"wrote {PLAN}")

    finished = datetime.now(timezone.utc)
    stats["elapsed_seconds"] = round((finished - started).total_seconds(), 1)
    (QA / "threshold_retune_stats.json").write_text(json.dumps(stats, indent=2))
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
