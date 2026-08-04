"""Step 2 — per-patient, per-category template sampling plan.

Produces `generation_plan.json`, the direct input to Step 3 (template filling).
Assigns, for every eligible patient, which template phrasings and which bound
entities will be used — before any GT is computed or any text is generated.

Confirmed parameters (not re-derived here):
    POPULATION_SCOPE            = per_category_maximal_eligibility
    STAGE1_PER_CATEGORY_SPLIT   = even
    SUBTYPE_CAP_PCT             = 35
    DIVERSITY_RULE              = one_template_per_patient_category_entity
    RANKING_DIVERSIFICATION     = true

Two structural constraints are enforced here rather than left to Step 3, because
both would otherwise produce items whose question and answer disagree:

  * **Template/parameter compatibility.** Ranking templates are not
    interchangeable — some ask for `{N}`, some for `{percentile}`, some for the
    *bottom*. A percentile bound into a "top {N}" template reads wrong. Each
    template therefore carries the parameter shape and direction it can accept,
    and only compatible bindings are assigned.
  * **Degenerate answers are never planned.** Bottom-N ranking ties at 0.0 in
    every sample checked (40/40), and thresholds outside a sample's own value
    range match all or no genes. Thresholds are drawn from each patient's own
    quantiles so every planned item has a real, non-degenerate answer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from qa_generation import gt_functions as gt

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "qa_generation/templates"
OUTDIR = REPO / "qa_generation"

SEED = 20260803
STAGE1_TARGET = 200_000
STAGE2_TARGET = 50_000
SUBTYPE_CAP_PCT = 35.0

#: Partner counts for interaction_network_query, bound explicitly per instance.
#: Every template in that category now carries an {N} placeholder, so the count
#: appears in the question text instead of being a silent GT default. Varied for
#: diversity; 10 stays the most common, matching the previous behaviour.
INTERACTION_N_CHOICES = (5, 10, 10, 15, 20)

#: Answer size for gene_driver_reasoning, keyed by template index.
#:
#: Unlike interaction_network_query, these templates need no `{N}` placeholder:
#: none states or implies a count, and none implies "all" either (checked
#: against every one of the 10 — see step2_final_cleanup_report.md), so any
#: bounded list satisfies the question as written.
#:
#: But `top_n` cannot then vary *within* a phrasing. "Determine the top
#: molecular signals..." appears ~855 times; if its answer were sometimes 10
#: genes and sometimes 20, identical question text would carry different
#: answers, which is the same defect this pass exists to remove. Keying the
#: bound to the template index gives variation across the corpus while keeping
#: each phrasing internally consistent.
GENE_DRIVER_TOP_N = {i: (10, 15, 20)[i % 3] for i in range(10)}

# --- Template/parameter compatibility -------------------------------------
# Indices into the YAML lists. Derived by reading each template's wording.

#: Ranking templates: (index -> (parameter shape, direction)).
RANKING_TEMPLATE_SPEC = {
    0: ("percentile", "top"),  # "top {percentile} when ranked by expression"
    1: ("count", "top"),       # "highest-expressed {N} genes"
    2: ("count", "top"),       # "top {N} genes with the greatest expression"
    3: ("count", "top"),       # "among the top {N} by expression ranking"
    4: ("count", "top"),       # "top {N} genes ordered by measured expression"
    5: ("percentile", "top"),  # "upper {percentile} of expression values"
    6: ("count", "bottom"),    # "bottom {N} genes by expression ranking"
    7: ("count", "bottom"),    # "among the bottom {N}"
    8: ("count", "top"),       # "order by expression and report the first {N}"
    9: ("count", "top"),       # "top {N} genes occupying the highest positions"
}

#: Bottom-direction ranking templates are excluded: the filtered pool's low end
#: is a long run of exact 0.0 values, so `boundary_tie` fires in every sample
#: sampled (40/40) and "the bottom 10 genes" has no well-defined answer.
RANKING_EXCLUDED = {6, 7}

#: Threshold templates: index -> direction the wording implies.
THRESHOLD_TEMPLATE_DIRECTION = {
    0: "above",  # "exceed {threshold}"
    1: "above",  # "above {threshold}"
    2: "above",  # "greater than {threshold}"
    3: "above",  # "meet the expression threshold of {threshold}"
    4: "above",  # "higher than {threshold}"
    5: "below",  # "below {threshold}"
    6: "below",  # "falls under {threshold}"
    8: "above",  # "beyond {threshold}"
}
#: Template 7 ("...that satisfy {threshold}") states no direction at all, so no
#: binding can be faithful to it. Excluded rather than guessed.
THRESHOLD_EXCLUDED = {7}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def load_templates() -> dict[str, list[str]]:
    out = {}
    for name in ("stage1", "stage2"):
        with open(TEMPLATES / f"{name}.yaml") as fh:
            out.update(yaml.safe_load(fh))
    return out


# --- Task 1: per-category eligible populations -----------------------------


def eligible_populations() -> tuple[dict[str, list[str]], dict]:
    """Each category's own maximal eligible set, restricted to usable input.

    Category gates are verified against `gt_functions` by calling each function
    rather than re-deriving the conditions: a category is eligible for a sample
    exactly when its GT function returns `ok`.

    Every set is then intersected with the samples that have a materialized
    BulkFormer input row. Stage 2's GT needs no expression — but a training item
    still needs an `image` `.npy`, and the 2,004 samples without one were
    excluded upstream for **data-quality** reasons, not incidentally: 1,832 are
    single-cell (`singlecellprobability >= 0.5`) and so the wrong modality for a
    bulk model, and 172 are bulk but have library sizes below 100,000 (median
    4,579 against 1,920,351 for kept samples; 66 below 1,000 total counts), where
    TPM is noise. See `qa_generation/step2_followup_report.md`.

    This keeps POPULATION_SCOPE = per_category_maximal_eligibility: each category
    still draws its own maximal set, now bounded by what can actually be turned
    into a training example rather than by the narrow four-way intersection.
    """
    labels = gt._sample_labels()
    probe = gt._probe_labels()

    materialized = set(gt._expression_rows())
    raw = {
        "stage1": sorted(materialized),
        "disease_subtype_classification": sorted(
            labels[
                labels.is_cvd_disease & ~labels.cvd_subtype.isin(gt.NON_SUBTYPE_LABELS)
            ].geo_accession.astype(str)
        ),
        "comparative_differential_reasoning": sorted(
            probe[probe.is_positive].index.astype(str)
        ),
        "gene_driver_reasoning": sorted(
            labels[labels.is_cvd_disease].geo_accession.astype(str)
        ),
    }
    kept = {k: sorted(set(v) & materialized) for k, v in raw.items()}
    dropped = {
        k: {
            "eligible_by_gate": len(v),
            "without_materialized_input": len(set(v) - materialized),
            "retained": len(kept[k]),
        }
        for k, v in raw.items()
    }
    return kept, dropped


# --- Task 2: subtype cap ---------------------------------------------------


def apply_subtype_cap(
    patients: list[str], cap_pct: float, rng: random.Random
) -> tuple[dict, list[str]]:
    """Cap any single subtype's share, redistributing proportionally.

    The cap targets `disease_matched_subtype_unresolved`. That bucket cannot
    appear here: `disease_subtype_classification` returns insufficient_data for
    it, so the eligible population is resolved subtypes only. The cap is
    evaluated against every subtype regardless, so it engages if the label
    distribution ever changes.
    """
    labels = gt._sample_labels()
    by_subtype: dict[str, list[str]] = defaultdict(list)
    for patient in patients:
        by_subtype[str(labels.loc[patient, "cvd_subtype"])].append(patient)
    before = {k: len(v) for k, v in by_subtype.items()}
    total = sum(before.values())

    cap_n = {k: int(math.floor(cap_pct / 100.0 * total)) for k in before}
    over = {k: v for k, v in before.items() if v > cap_n[k]}

    kept: list[str] = []
    after = {}
    for subtype, members in by_subtype.items():
        limit = cap_n[subtype] if subtype in over else len(members)
        chosen = rng.sample(members, limit) if limit < len(members) else list(members)
        after[subtype] = len(chosen)
        kept.extend(chosen)

    # The freed budget cannot be redistributed. Every category item here is one
    # per patient, and every remaining subtype already contributes all of its
    # patients — there is no spare capacity to move the surplus into. Inventing
    # extra items for the under-represented subtypes would mean asking the same
    # patient the same question twice, which DIVERSITY_RULE forbids.
    freed = sum(before[k] - after[k] for k in over)

    return {
        "before": before,
        "after": after,
        "total_before": total,
        "total_after": sum(after.values()),
        "cap_pct": cap_pct,
        "cap_n_per_subtype": cap_n,
        "subtypes_over_cap": sorted(over),
        "cap_engaged": bool(over),
        "freed_budget": freed,
        "freed_budget_redistributed": 0,
        "redistribution_note": (
            "not redistributable — one item per patient, and every remaining "
            "subtype already contributes all of its eligible patients"
        ),
        "unresolved_present": "disease_matched_subtype_unresolved" in before,
    }, sorted(kept)


# --- Task 3: allocation ----------------------------------------------------


def allocate(
    capacities: list[int], target: int, rng: random.Random
) -> tuple[list[int], int]:
    """Split `target` as evenly as integers allow, respecting per-patient caps.

    Capacity matters because entity supply is not uniform: a patient's threshold
    bindings are limited to the non-degenerate ones its own expression
    distribution supports, and ranking windows to the 13 distinct specs. Rather
    than emitting fewer items than allocated and discovering the gap later, the
    shortfall is redistributed to patients that still have room, and any
    genuinely unachievable remainder is returned for honest reporting.
    """
    n = len(capacities)
    base = target // n
    counts = [min(cap, base) for cap in capacities]
    deficit = target - sum(counts)

    spare = [i for i in range(n) if counts[i] < capacities[i]]
    rng.shuffle(spare)
    cursor = 0
    while deficit > 0 and spare:
        i = spare[cursor % len(spare)]
        if counts[i] < capacities[i]:
            counts[i] += 1
            deficit -= 1
            cursor += 1
        else:
            spare.remove(i)
            if cursor >= len(spare) and spare:
                cursor = 0
    return counts, deficit


# --- Per-patient expression context ----------------------------------------


#: Target answer sizes a threshold item should produce, as gene counts.
#: Thresholds are derived from the rank position that yields roughly this many
#: matches, then verified — quantile-derived thresholds do not work here,
#: because the filtered pool's lower half is a long run of exact 0.0 and any
#: threshold rounding to 0.0 makes "below" match nothing at all.
ABOVE_TARGET_COUNTS = (10, 25, 50, 100, 250, 500)
BELOW_TARGET_COUNTS = (500, 1000, 1500, 2000, 2500)


def threshold_candidates(values: np.ndarray) -> list[dict]:
    """Non-degenerate (threshold, direction) pairs for one patient.

    Each candidate's real match count is computed against the sorted values, so
    nothing degenerate — 0 matches or all 5,797 — is ever planned.
    """
    n = len(values)
    ascending = np.sort(values)
    out: list[dict] = []
    seen: set[tuple[float, str]] = set()

    for m in ABOVE_TARGET_COUNTS:
        threshold = round(float(ascending[n - m]), 1)
        count = int(n - np.searchsorted(ascending, threshold, side="right"))
        if 0 < count < n and (threshold, "above") not in seen:
            seen.add((threshold, "above"))
            out.append(
                {"threshold": threshold, "direction": "above", "n_matching": count}
            )

    for m in BELOW_TARGET_COUNTS:
        if m >= n:
            continue
        threshold = round(float(ascending[m]), 1)
        count = int(np.searchsorted(ascending, threshold, side="left"))
        if 0 < count < n and (threshold, "below") not in seen:
            seen.add((threshold, "below"))
            out.append(
                {"threshold": threshold, "direction": "below", "n_matching": count}
            )
    return out


def patient_context(patients: list[str]) -> tuple[dict[str, dict], list[str]]:
    """Per-patient ranking order and verified threshold candidates, computed once."""
    matrix = np.load(gt.EXPRESSION_NPY, mmap_mode="r")
    rows = gt._expression_rows()
    pool = list(gt._pool_genes())
    pool_arr = np.array(pool, dtype=object).astype(str)

    ctx = {}
    for i, patient in enumerate(patients):
        values = gt._pool_vector(np.asarray(matrix[rows[patient]], dtype=np.float64))
        order = np.lexsort((pool_arr, -values))
        ctx[patient] = {
            "order": order.astype(np.int32),
            # Descending values in rank order, kept so a cut at rank N can be
            # tested for a tie against rank N+1 before the item is emitted.
            "ranked_values": values[order].astype(np.float32),
            "thresholds": threshold_candidates(values),
        }
        if i % 2000 == 0:
            log(f"  context {i}/{len(patients)}")
    return ctx, pool


# --- Task 5: ranking diversification ---------------------------------------

#: Mitigation for mitochondrial dominance. 11 MT- genes sit in the filtered pool
#: and occupy ~64% of the average top-10, so a corpus of top-10 questions would
#: keep re-asking about the same 11 genes. Widening the window dilutes them:
#: MT share falls from 64% at top-10 to 19% at top-50. Bands are sampled per
#: instance; `wide` and `percentile` together are the diversified fraction.
RANKING_BAND_WEIGHTS = {"narrow": 0.55, "wide": 0.30, "percentile": 0.15}

#: (spec, parameter shape, band). Widening the window is what dilutes the
#: mitochondrial genes: they hold ~64% of an average top-10 but only ~19% of a
#: top-50, so `wide` and `percentile` instances carry most of the diversity.
RANKING_SPECS = (
    [(n, "count", "narrow") for n in (5, 10, 15, 20)]
    + [(n, "count", "wide") for n in (25, 30, 40, 50, 75, 100)]
    + [(f"{p}%", "percentile", "percentile") for p in (1.0, 2.0, 5.0)]
)


def sample_ranking_entities(rng: random.Random, k: int) -> list[dict]:
    """k distinct (spec, direction) bindings, band-weighted.

    Sampling is without replacement across the whole spec list, so a patient can
    never be assigned the same window twice — the failure mode a retry loop
    would leave open once the small band is exhausted.
    """
    remaining = list(RANKING_SPECS)
    chosen: list[dict] = []
    while len(chosen) < k and remaining:
        weights = [RANKING_BAND_WEIGHTS[band] for _, _, band in remaining]
        spec, shape, band = rng.choices(remaining, weights=weights, k=1)[0]
        remaining.remove((spec, shape, band))
        chosen.append(
            {"spec": spec, "direction": "top", "shape": shape, "band": band}
        )
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    rng = random.Random(SEED)
    templates = load_templates()
    pop, eligibility_detail = eligible_populations()
    log({k: len(v) for k, v in pop.items()})

    stage1_patients = pop["stage1"]
    log("building per-patient expression context")
    ctx, pool = patient_context(stage1_patients)

    stage1_categories = [
        "direct_abundance_query",
        "threshold_query",
        "ranking_ordering_query",
        "comparative_query",
        "interaction_network_query",
    ]
    per_category_target = STAGE1_TARGET // len(stage1_categories)

    plan: list[dict] = []
    dropped_assignments: list[dict] = []
    stats: dict = {"stage1": {}, "stage2": {}}
    ranking_answer_genes: list[list[str]] = []
    ranking_baseline_genes: list[list[str]] = []

    # --- Stage 1 -----------------------------------------------------------
    for category in stage1_categories:
        n_templates = len(templates[category])
        if category == "ranking_ordering_query":
            usable = [i for i in range(n_templates) if i not in RANKING_EXCLUDED]
            capacities = [len(RANKING_SPECS)] * len(stage1_patients)
        elif category == "threshold_query":
            usable = [i for i in range(n_templates) if i not in THRESHOLD_EXCLUDED]
            capacities = [len(ctx[p]["thresholds"]) for p in stage1_patients]
        else:
            usable = list(range(n_templates))
            capacities = [len(pool)] * len(stage1_patients)
        counts, unmet = allocate(capacities, per_category_target, rng)

        emitted = 0
        for patient, k in zip(stage1_patients, counts):
            if k == 0:
                continue
            # DIVERSITY_RULE: one template per (patient, category, entity), and
            # entities are distinct within the patient-category.
            tmpl_choices = rng.sample(usable, min(k, len(usable)))
            while len(tmpl_choices) < k:
                tmpl_choices.append(rng.choice(usable))

            if category in ("direct_abundance_query", "interaction_network_query"):
                genes = rng.sample(pool, k)
                entities = [{"gene": g} for g in genes]
                if category == "interaction_network_query":
                    for e in entities:
                        e["n"] = rng.choice(INTERACTION_N_CHOICES)

            elif category == "comparative_query":
                seen, entities = set(), []
                while len(entities) < k:
                    a, b = rng.sample(pool, 2)
                    key = tuple(sorted((a, b)))
                    if key in seen:
                        continue
                    seen.add(key)
                    entities.append({"gene_a": a, "gene_b": b})

            elif category == "threshold_query":
                # Entity first, template second: a template's wording fixes the
                # direction, so the binding has to exist before the phrasing is
                # chosen. Candidates are already distinct and non-degenerate.
                candidates = ctx[patient]["thresholds"]
                entities = rng.sample(candidates, min(k, len(candidates)))
                tmpl_choices = []
                for entity in entities:
                    matching = [
                        i
                        for i in usable
                        if THRESHOLD_TEMPLATE_DIRECTION[i] == entity["direction"]
                        and i not in tmpl_choices
                    ] or [
                        i
                        for i in usable
                        if THRESHOLD_TEMPLATE_DIRECTION[i] == entity["direction"]
                    ]
                    tmpl_choices.append(rng.choice(matching))

            else:  # ranking_ordering_query
                # Entity first: a "top {percentile}" template cannot take a
                # count, so the phrasing is chosen to fit the binding.
                entities = sample_ranking_entities(rng, k)
                tmpl_choices = []
                for entity in entities:
                    matching = [
                        i
                        for i in usable
                        if RANKING_TEMPLATE_SPEC[i][0] == entity["shape"]
                        and i not in tmpl_choices
                    ] or [
                        i for i in usable if RANKING_TEMPLATE_SPEC[i][0] == entity["shape"]
                    ]
                    tmpl_choices.append(rng.choice(matching))

            for t_idx, entity in zip(tmpl_choices, entities):
                if category == "ranking_ordering_query":
                    order = ctx[patient]["order"]
                    ranked = ctx[patient]["ranked_values"]
                    spec = entity["spec"]
                    n = (
                        int(math.ceil(float(str(spec).rstrip("%")) / 100.0 * len(pool)))
                        if isinstance(spec, str)
                        else int(spec)
                    )
                    # A cut that lands inside a run of equal values leaves "the
                    # top N genes" without a unique membership: the gene at rank
                    # N and the one at N+1 are indistinguishable on the only
                    # criterion the question states. The item is true of no
                    # single gene set, so it is dropped rather than emitted with
                    # an arbitrary tie-break baked into the answer.
                    if n < len(pool) and ranked[n - 1] == ranked[n]:
                        dropped_assignments.append(
                            {
                                "patient": patient,
                                "stage": 1,
                                "category": category,
                                "template_index": t_idx,
                                "entities": entity,
                                "reason": "boundary_tie",
                                "evidence": {
                                    "cut_at_rank": n,
                                    "value_at_rank_n": round(float(ranked[n - 1]), 6),
                                    "value_at_rank_n_plus_1": round(
                                        float(ranked[n]), 6
                                    ),
                                    "gene_at_rank_n": pool[order[n - 1]],
                                    "gene_at_rank_n_plus_1": pool[order[n]],
                                    "units": gt.EXPRESSION_UNITS,
                                },
                            }
                        )
                        continue
                    ranking_answer_genes.append([pool[i] for i in order[:n]])
                    ranking_baseline_genes.append([pool[i] for i in order[:10]])

                plan.append(
                    {
                        "patient": patient,
                        "stage": 1,
                        "category": category,
                        "template_index": t_idx,
                        "entities": entity,
                    }
                )
                emitted += 1

        stats["stage1"][category] = {
            "eligible_patients": len(stage1_patients),
            "target": per_category_target,
            "emitted": emitted,
            "unmet_capacity": unmet,
            "dropped_boundary_tie": sum(
                1 for d in dropped_assignments if d["category"] == category
            ),
            "items_per_patient_mean": round(emitted / len(stage1_patients), 3),
            "entity_capacity_min": int(min(capacities)),
            "templates_total": n_templates,
            "templates_usable": len(usable),
            "templates_excluded": sorted(set(range(n_templates)) - set(usable)),
        }
        log(f"{category}: {emitted}")

    # --- Stage 2 -----------------------------------------------------------
    # One item per eligible patient per category. Every Stage 2 category's GT is
    # either a single label or a corpus-level constant, so a second item for the
    # same patient restates an identical fact in different words. That is the
    # phrasing-only repetition DIVERSITY_RULE exists to prevent.
    subtype_cap, capped_subtype_patients = apply_subtype_cap(
        pop["disease_subtype_classification"], SUBTYPE_CAP_PCT, rng
    )
    pop["disease_subtype_classification"] = capped_subtype_patients

    labels = gt._sample_labels()
    for category in (
        "disease_subtype_classification",
        "comparative_differential_reasoning",
        "gene_driver_reasoning",
    ):
        patients = pop[category]
        n_templates = len(templates[category])
        for patient in patients:
            entity: dict = {}
            template_index = rng.randrange(n_templates)
            if category == "comparative_differential_reasoning":
                subtype = str(labels.loc[patient, "cvd_subtype"])
                entity = {
                    "condition": (
                        subtype.replace("_", " ")
                        if subtype not in gt.NON_SUBTYPE_LABELS
                        else "cardiovascular disease"
                    ),
                    "comparison_group": "neg_hard",
                }
            elif category == "gene_driver_reasoning":
                entity = {"top_n": GENE_DRIVER_TOP_N[template_index]}
            plan.append(
                {
                    "patient": patient,
                    "stage": 2,
                    "category": category,
                    "template_index": template_index,
                    "entities": entity,
                }
            )
        stats["stage2"][category] = {
            "eligible_patients": len(patients),
            "emitted": len(patients),
            "items_per_patient": 1,
            "templates_total": n_templates,
        }
        log(f"{category}: {len(patients)}")

    # --- Task 5 measurement ------------------------------------------------
    def repetition(lists: list[list[str]]) -> dict:
        slots = Counter()
        for lst in lists:
            slots.update(lst)
        total = sum(slots.values())
        top20 = slots.most_common(20)
        mt_slots = sum(v for g, v in slots.items() if g.startswith("MT-"))
        return {
            "n_instances": len(lists),
            "total_answer_slots": total,
            "distinct_genes": len(slots),
            "top20_share": round(sum(v for _, v in top20) / total, 4),
            "mt_share": round(mt_slots / total, 4),
            "top20_genes": [g for g, _ in top20],
        }

    diversification = {
        "method": (
            "parameter-band sampling (approach a). ~45% of ranking instances "
            "target windows wider than the top 10 — count bands 25-100 or "
            "percentile bands 1-5% — instead of always asking for the top 10."
        ),
        "why_not_gene_exclusion": (
            "approach (b), excluding MT- genes from top-N answers, would require "
            "an exclusion parameter on ranking_query. That is a GT-layer change, "
            "out of scope for Step 2, and it would also make the answer disagree "
            "with the question ('the highest-expressed 10 genes' would silently "
            "omit the actual highest)."
        ),
        "baseline_all_top10": repetition(ranking_baseline_genes),
        "diversified": repetition(ranking_answer_genes),
    }
    stats["ranking_diversification"] = diversification

    # --- Task 6 totals -----------------------------------------------------
    stage1_total = sum(v["emitted"] for v in stats["stage1"].values())
    stage2_total = sum(v["emitted"] for v in stats["stage2"].values())
    stats["dropped_assignments"] = {
        "total": len(dropped_assignments),
        "by_reason": dict(Counter(d["reason"] for d in dropped_assignments)),
        "by_category": dict(Counter(d["category"] for d in dropped_assignments)),
        "log": "dropped_assignments.json",
    }
    stats["totals"] = {
        "stage1": {
            "target": STAGE1_TARGET,
            "achievable": stage1_total,
            "pct_of_target": round(100 * stage1_total / STAGE1_TARGET, 2),
        },
        "stage2": {
            "target": STAGE2_TARGET,
            "achievable": stage2_total,
            "pct_of_target": round(100 * stage2_total / STAGE2_TARGET, 2),
            "shortfall": STAGE2_TARGET - stage2_total,
            "phrasings_per_patient_to_hit_target": round(
                STAGE2_TARGET / stage2_total, 2
            ),
        },
        "grand_total": stage1_total + stage2_total,
    }

    stats["subtype_cap"] = subtype_cap
    stats["eligible_populations"] = {k: len(v) for k, v in pop.items()}
    stats["eligibility_detail"] = eligibility_detail
    stats["parameters"] = {
        "POPULATION_SCOPE": "per_category_maximal_eligibility",
        "STAGE1_PER_CATEGORY_SPLIT": "even",
        "SUBTYPE_CAP_PCT": SUBTYPE_CAP_PCT,
        "DIVERSITY_RULE": "one_template_per_patient_category_entity",
        "RANKING_DIVERSIFICATION": True,
        "seed": SEED,
        "interaction_n_choices": list(INTERACTION_N_CHOICES),
        "gene_driver_top_n_by_template": GENE_DRIVER_TOP_N,
    }

    finished = datetime.now(timezone.utc)
    stats["elapsed_seconds"] = round((finished - started).total_seconds(), 1)

    args.outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outdir / "generation_plan.json", "w") as fh:
        json.dump(
            {
                "generated": finished.isoformat(),
                "parameters": stats["parameters"],
                "summary": stats["totals"],
                "assignments": plan,
            },
            fh,
        )
    with open(args.outdir / "dropped_assignments.json", "w") as fh:
        json.dump(
            {
                "generated": finished.isoformat(),
                "purpose": (
                    "Assignments removed from generation_plan.json before Step 3, "
                    "with the evidence for each removal."
                ),
                "count": len(dropped_assignments),
                "dropped": dropped_assignments,
            },
            fh,
            indent=2,
        )
    with open(args.outdir / "generation_plan_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    log(f"stage1={stage1_total} stage2={stage2_total} total={len(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
