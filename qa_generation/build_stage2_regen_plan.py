"""Step 2, rebuilt for the Stage-2 regeneration (regen plan, Phases 1a + 2f).

Emits `stage2_regen_plan.json` — the (patient, category, template, entities)
assignments that `fill_templates.py` turns into QA pairs.

This is a NEW builder rather than an edit to `build_generation_plan.py`, on
purpose. That file is the provenance record of what produced the Stage-1 corpus,
which this regeneration does not touch and must not silently invalidate. Shared
logic is imported from it rather than copied.

Three things differ from the original Stage-2 plan.

1. HOLDOUT EXCLUSION (Phase 2f). Every sample in one of the 92 mixed GEO series
   is removed. Those series contain both probe positives and `neg_hard`
   negatives, which is exactly what makes them evaluable: a series carrying both
   classes cannot be separated on its batch signature alone. The previous
   session's best-looking number (0.9343) was an artifact of held-out positives
   whose series contained no negatives at all — see
   `eval_binary_comparison/comparison_report.md` §3. The split must exist before
   the retrain, so it is read from `holdout_series.json`, never recomputed here.

2. THE NEW `magnitude_reasoning` CATEGORY (Phase 2d), grounded in the effect-size
   distribution `build_per_sample_de.py` already produces.

3. MIXTURE REWEIGHTING (Phase 1a). Before the fix the corpus was 43.2%
   `comparative_differential_reasoning`, 43.2% `gene_driver_reasoning`, 13.6%
   `disease_subtype_classification` — and the first two were the degenerate
   ones, so 86.4% of supervision carried no per-sample information.

   Both are now genuinely per-sample, so the reweighting is a safety margin
   rather than the fix; the plan calls for it regardless. Rather than DISCARD
   per-sample items to shrink those categories, this raises the subtype share by
   giving each subtype-eligible sample several phrasings. Downweighting by
   throwing away real grounded supervision would be a poor trade — the corpus is
   small enough that every per-sample item is worth keeping.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from qa_generation import gt_functions as gt
from qa_generation.build_generation_plan import apply_subtype_cap

REPO = Path(__file__).resolve().parent.parent
QA = REPO / "qa_generation"
TEMPLATES = QA / "templates"
HOLDOUT = REPO / "data/cvd_transcriptome/holdout_series.json"
OUT = QA / "stage2_regen_plan.json"
STATS_OUT = QA / "stage2_regen_plan_stats.json"

SEED = 20260829
SUBTYPE_CAP_PCT = 35.0

#: Phrasings per sample, per category — the reweighting lever (see §3 above).
#: The three per-sample-DE categories get one item each; the subtype category
#: gets several, which raises its share without discarding grounded items.
TEMPLATES_PER_SAMPLE = {
    "disease_subtype_classification": 3,
    "comparative_differential_reasoning": 1,
    "gene_driver_reasoning": 1,
    "magnitude_reasoning": 1,
}

#: Answer size, keyed by template index, for the two categories that name a
#: bounded gene list. Keyed by template rather than randomised per item so that
#: one phrasing always carries one answer size: if "Identify the genes most
#: elevated..." sometimes returned 5 genes and sometimes 15, identical question
#: text would carry different answers — the same defect this regeneration
#: exists to remove, reintroduced at a smaller scale.
TOP_N_BY_TEMPLATE = {
    "comparative_differential_reasoning": lambda i: (5, 8, 10)[i % 3],
    "gene_driver_reasoning": lambda i: (8, 10, 15)[i % 3],
}

#: Which side of the deviation each template actually asks about, read off its
#: wording. Templates are NOT interchangeable on this axis: "Identify the genes
#: most elevated in this patient" and "...most reduced" must not both receive a
#: list ranked by absolute deviation, or the answer contradicts the question it
#: is paired with. That is the same defect as the fixed-string degeneracy — a
#: target that does not follow from its input — only harder to notice, so the
#: binding is explicit per index rather than defaulted.
#:
#: "both" renders elevated and reduced; "abs" ranks by |z| without committing to
#: a direction, for templates that ask which genes "stand out" or are "atypical".
DIRECTION_BY_TEMPLATE = {
    "comparative_differential_reasoning": {
        0: "both",  # "...which genes show the largest differences?"
        1: "both",  # "What specific expression changes distinguish this sample..."
        2: "up",    # "Identify the genes most elevated in this patient..."
        3: "down",  # "Identify the genes most reduced in this patient..."
        4: "both",  # "How large is the deviation... and which genes drive it?"
        5: "both",  # "...deviate most..., and in which direction?"
        6: "both",  # "...most differentially expressed genes?"
        7: "both",  # "Determine the gene expression differences..."
        8: "both",  # "Analyze the molecular features that distinguish..."
    },
    "gene_driver_reasoning": {
        0: "up",    # "...which are most elevated in this specific patient?"
        1: "abs",   # "...show notable deviation in this patient's profile?"
        2: "abs",   # "...which established disease-associated genes stand out?"
        3: "up",    # "Does this patient's profile show elevated expression...?"
        4: "down",  # "...which are most reduced in this patient?"
        5: "abs",   # "...are abnormal in this particular sample?"
        6: "abs",   # "...whose expression is most atypical in this profile."
        7: "abs",   # "...depart most from the comparison population?"
    },
}


def load_templates() -> dict[str, list[str]]:
    with open(TEMPLATES / "stage2.yaml") as fh:
        return yaml.safe_load(fh)


def holdout_accessions() -> set[str]:
    if not HOLDOUT.exists():
        raise SystemExit(
            f"{HOLDOUT} missing — run qa_generation/build_holdout_split.py first. "
            "The split must exist BEFORE the training corpus is built."
        )
    return set(json.loads(HOLDOUT.read_text())["holdout_geo_accession"])


def populations(held: set[str]) -> tuple[dict[str, list[str]], dict]:
    """Each category's eligible samples, minus the holdout.

    Gates are the GT functions' own, read off the label tables the same way
    `build_generation_plan.eligible_populations` does, then intersected with the
    samples that actually have a materialized `.npy` input — a training item
    needs an image even when its GT does not.
    """
    labels = gt._sample_labels()
    probe = gt._probe_labels()
    materialized = set(gt._expression_rows())
    de_ok = set(gt._de_rows().index[gt._de_rows().status == "ok"].astype(str))

    raw = {
        "disease_subtype_classification": set(
            labels[
                labels.is_cvd_disease & ~labels.cvd_subtype.isin(gt.NON_SUBTYPE_LABELS)
            ].geo_accession.astype(str)
        ),
        "comparative_differential_reasoning": set(
            probe[probe.is_positive].index.astype(str)
        ) & de_ok,
        "gene_driver_reasoning": set(
            labels[labels.is_cvd_disease].geo_accession.astype(str)
        ) & de_ok,
    }
    raw["magnitude_reasoning"] = set(raw["gene_driver_reasoning"])

    kept, dropped = {}, {}
    for cat, pop in raw.items():
        with_input = pop & materialized
        final = with_input - held
        kept[cat] = sorted(final)
        dropped[cat] = {
            "eligible_by_gate": len(pop),
            "without_materialized_input": len(pop - materialized),
            "removed_as_holdout": len(with_input & held),
            "retained": len(final),
        }
    return kept, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rng = random.Random(SEED)
    tpl = load_templates()

    # Fail loudly if a template was added without deciding what direction it
    # asks about. A missing entry would otherwise silently fall back to a
    # default and pair a "most reduced" question with an absolute-ranked answer.
    for cat, dirs in DIRECTION_BY_TEMPLATE.items():
        if set(dirs) != set(range(len(tpl[cat]))):
            raise SystemExit(
                f"DIRECTION_BY_TEMPLATE[{cat!r}] covers {sorted(dirs)} but "
                f"stage2.yaml has {len(tpl[cat])} templates. Every template "
                f"needs an explicit direction — see the comment on that map."
            )

    held = holdout_accessions()
    pop, dropped = populations(held)

    # Same cap the original plan applied: no single subtype may exceed 35% of
    # this category, so the classifier target is not dominated by heart failure.
    # Returns (stats, patients) — keep both; the stats go in the manifest so the
    # cap's effect on the label distribution stays auditable.
    cap_stats, pop["disease_subtype_classification"] = apply_subtype_cap(
        pop["disease_subtype_classification"], SUBTYPE_CAP_PCT, rng
    )

    assignments: list[dict] = []
    for category in (
        "disease_subtype_classification",
        "comparative_differential_reasoning",
        "gene_driver_reasoning",
        "magnitude_reasoning",
    ):
        n_templates = len(tpl[category])
        per_sample = TEMPLATES_PER_SAMPLE[category]
        for patient in pop[category]:
            # Deterministic per-patient choice: same seed, same corpus.
            picks = rng.sample(range(n_templates), min(per_sample, n_templates))
            for ti in picks:
                entities: dict = {}
                if category in (
                    "comparative_differential_reasoning",
                    "magnitude_reasoning",
                ):
                    entities["comparison_group"] = "neg_hard"
                if category in TOP_N_BY_TEMPLATE:
                    entities["top_n"] = TOP_N_BY_TEMPLATE[category](ti)
                if category in DIRECTION_BY_TEMPLATE:
                    entities["direction"] = DIRECTION_BY_TEMPLATE[category][ti]
                assignments.append(
                    {
                        "patient": patient,
                        "stage": 2,
                        "category": category,
                        "template_index": ti,
                        "entities": entities,
                    }
                )

    rng.shuffle(assignments)
    counts = Counter(a["category"] for a in assignments)
    total = len(assignments)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "purpose": "Stage-2 regeneration plan (regen_retrain_plan.md phases 1a, 2d, 2f)",
        "parameters": {
            "seed": SEED,
            "subtype_cap_pct": SUBTYPE_CAP_PCT,
            "templates_per_sample": TEMPLATES_PER_SAMPLE,
            "holdout_series_file": str(HOLDOUT.relative_to(REPO)),
            "n_holdout_accessions": len(held),
        },
        "summary": {
            "total": total,
            "by_category": dict(counts),
            "share_by_category": {
                k: round(100 * v / total, 2) for k, v in counts.items()
            },
            "population_after_holdout": {k: len(v) for k, v in pop.items()},
            "dropped": dropped,
            "subtype_cap": cap_stats,
        },
        "assignments": assignments,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    STATS_OUT.write_text(json.dumps(
        {k: v for k, v in payload.items() if k != "assignments"}, indent=2) + "\n")

    print(f"assignments: {total:,}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:38s} {v:6,}  {100*v/total:5.1f}%")
    print(f"\nholdout accessions excluded: {len(held):,}")
    print(f"wrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
