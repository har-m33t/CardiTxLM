"""Phase 2 — render the binary discriminative QA items (Hypothesis B).

Consumes `scripts/hypothesis_b/discriminative_plan.json` and emits a Step-4
input JSONL in the same schema as `step4_input_stage2_regen.jsonl`, so the
existing DeepSeek paraphrase pipeline runs over it unchanged.

WHAT THE LABEL ACTUALLY MEANS — the one thing to get right here
---------------------------------------------------------------
The positive class is this project's `disease_confirmed` curation. The negative
class is `neg_hard` == `cvd_subtype == "tissue_only_disease_unconfirmed"`:
samples from CVD-relevant tissue whose disease status was NOT confirmed by the
extended-EDA metadata curation.

**That is not the same as healthy, and the answers must never say it is.** An
answer claiming a negative sample is disease-free asserts something the label
does not support — a fabrication, and exactly the failure mode this project's
verification discipline exists to catch. Every negative answer below says
*unconfirmed* / *no evidence of*, never *absent* / *healthy* / *normal*.

ANSWER FORM IS KEYED TO TEMPLATE INDEX
--------------------------------------
Same discipline as `TOP_N_BY_TEMPLATE` in `build_stage2_regen_plan.py`: one
question phrasing always carries one answer form. If identical question text
sometimes produced one wording and sometimes another, identical inputs would
carry different targets — a smaller version of the defect the regeneration
removed. The paraphrase pass adds surface variety on top, under a polarity
check (`verify_discriminative_pairs.py`).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLAN = REPO / "scripts/hypothesis_b/discriminative_plan.json"
OUT = REPO / "qa_generation/step4_input_stage2_discrim.jsonl"
STATS = REPO / "qa_generation/discriminative_pairs_stats.json"

CATEGORY = "cvd_presence_discrimination"

#: The comparison basis, stated identically in every answer so the model is not
#: given a lexical cue that tracks the label.
BASIS = ("a tissue-matched comparison population of samples without confirmed "
         "cardiovascular disease")

#: (question, positive answer, negative answer). Index-locked — see docstring.
TEMPLATES: list[tuple[str, str, str]] = [
    ("Based on this transcriptomic profile, does this sample show evidence of cardiovascular disease?",
     "Yes. This transcriptomic profile comes from a sample with confirmed cardiovascular disease.",
     "No. This transcriptomic profile shows no confirmed evidence of cardiovascular disease."),
    ("Does this expression profile indicate the presence of cardiovascular disease?",
     "Yes — the profile is from a case with cardiovascular disease confirmed in the sample annotation.",
     "No — cardiovascular disease is not confirmed for this profile."),
    ("Is this sample from a patient with cardiovascular disease, or from a comparison sample without confirmed disease?",
     "This sample is from a patient with confirmed cardiovascular disease.",
     f"This sample belongs to {BASIS}."),
    ("Classify this transcriptomic profile as cardiovascular-disease-positive or disease-unconfirmed.",
     "Disease-positive: cardiovascular disease is confirmed for this sample.",
     "Disease-unconfirmed: no cardiovascular disease is confirmed for this sample."),
    ("Does this molecular profile carry evidence of a confirmed cardiovascular condition?",
     "Yes, this profile carries a confirmed cardiovascular condition.",
     "No, this profile carries no confirmed cardiovascular condition."),
    ("Assess whether cardiovascular disease is present in the sample this expression profile came from.",
     "Cardiovascular disease is present and confirmed in this sample.",
     "Cardiovascular disease is not confirmed in this sample."),
    ("From the gene expression alone, would you classify this sample as a cardiovascular disease case?",
     "Yes — this is a confirmed cardiovascular disease case.",
     "No — this is not a confirmed cardiovascular disease case."),
    ("Determine the cardiovascular disease status of the sample behind this transcriptomic profile.",
     "Status: cardiovascular disease confirmed.",
     "Status: cardiovascular disease unconfirmed."),
    ("Does this profile belong to a cardiovascular disease case or to a tissue-matched comparison sample?",
     "It belongs to a cardiovascular disease case.",
     f"It belongs to {BASIS}."),
    ("Is there evidence of cardiovascular disease in this transcriptomic sample?",
     "Yes, there is confirmed evidence of cardiovascular disease in this sample.",
     "No, there is no confirmed evidence of cardiovascular disease in this sample."),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, default=PLAN)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text())
    samples = plan["samples"]

    # Template assignment is deterministic in the sample's own position so a
    # rerun reproduces the corpus exactly, and is independent of the label so
    # phrasing cannot correlate with the answer.
    lines, by_tpl, by_label = [], {}, {0: 0, 1: 0}
    for i, s in enumerate(samples):
        t = i % len(TEMPLATES)
        question, pos_a, neg_a = TEMPLATES[t]
        answer = pos_a if s["label"] == 1 else neg_a
        lines.append(json.dumps({
            "id": f"{CATEGORY}:{i}",
            "stage": 2,
            "category": CATEGORY,
            "templated_question": question,
            "filled_question": question,       # no placeholders in this category
            "gt_answer": answer,
            "image": f"{s['geo_accession']}.npy",
            "label": s["label"],
            "sample_index": s["sample_index"],
            "series_id": s["series_id"],
            "tissue_coarse": s["tissue_coarse"],
            "template_index": t,
        }))
        by_tpl[t] = by_tpl.get(t, 0) + 1
        by_label[s["label"]] += 1

    args.out.write_text("\n".join(lines) + "\n")

    # Balance must hold WITHIN each template too: if one phrasing skewed
    # positive the model could answer from the question text alone.
    per_tpl_pos = {}
    for i, s in enumerate(samples):
        t = i % len(TEMPLATES)
        per_tpl_pos.setdefault(t, [0, 0])[s["label"]] += 1
    skew = {t: round(v[1] / (v[0] + v[1]), 4) for t, v in per_tpl_pos.items()}
    assert all(0.45 <= r <= 0.55 for r in skew.values()), f"template/label skew: {skew}"

    STATS.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "category": CATEGORY,
        "n_items": len(lines),
        "n_positive": by_label[1],
        "n_negative": by_label[0],
        "n_templates": len(TEMPLATES),
        "items_per_template": by_tpl,
        "positive_rate_per_template": skew,
        "n_distinct_gold_answers": len({json.loads(l)["gt_answer"] for l in lines}),
        "label_semantics": ("negative == cvd_subtype 'tissue_only_disease_unconfirmed' "
                            "— disease UNCONFIRMED, not healthy. No answer claims "
                            "absence of disease."),
        "assertions": {"template_label_balance": True},
    }, indent=2) + "\n")

    print(f"wrote {len(lines):,} items -> {args.out.relative_to(REPO)}")
    print(f"  {by_label[1]:,} positive / {by_label[0]:,} negative")
    print(f"  {len({json.loads(l)['gt_answer'] for l in lines})} distinct gold answers "
          f"over {len(TEMPLATES)} templates")
    print(f"  positive rate per template: {sorted(skew.values())[0]:.3f}"
          f"..{sorted(skew.values())[-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
