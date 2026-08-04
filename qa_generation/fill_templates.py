"""Step 3 — template filling.

For every assignment in `generation_plan.json`, substitutes the bound entities
into its template and computes ground truth by calling the matching
`gt_functions` function with exactly the parameters the plan recorded. Emits the
(templated_question, filled_question, answer) triples Step 4 paraphrases.

No LLM is involved. Answers here are rendered deterministically from the GT
payload — Step 4 rewords them, and is forbidden from changing any fact.

Field names follow `prompts/stage1_generation_prompt.txt` and
`prompts/stage2_generation_prompt.txt` exactly, since Step 4 formats those
prompts with these keys: Stage 1 uses `deterministic_answer`, Stage 2 uses
`gold_answer` plus `comparison_group`.

Assignments whose GT function returns a rejection are written to
`step3_excluded.jsonl` with the reason instead of being passed forward. A pair
with no answer would only be skipped by the generator anyway, at the cost of a
real API call.

Output is appended line-by-line and the run is resumable: each record carries its
`assignment_index`, and `--resume` skips indices already present in the outputs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from qa_generation import gt_functions as gt

REPO = Path(__file__).resolve().parent.parent
QA = REPO / "qa_generation"
PLAN = QA / "generation_plan.json"
TEMPLATES = QA / "templates"

STAGE1_OUT = QA / "filled_pairs_stage1.jsonl"
STAGE2_OUT = QA / "filled_pairs_stage2.jsonl"
EXCLUDED_OUT = QA / "step3_excluded.jsonl"

FLUSH_EVERY = 2000

#: Human-readable surface form for the only permitted comparison group. The
#: stage-2 prompt requires `{comparison_group}` to reach it already bound to a
#: real verified label; "neg_hard" is the internal column name and is kept in the
#: machine-readable `comparison_group` field instead.
NEG_HARD_SURFACE = "tissue-matched samples without confirmed cardiovascular disease"

STAGE1_CATEGORIES = {
    "direct_abundance_query",
    "threshold_query",
    "ranking_ordering_query",
    "comparative_query",
    "interaction_network_query",
}


def load_templates() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in ("stage1", "stage2"):
        with open(TEMPLATES / f"{name}.yaml") as fh:
            out.update(yaml.safe_load(fh))
    return out


# --- Placeholder binding ---------------------------------------------------


def fill(template: str, category: str, entities: dict) -> str:
    """Substitute bound entities into a template's placeholders."""
    text = template
    if "gene" in entities:
        text = text.replace("{gene}", str(entities["gene"]))
    if "gene_a" in entities:
        text = text.replace("{gene_A}", str(entities["gene_a"]))
    if "gene_b" in entities:
        text = text.replace("{gene_B}", str(entities["gene_b"]))
    if "threshold" in entities:
        text = text.replace("{threshold}", str(entities["threshold"]))
    if category == "interaction_network_query":
        text = text.replace("{N}", str(entities["n"]))
    if category == "ranking_ordering_query":
        spec = entities["spec"]
        if entities["shape"] == "percentile":
            text = text.replace("{percentile}", str(spec))
        else:
            text = text.replace("{N}", str(spec))
    if "condition" in entities:
        text = text.replace("{condition}", str(entities["condition"]))
    if "comparison_group" in entities:
        text = text.replace("{comparison_group}", NEG_HARD_SURFACE)
    return text


# --- Deterministic answer rendering ----------------------------------------
# These render the GT payload into prose. They add no fact the payload does not
# carry, and never truncate a list: a shortened gene list would present a
# partial answer as a complete one.


def _gene_values(genes: list[str], values: list[float]) -> str:
    return ", ".join(f"{g} ({v})" for g, v in zip(genes, values))


def render_stage1(category: str, payload: dict, entities: dict) -> str:
    units = payload["units"]

    if category == "direct_abundance_query":
        return (
            f"The expression level of {payload['gene']} in this sample is "
            f"{payload['expression']} ({units})."
        )

    if category == "threshold_query":
        direction = "above" if payload["direction"] == "above" else "below"
        return (
            f"{payload['n_matching']} genes in this sample have expression "
            f"{direction} {payload['threshold']} ({units}): "
            f"{_gene_values(payload['genes'], payload['expression'])}."
        )

    if category == "ranking_ordering_query":
        if payload["mode"] == "percentile":
            head = (
                f"The top {payload['percentile']}% of genes by expression in this "
                f"sample ({payload['n']} genes)"
            )
        else:
            head = f"The top {payload['n']} genes by expression in this sample"
        return f"{head}, ranked highest first, are: " + _gene_values(
            payload["genes"], payload["expression"]
        ) + f" ({units})."

    if category == "comparative_query":
        a, b = payload["gene_a"], payload["gene_b"]
        va, vb = payload["expression_a"], payload["expression_b"]
        if payload["equal"]:
            return (
                f"In this sample, {a} and {b} are expressed at the same level, "
                f"{va} ({units})."
            )
        return (
            f"In this sample, {a} is expressed at {va} ({units}) and {b} at "
            f"{vb} ({units}). {payload['higher']} is the higher of the two, by "
            f"{payload['difference']} {units} units "
            f"(log2 fold change {payload['log2_fold_change_a_vs_b']} for "
            f"{a} relative to {b})."
        )

    if category == "interaction_network_query":
        partners = payload["partners"]
        names = ", ".join(p["gene"] for p in partners)
        values = ", ".join(str(p["expression"]) for p in partners)
        return (
            f"{payload['gene']}'s top {payload['n_partners']} co-expressed "
            f"partners are {names}. In this sample, their expression levels are "
            f"{values} ({units}) respectively."
        )

    raise ValueError(f"no stage 1 renderer for {category}")


def render_stage2(category: str, payload: dict, entities: dict) -> str:
    if category == "disease_subtype_classification":
        return f"disease_confirmed_subtype: {payload['subtype']}"

    if category == "comparative_differential_reasoning":
        sep = payload["separability"]
        subtype = payload["sample_subtype"] or "cardiovascular disease"
        # Deliberately corpus-level. No per-gene differential exists for this
        # comparison — no expression matrix covers the neg_hard pool — so the
        # answer states the separability that was measured and explicitly says
        # the per-gene contrast is not established. See gt_functions_report.md.
        return (
            f"Relative to {NEG_HARD_SURFACE} (n = {payload['n_comparison']}, the "
            f"neg_hard pool), disease-confirmed cardiovascular samples "
            f"(n = {payload['n_positive']}) are separable at "
            f"ROC-AUC {sep['roc_auc_mean']} "
            f"(sd {sep['roc_auc_std']}, {sep['metric']}, {sep['primary_variant']}). "
            f"This sample is labelled {subtype}. A per-gene differential "
            f"expression contrast has not been computed for this comparison, so "
            f"no individual genes are attributed."
        )

    if category == "gene_driver_reasoning":
        genes = ", ".join(g["gene"] for g in payload["genes"])
        return (
            f"Top {payload['n_returned']} stable elastic-net signal genes for "
            f"cardiovascular disease vs. random bulk tissue (nonzero in all 5 CV "
            f"folds): {genes}."
        )

    raise ValueError(f"no stage 2 renderer for {category}")


# --- GT dispatch -----------------------------------------------------------


def compute_gt(category: str, patient: str, entities: dict):
    if category == "direct_abundance_query":
        return gt.direct_abundance_query(patient, entities["gene"])
    if category == "threshold_query":
        return gt.threshold_query(
            patient, entities["threshold"], entities["direction"]
        )
    if category == "ranking_ordering_query":
        return gt.ranking_query(patient, entities["spec"], entities["direction"])
    if category == "comparative_query":
        return gt.comparative_query(patient, entities["gene_a"], entities["gene_b"])
    if category == "interaction_network_query":
        return gt.interaction_network_query(patient, entities["gene"], entities["n"])
    if category == "disease_subtype_classification":
        return gt.disease_subtype_classification(patient)
    if category == "comparative_differential_reasoning":
        return gt.comparative_differential_reasoning(
            patient, entities["comparison_group"]
        )
    if category == "gene_driver_reasoning":
        return gt.gene_driver_reasoning(patient, entities["top_n"])
    raise ValueError(f"unknown category {category}")


def preflight(assignments: list[dict]) -> dict:
    """Confirm the plan is the closed Step 2 state before any work is done."""
    counts = Counter(a["category"] for a in assignments)
    driver = [a for a in assignments if a["category"] == "gene_driver_reasoning"]
    inter = [a for a in assignments if a["category"] == "interaction_network_query"]

    def bound_int(entity, key):
        v = entity.get(key)
        return isinstance(v, int) and not isinstance(v, bool)

    checks = {
        "ranking_ties_dropped": counts["ranking_ordering_query"] == 39954,
        "gene_driver_top_n_bound": all(bound_int(a["entities"], "top_n") for a in driver),
        "interaction_n_bound": all(bound_int(a["entities"], "n") for a in inter),
        "total_assignments": len(assignments),
        "expected_total": 219747,
        "total_matches": len(assignments) == 219747,
    }
    checks["passed"] = all(
        checks[k]
        for k in (
            "ranking_ties_dropped",
            "gene_driver_top_n_bound",
            "interaction_n_bound",
            "total_matches",
        )
    )
    return checks


def already_done(paths: list[Path]) -> set[int]:
    seen: set[int] = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        seen.add(json.loads(line)["assignment_index"])
                    except (json.JSONDecodeError, KeyError):
                        continue  # truncated final line from an interrupted run
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    plan = json.loads(PLAN.read_text())
    assignments = plan["assignments"]

    checks = preflight(assignments)
    print(f"preflight: {json.dumps(checks)}", flush=True)
    if not checks["passed"]:
        print("STOP: generation_plan.json is not the closed Step 2 state.")
        return 1

    templates = load_templates()
    outputs = [STAGE1_OUT, STAGE2_OUT, EXCLUDED_OUT]
    skip = already_done(outputs) if args.resume else set()
    if not args.resume:
        for path in outputs:
            if path.exists():
                path.unlink()
    else:
        print(f"resuming — {len(skip)} assignments already written", flush=True)

    stats = {
        "attempted": 0,
        "stage1_written": 0,
        "stage2_written": 0,
        "excluded": 0,
        "excluded_by_reason": Counter(),
        "excluded_by_category": Counter(),
        "written_by_category": Counter(),
        "answer_chars_by_category": Counter(),
        "skipped_resume": len(skip),
    }

    f1 = open(STAGE1_OUT, "a")
    f2 = open(STAGE2_OUT, "a")
    fx = open(EXCLUDED_OUT, "a")
    try:
        for index, assignment in enumerate(assignments):
            if args.limit is not None and stats["attempted"] >= args.limit:
                break
            if index in skip:
                continue

            category = assignment["category"]
            patient = assignment["patient"]
            entities = assignment["entities"]
            template = templates[category][assignment["template_index"]]
            stats["attempted"] += 1

            result = compute_gt(category, patient, entities)
            if not result.ok:
                fx.write(
                    json.dumps(
                        {
                            "assignment_index": index,
                            "sample_id": patient,
                            "category": category,
                            "template_index": assignment["template_index"],
                            "bound_entities": entities,
                            "reason": result.reason,
                            "status": result.status,
                        }
                    )
                    + "\n"
                )
                stats["excluded"] += 1
                stats["excluded_by_reason"][result.reason] += 1
                stats["excluded_by_category"][category] += 1
            else:
                filled = fill(template, category, entities)
                record = {
                    "assignment_index": index,
                    "sample_id": patient,
                    "category": category,
                    "template_index": assignment["template_index"],
                    "templated_question": template,
                    "filled_question": filled,
                    "bound_entities": entities,
                }
                if category in STAGE1_CATEGORIES:
                    answer = render_stage1(category, result.payload, entities)
                    record["deterministic_answer"] = answer
                    f1.write(json.dumps(record) + "\n")
                    stats["stage1_written"] += 1
                else:
                    answer = render_stage2(category, result.payload, entities)
                    record["gold_answer"] = answer
                    if category == "comparative_differential_reasoning":
                        record["comparison_group"] = entities["comparison_group"]
                    f2.write(json.dumps(record) + "\n")
                    stats["stage2_written"] += 1
                stats["written_by_category"][category] += 1
                stats["answer_chars_by_category"][category] += len(answer)

            if stats["attempted"] % FLUSH_EVERY == 0:
                f1.flush()
                f2.flush()
                fx.flush()
                print(
                    f"  {stats['attempted']}/{len(assignments)} "
                    f"(s1={stats['stage1_written']} s2={stats['stage2_written']} "
                    f"excl={stats['excluded']})",
                    flush=True,
                )
    finally:
        f1.close()
        f2.close()
        fx.close()

    finished = datetime.now(timezone.utc)
    stats["excluded_by_reason"] = dict(stats["excluded_by_reason"])
    stats["excluded_by_category"] = dict(stats["excluded_by_category"])
    stats["written_by_category"] = dict(stats["written_by_category"])
    stats["mean_answer_chars_by_category"] = {
        k: round(v / stats["written_by_category"][k])
        for k, v in stats["answer_chars_by_category"].items()
    }
    del stats["answer_chars_by_category"]
    stats["preflight"] = checks
    stats["elapsed_seconds"] = round((finished - started).total_seconds(), 1)
    (QA / "step3_stats.json").write_text(json.dumps(stats, indent=2))

    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
