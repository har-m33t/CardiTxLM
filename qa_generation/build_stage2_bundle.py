"""Step 5 — turn generated pairs into the Stage-2 training bundle.

Emits `data/cvd_transcriptome/text_files/stage2_train.json` in the shape
`tinyllava/data/dataset.py` expects:

    {"image": "GSM1126665.npy",
     "conversations": [{"from": "human", "value": "<image>\\n<question>"},
                       {"from": "gpt",   "value": "<answer>"}]}

Only `status == "ok"` records with a non-empty paraphrase are written. A skip or
an error is dropped, never patched with the gold answer: the point of the
paraphrase pass is fluency, and silently substituting the raw rendered GT for
records the model refused would put a different text distribution into part of
the corpus without that being visible anywhere.

Three assertions run before anything is written, because each guards a failure
that would be invisible in the trained model and expensive to discover later:

  holdout    no sample from the 92 reserved series may appear. This is the
             invariant the whole regeneration depends on; if it fails the
             evaluation is worthless and the retrain has to be redone.
  images     every referenced .npy must exist in the encoder cache, or training
             dies partway through an epoch on a missing file.
  degeneracy the distinct-answer ratio per category, which is the number that
             says the root cause is actually fixed. It is asserted here, not
             just reported, so a regression cannot reach a GPU.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from qa_generation.verify_regen_pairs import check as factual_check

REPO = Path(__file__).resolve().parent.parent
QA = REPO / "qa_generation"
DATA = REPO / "data/cvd_transcriptome"
HOLDOUT = DATA / "holdout_series.json"
ENCODED = DATA / "embeddings_encoded"

#: Minimum distinct-answer ratio per category, for the categories whose answers
#: are per-sample facts. Before the regeneration these sat at 0.0007 and 0.0004.
#: disease_subtype_classification is deliberately exempt: its answer is one of
#: five subtype labels, so a low ratio there is a classification target working
#: correctly, not degenerate supervision.
MIN_DISTINCT_RATIO = 0.90
PER_SAMPLE_CATEGORIES = {
    "comparative_differential_reasoning",
    "gene_driver_reasoning",
    "magnitude_reasoning",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=QA / "generated_pairs_stage2_regen.jsonl")
    ap.add_argument("--output", type=Path, default=DATA / "text_files/stage2_train.json")
    ap.add_argument("--zip", dest="do_zip", action="store_true", default=True)
    ap.add_argument("--report", type=Path, default=QA / "stage2_bundle_stats.json")
    args = ap.parse_args()

    held = set(json.loads(HOLDOUT.read_text())["holdout_geo_accession"])
    have_images = {p.name for p in ENCODED.glob("*.npy")}

    items: list[dict] = []
    answers: dict[str, list[str]] = defaultdict(list)
    status_counts: Counter = Counter()
    dropped: Counter = Counter()
    per_sample: Counter = Counter()
    rejected: list[dict] = []

    with args.input.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            status_counts[r.get("status")] += 1
            q = (r.get("paraphrased_question") or "").strip()
            a = (r.get("paraphrased_answer") or "").strip()
            if r.get("status") != "ok" or not q or not a:
                dropped[r.get("skip_reason") or r.get("status") or "empty"] += 1
                continue
            # Hard filter, not a warning. The paraphraser corrupts a gene symbol
            # or a number in a small fraction of calls (measured: 0.06%, e.g.
            # NCAPG -> NCGAP, RBMY1B -> RBM1Y8, 19191 -> 19184). Those are
            # fabricated biology dressed as fact, and this corpus exists to
            # remove exactly that class of defect -- so they are dropped rather
            # than repaired. Repairing would mean guessing which side is right.
            probs = factual_check(r)
            if probs:
                dropped[f"factual:{probs[0].split(':')[0]}"] += 1
                rejected.append({"id": r.get("id"), "problems": probs})
                continue
            image = r["image"]
            items.append({
                "image": image,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{q}"},
                    {"from": "gpt", "value": a},
                ],
            })
            answers[r["category"]].append(a)
            per_sample[image] += 1

    # --- assertions -------------------------------------------------------
    failures: list[str] = []

    leaked = sorted({it["image"][:-4] for it in items} & held)
    if leaked:
        failures.append(f"{len(leaked)} holdout samples in the bundle, "
                        f"e.g. {leaked[:3]}")

    missing = sorted({it["image"] for it in items} - have_images)
    if missing:
        failures.append(f"{len(missing)} referenced vectors absent from "
                        f"{ENCODED.name}, e.g. {missing[:3]}")

    ratios = {}
    for cat, a in answers.items():
        ratios[cat] = len(set(a)) / len(a)
        if cat in PER_SAMPLE_CATEGORIES and ratios[cat] < MIN_DISTINCT_RATIO:
            failures.append(
                f"{cat} distinct-answer ratio {ratios[cat]:.4f} is below "
                f"{MIN_DISTINCT_RATIO} — the degeneracy this regeneration "
                f"exists to fix has regressed"
            )

    print(f"generated records   {sum(status_counts.values()):,} {dict(status_counts)}")
    print(f"written             {len(items):,}")
    if dropped:
        print(f"dropped             {dict(dropped)}")
    print(f"unique samples      {len(per_sample):,}")
    if rejected:
        print(f"factually rejected  {len(rejected):,} "
              f"({100*len(rejected)/max(sum(status_counts.values()),1):.3f}%)")
        (QA / "stage2_regen_rejected.json").write_text(
            json.dumps(rejected, indent=2) + "\n")
    print("\ndistinct-answer ratio by category:")
    for cat, a in sorted(answers.items(), key=lambda kv: -len(kv[1])):
        mark = "" if cat not in PER_SAMPLE_CATEGORIES else "  <- per-sample gate"
        print(f"  {cat:38s} {len(a):6,d} items  ratio {ratios[cat]:.4f}{mark}")

    if failures:
        print()
        for f in failures:
            print(f"FAIL: {f}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(items, indent=1))
    print(f"\nwrote {args.output.relative_to(REPO)} ({args.output.stat().st_size/1e6:.1f} MB)")

    if args.do_zip:
        zpath = args.output.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(args.output, args.output.name)
        print(f"wrote {zpath.relative_to(REPO)} ({zpath.stat().st_size/1e6:.1f} MB)")

    args.report.write_text(json.dumps({
        "input": str(args.input.relative_to(REPO)),
        "n_generated": sum(status_counts.values()),
        "status_counts": dict(status_counts),
        "n_written": len(items),
        "dropped": dict(dropped),
        "n_factually_rejected": len(rejected),
        "n_unique_samples": len(per_sample),
        "items_by_category": {k: len(v) for k, v in answers.items()},
        "distinct_answer_ratio": ratios,
        "min_distinct_ratio_gate": MIN_DISTINCT_RATIO,
        "per_sample_categories": sorted(PER_SAMPLE_CATEGORIES),
        "holdout_series_file": str(HOLDOUT.relative_to(REPO)),
        "n_holdout_accessions": len(held),
        "assertions": {
            "no_holdout_leak": True,
            "all_images_present": True,
            "distinct_ratio_gate_met": True,
        },
    }, indent=2) + "\n")
    print(f"wrote {args.report.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
