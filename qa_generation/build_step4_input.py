"""Step 3 -> Step 4 adapter: filled pairs into DeepSeek request records.

`fill_templates.py` writes records keyed for its own bookkeeping
(`assignment_index`, `gold_answer`); `deepseek_client.py` requires `id`, `stage`
and `gt_answer`, plus the `image` field the training-bundle builder needs later.
The original run produced `step4_input_stage{1,2}.jsonl` in that shape, but no
script in the repo does the conversion — it was done ad hoc. This is that step,
written down, so the regeneration is reproducible.

`id` is `"<category>:<assignment_index>"`, matching the original files exactly.
The client uses it to resume: a run that dies partway re-reads its output and
skips ids already present, so the id must be stable across invocations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QA = REPO / "qa_generation"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=QA / "filled_pairs_stage2_regen.jsonl")
    ap.add_argument("--output", type=Path,
                    default=QA / "step4_input_stage2_regen.jsonl")
    ap.add_argument("--stage", type=int, default=2, choices=(1, 2))
    args = ap.parse_args()

    counts: Counter = Counter()
    seen_ids: set[str] = set()
    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = f"{r['category']}:{r['assignment_index']}"
            if rid in seen_ids:
                raise SystemExit(
                    f"duplicate request id {rid!r}. Ids must be unique or the "
                    f"client's resume logic will silently drop work."
                )
            seen_ids.add(rid)
            out = {
                "id": rid,
                "stage": args.stage,
                "category": r["category"],
                "templated_question": r["templated_question"],
                "filled_question": r["filled_question"],
                "gt_answer": r["gold_answer"],
                "image": f"{r['sample_id']}.npy",
                "assignment_index": r["assignment_index"],
                "sample_id": r["sample_id"],
            }
            # Carried through for the two categories bound to a comparison
            # group, so the prompt's "skip unless bound to neg_hard" rule has
            # something to check rather than being taken on trust.
            if r.get("comparison_group"):
                out["comparison_group"] = r["comparison_group"]
            fout.write(json.dumps(out) + "\n")
            counts[r["category"]] += 1

    total = sum(counts.values())
    print(f"wrote {total:,} requests to {args.output.relative_to(REPO)}")
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:38s} {n:6,d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
