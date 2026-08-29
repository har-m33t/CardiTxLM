"""Factual verification of the regenerated Stage-2 pairs.

The paraphraser is only ever allowed to reword. It must not invent a gene, drop
one, or change a number — and the whole point of this regeneration is that the
answers now carry real per-sample facts, so an unchecked paraphrase would put
fabricated biology into the training targets while looking fluent.

Three checks, each comparing the generated answer against its own gold answer:

  genes    every gene symbol named in the paraphrase must appear in the gold
           answer, and every gene in the gold answer must survive into the
           paraphrase. A dropped gene turns a complete answer into a partial
           one presented as complete; an added gene is a fabrication.
  numbers  every numeric literal in the paraphrase must appear in the gold
           answer. Thousands separators are normalised (the prompt permits
           "22,307" for 22307) but rounding is not: 0.78 for 0.7806 fails.
  skips    an `insufficient_data` gold answer must produce a SKIP, never prose.

Gene detection is deliberately conservative — an uppercase alphanumeric token of
2+ chars that appears in the gold answer's own gene list — so ordinary words and
units are not mistaken for symbols, and the check cannot fail for a reason that
has nothing to do with fabrication.

Run:
    python -m qa_generation.verify_regen_pairs --input <results.jsonl>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: A gene as this corpus renders it. The gold answers always write
#: "SYMBOL (z = ...", but the paraphraser legitimately rewrites that as
#: "SYMBOL at z = ..." — so both forms must be recognised, or a corrupted
#: symbol in the reworded form slips through. RBMY1B -> RBM1Y8 did exactly
#: that, and was only caught incidentally by the numeric check.
GENE_IN_GOLD = re.compile(r"\b([A-Z][A-Z0-9\-]{1,14})\s*(?:\(|\bat\s+z\s*[=<>])")
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

#: Tokens that look like symbols but are not, and must not be treated as genes.
NOT_GENES = {"OK", "AND", "THE", "ROC", "AUC", "TPM", "CV", "SD", "DNA", "RNA"}


def _numbers(text: str) -> set[str]:
    out = set()
    for m in NUMBER.findall(text):
        n = m.rstrip(".").replace(",", "")
        if n and n not in ("-",):
            out.add(n)
    return out


def check(rec: dict) -> list[str]:
    gold = rec.get("gt_answer") or ""
    gen = rec.get("paraphrased_answer") or ""
    status = rec.get("status")
    problems: list[str] = []

    if "insufficient_data" in gold.lower():
        if status != "skipped" and gen.strip():
            problems.append("gold was insufficient_data but an answer was generated")
        return problems

    if status != "ok" or not gen.strip():
        return problems  # error/skip rows are counted separately, not failures here

    gold_genes = {g for g in GENE_IN_GOLD.findall(gold) if g not in NOT_GENES}
    if gold_genes:
        # Only symbols the paraphrase actually PRESENTS as genes (followed by
        # "(" or "at z =") are candidates, so ordinary capitalised words are
        # never mistaken for fabricated symbols.
        named_in_gen = set(GENE_IN_GOLD.findall(gen)) - NOT_GENES
        invented = named_in_gen - gold_genes
        missing = {g for g in gold_genes if not re.search(rf"\b{re.escape(g)}\b", gen)}
        if invented:
            problems.append(f"invented gene(s): {sorted(invented)[:5]}")
        if missing:
            problems.append(f"dropped gene(s): {sorted(missing)[:5]}")

    gold_nums, gen_nums = _numbers(gold), _numbers(gen)
    altered = gen_nums - gold_nums
    if altered:
        problems.append(f"number(s) not in gold: {sorted(altered)[:5]}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--show", type=int, default=5, help="print this many failures")
    args = ap.parse_args()

    n = 0
    status_counts: Counter = Counter()
    by_category: Counter = Counter()
    failures: list[tuple[str, list[str]]] = []
    fail_by_cat: Counter = Counter()

    with args.input.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            status_counts[rec.get("status")] += 1
            by_category[rec.get("category")] += 1
            probs = check(rec)
            if probs:
                failures.append((rec.get("id", "?"), probs))
                fail_by_cat[rec.get("category")] += 1

    print(f"records            {n:,}")
    print(f"status             {dict(status_counts)}")
    print(f"by category        {dict(by_category)}")
    print(f"factual failures   {len(failures):,}  ({100*len(failures)/max(n,1):.2f}%)")
    if fail_by_cat:
        print(f"failures by cat    {dict(fail_by_cat)}")
    for rid, probs in failures[: args.show]:
        print(f"  {rid}: {'; '.join(probs)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
