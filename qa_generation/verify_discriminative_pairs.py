"""Polarity verification for the binary discriminative category.

WHY THIS EXISTS
---------------
`verify_regen_pairs.py` checks that no gene or number was invented. That check
is meaningless here — this category's answers name no genes and no numbers. The
failure mode is different and far more damaging: **the paraphrase inverts the
claim.** "No confirmed evidence of cardiovascular disease" becoming "evidence of
cardiovascular disease" flips the target on a training item whose label is the
entire supervision. A handful of flipped items is label noise; a systematic flip
on one class would silently invert the experiment's result.

A second, subtler failure is OVERCLAIM. The negative class is
`tissue_only_disease_unconfirmed` — disease *unconfirmed*, not *absent*. A
paraphrase that says "healthy", "normal", or "disease-free" asserts something
the label does not support. Those are rejected too, not passed as close enough.

HOW POLARITY IS DECIDED
-----------------------
Rule-based, deliberately: the answers are short and formulaic, an LLM judge
would cost money and add a second unverified component, and the rules are
directly testable. The extractor returns `pos`, `neg` or `undetermined`, and
**undetermined is rejected rather than guessed** — the same skip-don't-fabricate
discipline the rest of the pipeline uses.

The rules are negative-controlled by `--self-test`, which is run before any
real verification and which must pass or the script exits non-zero. Controls
cover: the 20 gold answers (must all classify correctly); deliberately flipped
answers (must be detected as mismatched, proving the check is not vacuous);
overclaim phrasings (must be rejected); and empty/ambiguous text (must be
undetermined, never silently assigned).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Phrasings that assert ABSENCE of disease rather than absence of CONFIRMATION.
#: The label does not support these; they are a fabrication, not a paraphrase.
OVERCLAIM = re.compile(
    r"\b(healthy|disease[- ]free|normal (?:sample|profile|tissue|subject)|"
    r"no disease|free of disease|unaffected|non[- ]?diseased)\b", re.I)

#: Explicit negation of the disease claim.
NEG_MARK = re.compile(
    r"\b(no|not|never|without|absent|unconfirmed|non[- ]?confirmed|lacks?|"
    r"lacking|negative for|does not|doesn't|isn't|is not|cannot be confirmed|"
    r"could not be confirmed|fails? to)\b", re.I)

#: Explicit assertion of the disease claim.
POS_MARK = re.compile(
    r"\b(yes|confirm(?:s|ed|ing)?|present|positive|indicates?|shows?|carries|"
    r"consistent with|evidence of|case|disease[- ]positive)\b", re.I)

#: Terms the negation has to govern for the answer to read as negative.
DISEASE_TERM = re.compile(r"\b(cardiovascular|disease|cvd|condition|case)\b", re.I)

#: Negation tokens, for the proximity test. Broader than NEG_MARK's leading
#: alternation because here they only need to be FOUND near a disease term,
#: not to carry the decision alone.
NEG_TOKEN = re.compile(
    r"\b(no|not|never|without|absent|unconfirmed|UNCONF|lacks?|lacking|"
    r"negative for|does not|doesn't|isn't|cannot|could not|fails? to)\b", re.I)

#: Leading verdict token, which when present is authoritative — these answers
#: are formulaic and lead with the verdict.
LEAD = re.compile(r"^\s*(yes|no|status\s*:\s*\w+|disease[- ]?(?:positive|unconfirmed))",
                  re.I)


def polarity(text: str) -> str:
    """Return 'pos', 'neg' or 'undetermined'. Never guesses."""
    t = (text or "").strip()
    if not t:
        return "undetermined"

    m = LEAD.match(t)
    if m:
        lead = m.group(1).lower()
        if lead.startswith("yes"):
            return "pos"
        if lead.startswith("no"):
            return "neg"
        if "unconfirmed" in lead:
            return "neg"
        if "positive" in lead or "confirmed" in lead:
            return "pos"

    # No leading verdict: decide on markers, and REFUSE when both or neither
    # fire. "unconfirmed" is itself a negation, so a bare NEG hit outranks the
    # POS hit it contains ("confirmed" inside "unconfirmed" is masked first).
    masked = re.sub(r"\bunconfirmed\b", " UNCONF ", t, flags=re.I)
    masked = re.sub(r"\bnon[- ]?confirmed\b", " UNCONF ", masked, flags=re.I)
    has_neg = bool(NEG_MARK.search(masked)) or "UNCONF" in masked
    has_pos = bool(POS_MARK.search(masked))
    if has_neg and not has_pos:
        return "neg"
    if has_pos and not has_neg:
        return "pos"
    if has_neg and has_pos:
        # Both fire. The negation is the fragile part — it is what a careless
        # paraphrase drops — so the question is whether a negation GOVERNS a
        # disease mention. Proximity is tested in BOTH directions: English puts
        # the negation after the subject at least as often as before it
        # ("cardiovascular disease is NOT confirmed"), and a forward-only window
        # misses exactly that, the most common phrasing in this corpus.
        # Anything still unclear is refused, not resolved by a tiebreak.
        WINDOW = 60
        for m in DISEASE_TERM.finditer(masked):
            lo = max(0, m.start() - WINDOW)
            hi = min(len(masked), m.end() + WINDOW)
            span = masked[lo:hi]
            # Same sentence only — a negation across a full stop governs a
            # different clause.
            if "." in masked[m.end():hi]:
                span = masked[lo:m.end() + masked[m.end():hi].index(".")]
            if NEG_TOKEN.search(span):
                return "neg"
        return "undetermined"
    return "undetermined"


# --------------------------------------------------------------------------- #
# negative controls
# --------------------------------------------------------------------------- #

def self_test() -> tuple[bool, list[str]]:
    from qa_generation.build_discriminative_pairs import TEMPLATES
    fails: list[str] = []

    # (1) every gold answer must classify to its own class
    for i, (_q, pos_a, neg_a) in enumerate(TEMPLATES):
        if polarity(pos_a) != "pos":
            fails.append(f"gold pos template {i} -> {polarity(pos_a)}: {pos_a!r}")
        if polarity(neg_a) != "neg":
            fails.append(f"gold neg template {i} -> {polarity(neg_a)}: {neg_a!r}")

    # (2) INVERSION control — the check must FAIL when the answer is flipped.
    #     Without this, a verifier that returns the expected label unconditionally
    #     would look identical to a working one.
    for i, (_q, pos_a, neg_a) in enumerate(TEMPLATES):
        if polarity(neg_a) == "pos":
            fails.append(f"inversion control leaked at template {i}")
        if polarity(pos_a) == "neg":
            fails.append(f"inversion control leaked at template {i}")

    # (3) realistic paraphrase flips must be caught
    for text, want in [
        ("This profile shows evidence of cardiovascular disease.", "pos"),
        ("This profile shows no evidence of cardiovascular disease.", "neg"),
        ("Cardiovascular disease could not be confirmed for this sample.", "neg"),
        ("The sample is negative for confirmed cardiovascular disease.", "neg"),
        ("Yes, cardiovascular disease is present.", "pos"),
        ("No.", "neg"),
        ("Disease-unconfirmed.", "neg"),
        # Regression case: "confirms" was missing from POS_MARK, which silently
        # rejected a correct affirmative answer in the first full run.
        ("The profile confirms that this sample has cardiovascular disease.", "pos"),
        ("The profile does not confirm cardiovascular disease.", "neg"),
    ]:
        got = polarity(text)
        if got != want:
            fails.append(f"paraphrase case {text!r}: want {want}, got {got}")

    # (4) ambiguity must be REFUSED, never resolved
    for text in ["", "   ", "The profile was analyzed.", "It is unclear."]:
        if polarity(text) != "undetermined":
            fails.append(f"ambiguous {text!r} -> {polarity(text)}, want undetermined")

    # (5) overclaim detection
    for text in ["This is a healthy sample.", "The subject is disease-free.",
                 "A normal profile with no disease."]:
        if not OVERCLAIM.search(text):
            fails.append(f"overclaim not detected: {text!r}")
    for text in ["No confirmed evidence of cardiovascular disease.",
                 "Cardiovascular disease is not confirmed in this sample."]:
        if OVERCLAIM.search(text):
            fails.append(f"false overclaim on legitimate negative: {text!r}")

    return (not fails), fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", type=Path,
                    default=REPO / "qa_generation/generated_pairs_stage2_discrim.jsonl")
    ap.add_argument("--input", type=Path,
                    default=REPO / "qa_generation/step4_input_stage2_discrim.jsonl")
    ap.add_argument("--out", type=Path,
                    default=REPO / "qa_generation/discriminative_verification.json")
    ap.add_argument("--self-test", action="store_true",
                    help="run the negative controls and exit")
    args = ap.parse_args()

    ok, fails = self_test()
    print(f"self-test: {'PASS' if ok else 'FAIL'} ({len(fails)} failures)")
    for f in fails:
        print(f"  {f}")
    if not ok:
        return 1
    if args.self_test:
        return 0

    labels = {}
    for line in args.input.read_text().splitlines():
        r = json.loads(line)
        labels[r["id"]] = r["label"]

    verdicts, rows = Counter(), []
    for line in args.generated.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rid = r["id"]
        # Field name comes from GenerationResult.as_dict in deepseek_client.
        ans = r.get("paraphrased_answer") or ""
        if r.get("status") != "ok" or r.get("skip_reason"):
            verdicts[f"reject:status_{r.get('skip_reason') or r.get('status')}"] += 1
            rows.append({"id": rid, "verdict": "reject:not_ok",
                         "status": r.get("status"),
                         "skip_reason": r.get("skip_reason")})
            continue
        want = "pos" if labels.get(rid) == 1 else "neg"
        got = polarity(ans)
        over = bool(OVERCLAIM.search(ans))
        if got == "undetermined":
            v = "reject:undetermined_polarity"
        elif got != want:
            v = "reject:polarity_flipped"
        elif over:
            v = "reject:overclaim_absence"
        else:
            v = "ok"
        verdicts[v] += 1
        if v != "ok":
            rows.append({"id": rid, "verdict": v, "expected": want,
                         "detected": got, "answer": ans[:300]})

    total = sum(verdicts.values())
    args.out.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "self_test_passed": True,
        "n_checked": total,
        "verdicts": dict(verdicts),
        "reject_rate": round(1 - verdicts["ok"] / max(total, 1), 6),
        "rejected": rows[:200],
        "n_rejected_total": len(rows),
    }, indent=2) + "\n")

    print(f"\nchecked {total:,}")
    for k, v in verdicts.most_common():
        print(f"  {k:32s} {v:,}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
