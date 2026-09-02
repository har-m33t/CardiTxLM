"""Is the near-chance binary result a broken scorer, or a real model failure?

WHY THIS EXISTS
---------------
`run_binary_cvd_eval.py` returned pooled ROC-AUC 0.512 on the holdout for a
model that was EXPLICITLY TRAINED on that task, with AUC swinging 0.461-0.512
across prompt phrasings — some of it below chance. Two very different things
produce that picture:

  (a) the model genuinely did not learn a generalizable disease signal, or
  (b) `TinyLlavaScorer.score` is wrong.

(b) is a live possibility, not a hedge: that class had never executed against a
real model before the run that produced these numbers — its author said so
plainly. Reporting (a) without excluding (b) would be reporting a bug as a
scientific finding.

THE TEST
--------
Score samples the model was TRAINED ON, using their own training questions and
answers. Training accuracy is not evidence of generalization and is not being
used as such — it is a functional check of the readout path:

  * near-chance on TRAINING data  -> the scorer is broken. Whatever the loss
    curve says the model fit, the scorer cannot read it back out, and the
    holdout number means nothing.
  * high on training, low on holdout -> the scorer works and the model really
    did fail to generalize. The holdout number stands.

Deliberately uses the exact (question, answer) strings from the training bundle
rather than the eval harness's paraphrases, so a failure cannot be blamed on
prompt drift between training and evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-ckpt", required=True)
    ap.add_argument("--stage2-json", type=Path,
                    default=REPO / "data/cvd_transcriptome/text_files/stage2_train_hypb.json")
    ap.add_argument("--plan", type=Path,
                    default=REPO / "scripts/hypothesis_b/discriminative_plan.json")
    ap.add_argument("--encoded-dir", type=Path,
                    default=REPO / "data/cvd_transcriptome/embeddings_encoded")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", type=Path,
                    default=REPO / "stage2_regen_report/tables/scorer_diagnostic.json")
    ap.add_argument("--conv-version", default="llama")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    from eval_binary_comparison.run_binary_cvd_eval import TinyLlavaScorer
    from eval_binary_comparison.evaluate_and_compare import load_model

    label_by_acc = {s["geo_accession"]: s["label"]
                    for s in json.loads(args.plan.read_text())["samples"]}

    # The discriminative items are exactly those whose sample is in the plan.
    items = [it for it in json.loads(args.stage2_json.read_text())
             if it["image"][:-4] in label_by_acc]
    print(f"discriminative training items available: {len(items):,}")

    rng = random.Random(20260902)
    rng.shuffle(items)
    picked, seen = [], set()
    for it in items:                       # one item per sample
        a = it["image"][:-4]
        if a in seen:
            continue
        seen.add(a)
        picked.append(it)
        if len(picked) >= args.n:
            break
    ys = np.array([label_by_acc[it["image"][:-4]] for it in picked])
    print(f"scoring {len(picked)} training samples "
          f"({ys.sum()} positive / {(1-ys).sum()} negative)")

    model, tokenizer = load_model(args.lora_ckpt)
    model = model.cuda().eval()
    scorer = TinyLlavaScorer(model, tokenizer, conv_version=args.conv_version)

    # Group by the exact question text so each batch shares one prompt.
    by_q: dict[str, list[int]] = {}
    for i, it in enumerate(picked):
        q = it["conversations"][0]["value"].replace("<image>\n", "")
        by_q.setdefault(q, []).append(i)

    pos_a = "Yes. This transcriptomic profile comes from a sample with confirmed cardiovascular disease."
    neg_a = "No. This transcriptomic profile shows no confirmed evidence of cardiovascular disease."

    scores = np.zeros(len(picked))
    for q, idxs in by_q.items():
        emb = np.stack([np.load(args.encoded_dir / picked[i]["image"]) for i in idxs])
        res = scorer.score(emb, q, [pos_a, neg_a])
        lp = res.sum_logprob if hasattr(res, "sum_logprob") else np.asarray(res)
        lp = np.asarray(lp, dtype=np.float64)
        for k, i in enumerate(idxs):
            scores[i] = lp[k, 0] - lp[k, 1]      # logP(yes) - logP(no)

    auc = float(roc_auc_score(ys, scores))
    acc = float(((scores > 0).astype(int) == ys).mean())
    verdict = ("SCORER LOOKS BROKEN — near chance on data the model was trained on; "
               "the holdout number cannot be interpreted"
               if auc < 0.60 else
               "SCORER WORKS — it reads the trained signal back out on training data, "
               "so the low holdout number is a real generalization failure")

    print(f"\ntraining-set AUC : {auc:.4f}")
    print(f"training-set acc : {acc:.4f}")
    print(f"n unique questions: {len(by_q)}")
    print(f"\n{verdict}")

    args.out.write_text(json.dumps({
        "purpose": "distinguish a broken forced-choice scorer from a real model failure",
        "n_scored": len(picked), "n_positive": int(ys.sum()),
        "n_unique_questions": len(by_q),
        "training_set_roc_auc": auc, "training_set_accuracy": acc,
        "holdout_pooled_roc_auc_for_reference": 0.5118492247774349,
        "threshold": 0.60,
        "verdict": verdict,
        "caveat": ("training-set performance is NOT evidence of generalization and "
                   "is not used as such here; this is a functional check of the "
                   "readout path only"),
    }, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
