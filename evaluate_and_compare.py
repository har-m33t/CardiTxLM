"""evaluate_and_compare.py — score the Stage-2 LoRA model, and place it beside
the linear-probe evaluation floor.

READ THIS BEFORE QUOTING ANY NUMBER THIS SCRIPT PRINTS
------------------------------------------------------
Two limitations are structural, not cosmetic, and no flag turns them off:

1. THERE IS NO HELD-OUT SPLIT. The Stage-2 LoRA run consumed every one of the
   19,793 items in stage2_train.json. A split carved out now would still be data
   the model trained on, so every generation metric below is TRAINING-SET
   performance and is an upper bound, not an estimate of generalization.

2. THE PROBE BASELINE IS A DIFFERENT TASK ON A DIFFERENT SAMPLE POOL. The linear
   probe in linear_probe/results/BulkFormer-93M/ does binary CVD-vs-control
   classification over 31,032 (neg_hard) / 34,900 (neg_whole_corpus) samples with
   grouped 5-fold CV by series. This model names a cardiovascular condition for a
   sample drawn from the 8,553-sample CVD pool. The negatives the probe is scored
   against are not in this model's data at all. So the two numbers are NOT
   commensurable, and the script prints them side by side as CONTEXT, never as a
   head-to-head. Anyone reporting "the LLM beat the probe" from this output is
   misreading it.

What the script does measure honestly:
  * whether the trained model emits well-formed, on-task answers at all
  * exact-answer and condition-label agreement on disease_subtype_classification
  * gene-symbol recall on gene_driver_reasoning
  * qualitative samples for every category

Run:
    python evaluate_and_compare.py \
        --lora-ckpt checkpoints/stage2-lora-bulkformer-93M \
        --data data/cvd_transcriptome/text_files/stage2_train.json \
        --image-folder data/cvd_transcriptome/embeddings_encoded \
        --baseline-dir linear_probe/results/BulkFormer-93M \
        --n-per-category 60 --out eval_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

GENE_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,9}\b")
# Tokens that look like gene symbols but are not, in this corpus' phrasing.
GENE_STOP = {"ROC", "AUC", "TPM", "CVD", "RNA", "DNA", "THE", "AND", "FOR", "NOT",
             "TOP", "PCA", "SD", "CV", "ID"}


def normalize(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).strip()


def genes(t: str) -> set:
    return {g for g in GENE_RE.findall(t or "") if g not in GENE_STOP}


def build_condition_vocab(records) -> list:
    """Derive the condition label set from the corpus itself rather than
    hard-coding clinical terms we might get wrong."""
    vocab = set()
    pats = [r"consistent with ([a-z ]+?)[.,]", r"corresponds to ([a-z ]+?)[.,]",
            r"is labeled ([a-z ]+?)[.,]", r"most probable[^.]*?is ([a-z ]+?)[.,]"]
    for r in records:
        if r.get("_category") != "disease_subtype_classification":
            continue
        a = normalize(r["conversations"][1]["value"])
        for p in pats:
            for m in re.finditer(p, a + "."):
                v = m.group(1).strip()
                if 3 <= len(v) <= 60:
                    vocab.add(v)
    return sorted(vocab, key=len, reverse=True)


def extract_condition(text: str, vocab) -> str | None:
    n = normalize(text)
    for v in vocab:                     # longest-first, so specific beats generic
        if v in n:
            return v
    return None


def categorize(rec) -> str:
    """stage2_train.json ships {image, conversations} with no category field, so
    recover it. Key off the ANSWER, not the question: question phrasings vary
    ("distinguish"/"separate"/"differentiate") and misroute items, whereas each
    category's answer has an unmistakable fingerprint."""
    a = normalize(rec["conversations"][1]["value"])
    if "elastic net" in a or "cross validation folds" in a or "signal genes" in a:
        return "gene_driver_reasoning"
    if "neg hard" in a or "roc auc" in a or "compared" in a or "n 22 307" in a:
        return "comparative_differential_reasoning"
    return "disease_subtype_classification"


def load_model(path, device="cuda"):
    from tinyllava.model.load_model import load_pretrained_model
    model, tokenizer, image_processor, context_len = load_pretrained_model(path)
    # load_pretrained_model's LoRA branch constructs the model directly and only
    # calls .to(torch.float16) — its device_map kwargs apply solely to the
    # non-LoRA from_pretrained path. So the model comes back ON CPU, and
    # generating a merged 7B there is unusably slow. Move it explicitly.
    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(device=device, dtype=torch.float16)
    model.eval()
    print(f"model on {next(model.parameters()).device}, dtype {next(model.parameters()).dtype}")
    return model, tokenizer


def build_input_ids(question, tokenizer, conv_version="llama"):
    """Use the repo's own template so the eval prompt matches the training one."""
    from tinyllava.data.template import TemplateFactory
    template = TemplateFactory(conv_version)()
    msg = [{"from": "human", "value": question}, {"from": "gpt", "value": ""}]
    try:
        enc = template.encode(msg, tokenizer, mode="eval")
        if isinstance(enc, dict) and "input_ids" in enc:
            ids = enc["input_ids"]
            return ids if torch.is_tensor(ids) else torch.tensor(ids)
        prompt = enc["prompt"] if isinstance(enc, dict) else str(enc)
    except Exception:
        prompt = ("A chat between a curious user and an artificial intelligence "
                  "assistant. The assistant gives helpful, detailed, and polite "
                  f"answers to the user's questions. USER: {question} ASSISTANT:")
    return template.tokenizer_image_token(prompt, tokenizer, return_tensors="pt")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--baseline-dir", default="linear_probe/results/BulkFormer-93M")
    ap.add_argument("--n-per-category", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("STAGE-2 LoRA EVALUATION — TRAINING-SET ONLY, NOT A HELD-OUT ESTIMATE")
    print("=" * 78)

    records = json.load(open(args.data))
    for r in records:
        r["_category"] = categorize(r)
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["_category"]].append(r)
    print("corpus:", {k: len(v) for k, v in sorted(by_cat.items())})

    vocab = build_condition_vocab(records)
    print(f"condition labels derived from corpus: {len(vocab)}")

    rng = random.Random(args.seed)
    picks = []
    for cat, rs in sorted(by_cat.items()):
        picks += rng.sample(rs, min(args.n_per_category, len(rs)))
    print(f"evaluating {len(picks)} items\n")

    model, tokenizer = load_model(args.lora_ckpt)
    img_dir = Path(args.image_folder)

    results, per_cat = [], defaultdict(lambda: defaultdict(float))
    for i, rec in enumerate(picks):
        q = rec["conversations"][0]["value"]
        ref = rec["conversations"][1]["value"]
        vec = np.load(img_dir / rec["image"])
        images = torch.from_numpy(vec).unsqueeze(0).to(model.device, dtype=model.dtype)
        input_ids = build_input_ids(q, tokenizer).unsqueeze(0).to(model.device)
        with torch.inference_mode():
            out = model.generate(input_ids, images=images, do_sample=False,
                                 max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        # Decode defensively. HF generate normally returns prompt+continuation,
        # but TinyLlava's multimodal path folds input_ids into inputs_embeds and
        # returns ONLY the new tokens. Blindly slicing at input_ids.shape[1]
        # then amputates the front of every answer. Slice only when the prompt
        # is actually echoed back.
        seq = out[0]
        n_in = input_ids.shape[1]
        if seq.shape[0] > n_in and torch.equal(seq[:n_in].cpu(), input_ids[0].cpu()):
            seq = seq[n_in:]
        gen = tokenizer.decode(seq, skip_special_tokens=True).strip()

        cat = rec["_category"]
        row = {"category": cat, "image": rec["image"], "question": q,
               "reference": ref, "prediction": gen}
        per_cat[cat]["n"] += 1
        per_cat[cat]["exact"] += float(normalize(gen) == normalize(ref))
        if cat == "disease_subtype_classification":
            cr, cp = extract_condition(ref, vocab), extract_condition(gen, vocab)
            row["ref_condition"], row["pred_condition"] = cr, cp
            per_cat[cat]["cond_parsed"] += float(cp is not None)
            per_cat[cat]["cond_correct"] += float(cr is not None and cr == cp)
        if cat == "gene_driver_reasoning":
            gr, gp = genes(ref), genes(gen)
            row["gene_recall"] = len(gr & gp) / max(len(gr), 1)
            per_cat[cat]["gene_recall"] += row["gene_recall"]
        results.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(picks)} generated", flush=True)

    print("\n" + "=" * 78)
    print("GENERATION METRICS (training-set; upper bound, not generalization)")
    print("=" * 78)
    summary = {}
    for cat, d in sorted(per_cat.items()):
        n = d["n"]
        s = {"n": int(n), "exact_match": round(d["exact"] / n, 4)}
        if cat == "disease_subtype_classification":
            s["condition_parse_rate"] = round(d["cond_parsed"] / n, 4)
            s["condition_accuracy"] = round(d["cond_correct"] / n, 4)
        if cat == "gene_driver_reasoning":
            s["mean_gene_recall"] = round(d["gene_recall"] / n, 4)
        summary[cat] = s
        print(f"  {cat}: {s}")

    print("\n" + "=" * 78)
    print("LINEAR-PROBE EVALUATION FLOOR (BulkFormer-93M) — CONTEXT, NOT A RIVAL SCORE")
    print("=" * 78)
    baseline = {}
    for pool in ("neg_hard", "neg_whole_corpus"):
        p = Path(args.baseline_dir) / pool / "probe_results.json"
        if not p.exists():
            print(f"  {pool}: MISSING ({p})")
            continue
        b = json.load(open(p))["summary"]
        baseline[pool] = {k: b[k] for k in ("roc_auc_mean", "roc_auc_std",
                                            "pr_auc_mean", "pr_auc_std")}
        print(f"  {pool}: ROC-AUC {b['roc_auc_mean']:.4f} +/- {b['roc_auc_std']:.4f} | "
              f"PR-AUC {b['pr_auc_mean']:.4f} +/- {b['pr_auc_std']:.4f}")
    print("  ^ binary CVD-vs-control over 31k-35k samples, grouped 5-fold by series.")
    print("    Different task, different pool. NOT comparable to the rates above.")

    print("\n--- sample generations ---")
    shown = set()
    for r in results:
        if r["category"] in shown:
            continue
        shown.add(r["category"])
        print(f"\n[{r['category']}] {r['image']}")
        print(f"  Q   : {r['question'][:150].replace(chr(10), ' ')}")
        print(f"  REF : {r['reference'][:220]}")
        print(f"  PRED: {r['prediction'][:220]}")

    payload = {
        "limitations": [
            "No held-out split: the LoRA run trained on all 19,793 stage-2 items, "
            "so every generation metric here is training-set performance and an "
            "upper bound, not a generalization estimate.",
            "The linear-probe baseline is a different task (binary CVD vs control) "
            "on a different sample pool (31k-35k samples incl. negatives absent "
            "from this model's training data). It is context, not a comparison.",
        ],
        "n_evaluated": len(results),
        "generation_metrics": summary,
        "linear_probe_baseline_93M": baseline,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
