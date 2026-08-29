"""matched_binary_eval.py — put the Stage-2 LoRA model on the linear probe's
own axis: binary CVD-vs-control, ROC-AUC over the same population and folds.

WHY THIS EXISTS
---------------
The probe scores a continuous decision value over positives AND negatives. The
LLM had never been asked a binary question and had never seen a negative, so the
two were not comparable. This builds the missing path WITHOUT retraining:

  * same population   — is_positive + is_neg_hard from the probe's own embedding
                        table (8,725 + 22,307 = 31,032, matching its neg_hard run)
  * same grouping     — StratifiedGroupKFold(5, shuffle, seed) over series_id
  * same metric code  — linear_probe.probe._fold_metrics, imported, not rewritten
  * same input space  — the probe's stored 515-d vectors ARE the tower's pooled
                        output (extract.py mean-pools over genes exactly as
                        BulkFormerVisionTower.forward does), so they are fed
                        through the tower's pre-encoded passthrough unchanged.

FORCED CHOICE, NOT GENERATION. ROC-AUC needs a continuous score, so we run one
forward pass and read the logits for " Yes" and " No" at the answer position,
then softmax over just those two. No .generate(), no string matching.

THE LIMITATION THAT DOES NOT GO AWAY
------------------------------------
Stage 2 trained on 8,553 of the 8,725 positives (98.0%). The negative side is
fully held out (the model never saw a negative), but the positive side is not.
A model that memorised its positives can separate them from unseen negatives
without having learned anything transferable. So this script reports TWO
results and leads with the second:

  (a) FULL      — all 8,725 positives. Contaminated; an upper bound.
  (b) HELD-OUT  — only the 172 positives absent from stage2_train.json, against
                  all 22,307 negatives. Honest, but a small positive set, so the
                  confidence interval is wide.

Run:
    python -m eval_binary_comparison.matched_binary_eval \
        --lora-ckpt checkpoints/stage2-lora-bulkformer-93M \
        --embeddings linear_probe/embeddings/embeddings_BulkFormer-93M.parquet \
        --stage2-json data/cvd_transcriptome/text_files/stage2_train.json \
        --probe-results linear_probe/results/BulkFormer-93M/neg_hard/probe_results.json \
        --outdir eval_binary_comparison --batch-size 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

PROMPT = ("Based on this transcriptomic profile, does this sample show evidence "
          "of cardiovascular disease?")


def load_population(parquet_path, stage2_json):
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path).to_pydict()
    names = [n for n in t if n.startswith("e0")]
    names.sort()
    n = len(t["geo_accession"])
    keep = [i for i in range(n) if t["is_positive"][i] or t["is_neg_hard"][i]]
    emb = np.stack([np.array([t[c][i] for c in names], dtype=np.float32) for i in keep])
    meta = {
        "geo_accession": [t["geo_accession"][i] for i in keep],
        "series_id": [t["series_id"][i] for i in keep],
        "y": np.array([1 if t["is_positive"][i] else 0 for i in keep], dtype=int),
    }
    trained = {r["image"].replace(".npy", "") for r in json.load(open(stage2_json))}
    meta["seen_in_training"] = np.array(
        [1 if g in trained else 0 for g in meta["geo_accession"]], dtype=int)
    return emb, meta, names


def yes_no_token_ids(tokenizer):
    """First token of each answer word. Llama-family tokenizers prefix a word
    boundary, so encode the word alone and take token 0."""
    def first(word):
        ids = tokenizer.encode(word, add_special_tokens=False)
        return ids[0]
    return first("Yes"), first("No")


def build_prompt_ids(tokenizer, conv_version="llama"):
    from tinyllava.data.template import TemplateFactory
    template = TemplateFactory(conv_version)()
    q = "<image>\n" + PROMPT
    msg = [{"from": "human", "value": q}, {"from": "gpt", "value": ""}]
    try:
        enc = template.encode(msg, tokenizer, mode="eval")
        prompt = enc["prompt"] if isinstance(enc, dict) and "prompt" in enc else None
    except Exception:
        prompt = None
    if prompt is None:
        prompt = ("A chat between a curious user and an artificial intelligence "
                  "assistant. The assistant gives helpful, detailed, and polite "
                  f"answers to the user's questions. USER: {q} ASSISTANT:")
    return template.tokenizer_image_token(prompt, tokenizer, return_tensors="pt")


def score_samples(model, tokenizer, emb, batch_size, device="cuda"):
    """One forward pass per sample; softmax over the {Yes, No} logits only.
    Every prompt is textually identical, so all rows share one input_ids and the
    batch needs no padding."""
    yes_id, no_id = yes_no_token_ids(tokenizer)
    ids = build_prompt_ids(tokenizer).to(device)
    print(f"  yes_id={yes_id} ({tokenizer.decode([yes_id])!r})  "
          f"no_id={no_id} ({tokenizer.decode([no_id])!r})  prompt_len={ids.shape[0]}")

    out = np.zeros(len(emb), dtype=np.float64)
    model.eval()
    for i in range(0, len(emb), batch_size):
        chunk = emb[i:i + batch_size]
        b = len(chunk)
        images = torch.from_numpy(chunk).to(device=device, dtype=model.dtype)
        input_ids = ids.unsqueeze(0).expand(b, -1).contiguous().to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, images=images).logits
        last = logits[:, -1, :].float()
        pair = torch.stack([last[:, no_id], last[:, yes_id]], dim=-1)
        p_yes = torch.softmax(pair, dim=-1)[:, 1]
        out[i:i + b] = p_yes.cpu().numpy()
        if (i // batch_size) % 20 == 0:
            print(f"  scored {min(i+b, len(emb))}/{len(emb)}", flush=True)
    return out


def fold_metrics_like_probe(y, p, groups, seed, k=5):
    """Reuse the probe's own per-fold metric function and fold construction so
    the numbers are computed by identical code."""
    from sklearn.model_selection import StratifiedGroupKFold
    from linear_probe.probe import _fold_metrics

    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    rows = []
    for _, val_idx in skf.split(np.zeros(len(y)), y, groups):
        yv, pv = y[val_idx], p[val_idx]
        if len(set(yv.tolist())) < 2:
            continue
        rows.append(_fold_metrics(yv, pv))
    agg = {}
    for key in ("roc_auc", "pr_auc", "accuracy", "sensitivity", "specificity", "f1", "brier"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        agg[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        agg[f"{key}_std"] = float(np.std(vals)) if vals else None
    agg["n_folds"] = len(rows)
    return agg, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-ckpt", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--stage2-json", required=True)
    ap.add_argument("--probe-results", required=True)
    ap.add_argument("--outdir", default="eval_binary_comparison")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=None,
                    help="fold seed; defaults to the seed in the probe results file")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    probe = json.load(open(args.probe_results))
    seed = args.seed if args.seed is not None else probe.get("seed", 20260707)
    print(f"probe file: {args.probe_results}  seed={seed} k={probe.get('k_folds', 5)}")

    emb, meta, cols = load_population(args.embeddings, args.stage2_json)
    y, groups = meta["y"], np.array(meta["series_id"])
    seen = meta["seen_in_training"]
    print(f"population: {len(y)}  positives={int(y.sum())}  negatives={int((y==0).sum())}")
    print(f"positives seen in stage-2 training: {int(seen[y==1].sum())} / {int(y.sum())}")
    assert emb.shape[1] == 515, emb.shape

    from evaluate_and_compare import load_model
    model, tokenizer = load_model(args.lora_ckpt)

    p = score_samples(model, tokenizer, emb, args.batch_size)

    import csv
    with open(outdir / "matched_eval_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "series_id", "true_label", "predicted_probability",
                    "seen_in_stage2_training"])
        for g, s, yy, pp, sn in zip(meta["geo_accession"], meta["series_id"], y, p, seen):
            w.writerow([g, s, int(yy), f"{pp:.6f}", int(sn)])
    print(f"wrote {outdir/'matched_eval_results.csv'}")

    full, _ = fold_metrics_like_probe(y, p, groups, seed)
    mask = (y == 0) | (seen == 0)          # all negatives + unseen positives only
    held, _ = fold_metrics_like_probe(y[mask], p[mask], groups[mask], seed)
    print(f"\nFULL      (n={len(y)}, pos={int(y.sum())}): "
          f"ROC-AUC {full['roc_auc_mean']:.4f} +/- {full['roc_auc_std']:.4f}")
    print(f"HELD-OUT  (n={int(mask.sum())}, pos={int(y[mask].sum())}): "
          f"ROC-AUC {held['roc_auc_mean']:.4f} +/- {held['roc_auc_std']:.4f}")

    payload = {"seed": seed, "prompt": PROMPT,
               "population": {"total": int(len(y)), "positives": int(y.sum()),
                              "negatives": int((y == 0).sum()),
                              "positives_seen_in_training": int(seen[y == 1].sum())},
               "llm_full_contaminated": full, "llm_heldout_positives_only": held,
               "probe_neg_hard": probe["summary"]}
    (outdir / "metrics.json").write_text(json.dumps(payload, indent=2))

    try:
        make_plot(y, p, mask, probe, outdir)
    except Exception as e:
        print(f"[warn] plot skipped: {type(e).__name__}: {e}")
    return 0


def make_plot(y, p, mask, probe, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for lab, m in (("LLM: all positives (contaminated)", np.ones(len(y), bool)),
                   ("LLM: held-out positives only", mask)):
        fpr, tpr, _ = roc_curve(y[m], p[m])
        ax[0].plot(fpr, tpr, label=f"{lab} (AUC={roc_auc_score(y[m], p[m]):.3f})")
        pr, rc, _ = precision_recall_curve(y[m], p[m])
        ax[1].plot(rc, pr, label=lab)
    pa = probe["summary"]["roc_auc_mean"]
    ax[0].axhline(y=-1)  # no-op keeps legend ordering stable
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="chance (0.500)")
    ax[0].scatter([], [], marker="*", s=0)
    ax[0].set_title(f"ROC — probe neg_hard AUC = {pa:.3f}")
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend(fontsize=7)
    ax[1].set_title("Precision–Recall"); ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision"); ax[1].legend(fontsize=7)
    ax[0].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(outdir / "roc_pr_curves.png", dpi=150)
    print(f"wrote {outdir/'roc_pr_curves.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
