"""run_latent_probe.py — linear probe on the trained multimodal LLM's latents.

Fits the same probe, folds and metric to three feature sets so the rows are
comparable by construction:

  * LLM-latent-imgtok   — final-layer hidden state at the expression-token position
  * LLM-latent-meanpool — final-layer hidden state mean-pooled over the sequence
  * BulkFormer-93M      — the frozen encoder embedding (CONTROL: this must
                          reproduce the number measured elsewhere, otherwise
                          something about this environment changed the result)

Reports the pooled binary CVD-vs-control AUROC (directly comparable to
linear_probe/results/BulkFormer-93M/neg_hard) and the per-condition breakdown.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval_binary_comparison.per_condition_probe import (
    CONDITIONS, load_variant, run_pooled_breakdown,
)

VARIANTS = ["LLM-latent-imgtok", "LLM-latent-meanpool", "BulkFormer-93M"]
DEFAULT_OUT = "eval_binary_comparison/llm_latent_probe.json"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    out = {}
    for tag in args.variants:
        p = Path("linear_probe/embeddings") / f"embeddings_{tag}.parquet"
        if not p.exists():
            print(f"[skip] {tag}: {p} missing", flush=True)
            continue
        print(f"\n=== {tag} ===", flush=True)
        X, meta = load_variant(p)
        print(f"  features: {X.shape}", flush=True)
        per, macro, pooled = run_pooled_breakdown(X, meta, "neg_hard", 5, 20260707)

        pb = pooled["roc_auc_mean"]
        pstd = pooled["roc_auc_std"]
        print(f"  POOLED BINARY CVD ROC-AUC = {pb:.4f} +/- {pstd:.4f}", flush=True)
        for key, label in CONDITIONS:
            v = per[key]["roc_auc_mean"]
            s = f"{v:.4f}" if v is not None else "n/a"
            print(f"    {label:22s} {s}", flush=True)
        print(f"    {'MACRO':22s} {macro:.4f}", flush=True)

        out[tag] = {
            "per_condition": {k: per[k] for k, _ in CONDITIONS},
            "macro_auroc": macro,
            "pooled_binary_roc_auc": pb,
            "pooled_binary_std": pstd,
            "pooled_binary_pr_auc": pooled.get("pr_auc_mean"),
            "n_features": int(X.shape[1]),
        }

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
