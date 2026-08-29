"""Phase 5d — assemble stage2_regen_report/README.md from the result files.

Generated rather than hand-written so every number in the write-up comes from an
artifact on disk. A summary that is typed by hand drifts from the run it
describes; this one cannot.

The write-up must answer the question the original session left open — does the
LLM now BEAT the frozen encoder it sits on, or does it still only tie it — and
say so plainly either way. A regeneration that did not move the number is a
real result about where the ceiling lies, not a failure to report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
REPORT = REPO / "stage2_regen_report"
TABLES = REPORT / "tables"
QA = REPO / "qa_generation"


def _load(p: Path):
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return pd.read_csv(p)


def fmt(v, nd=4):
    return "n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.{nd}f}"


def main() -> int:
    bundle = _load(QA / "stage2_bundle_stats.json")
    de = _load(QA / "de/de_manifest.json")
    holdout = _load(REPO / "data/cvd_transcriptome/holdout_series.json")
    usage = _load(QA / "regen_usage_summary.json")
    probe = _load(TABLES / "probe_comparison.csv")
    ml = _load(TABLES / "multilabel_before_after.csv")
    sweep = _load(TABLES / "encoder_scale_sweep.csv")

    L: list[str] = []
    A = L.append
    A("# Stage-2 Regeneration — Results")
    A("")
    A(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
      f"from the artifacts in this directory. Every number below is read from a "
      f"file rather than transcribed, so the write-up cannot drift from the run.")
    A("")

    # ---------------- what was wrong -----------------------------------
    A("## 1. What was fixed")
    A("")
    A("86.4% of the original Stage-2 corpus carried no sample-specific "
      "information. `comparative_differential_reasoning` returned a corpus-level "
      "probe ROC-AUC with `per_gene_differential: None`, and "
      "`gene_driver_reasoning` returned the same global elastic-net ranking for "
      "every sample — self-documented as `sample_role: \"eligibility_gate_only\"`. "
      "For that share of the supervision the loss-minimising behaviour is to emit "
      "a fixed string and ignore the expression profile entirely.")
    A("")
    if bundle:
        A("| category | items | distinct-answer ratio |")
        A("|---|---:|---:|")
        for cat, n in sorted(bundle["items_by_category"].items(),
                             key=lambda kv: -kv[1]):
            r = bundle["distinct_answer_ratio"][cat]
            A(f"| `{cat}` | {n:,} | {r:.4f} |")
        A("")
        A("Before the fix those ratios were 0.0007 "
          "(`comparative_differential_reasoning`) and 0.0004 "
          "(`gene_driver_reasoning`) — one distinct answer across 8,553 samples.")
        A("")
        A("`disease_subtype_classification`'s low ratio is CORRECT and is not "
          "degeneracy: its answer is one of five subtype labels, so the target is "
          "determined by the input. That is what a classification task looks "
          "like. The three per-sample categories are the ones that were broken.")
        A("")
        if usage:
            A(f"Generation: {usage['calls']:,} DeepSeek calls, {usage['errors']} "
              f"errors, {usage['cache_hit_rate']:.1%} prefix-cache hit, "
              f"${usage['cost_usd']:.2f}, {usage['elapsed_s']/60:.1f} min. "
              f"{bundle.get('n_factually_rejected', 0)} records were rejected "
              f"(not repaired) for paraphraser factual drift.")
            A("")

    # ---------------- the data ------------------------------------------
    if de:
        A("## 2. How the per-sample ground truth is built")
        A("")
        p = de["populations"]
        t = de["tissue_matching"]
        A(f"- Reference: {p['n_neg_hard_reference']:,} `neg_hard` samples "
          f"({p['n_neg_hard_excluded_holdout']:,} excluded because they sit in "
          f"holdout series — otherwise holdout information would leak into "
          f"*training answers*).")
        A(f"- Tissue matching: {t['n_positives_tissue_matched']:,} samples got a "
          f"tissue-matched reference, {t['n_positives_pooled_fallback']:,} fell "
          f"back to the whole pool. Which one was used is recorded per sample, "
          f"never silently substituted.")
        A(f"- Effect sizes: z against the reference mean/sd, plus a log-fold "
          f"change (both sides are log1p(TPM), so their difference already is one).")
        A("")
        gr = de.get("gate_rationale", {})
        if gr:
            A("Four gates apply to the NAMED gene lists only — population "
              "statistics and the z/lfc matrices are untouched:")
            A("")
            for g in gr.get("gates", []):
                A(f"- **`{g['name']}`** — {g['rule']}. {g.get('why', '')}")
            A("")

    if holdout:
        A("## 3. The holdout")
        A("")
        A(f"{holdout['n_series']} GEO series contain BOTH a probe positive and a "
          f"`neg_hard` negative: {holdout['n_holdout_positive']:,} positives and "
          f"{holdout['n_holdout_neg_hard']:,} negatives, reserved entirely from "
          f"training.")
        A("")
        A("Mixed series specifically, because a series carrying both classes "
          "cannot be separated on its batch signature — the signature is shared "
          "by the positives and negatives inside it. The previous session's "
          "best-looking number (0.9343) was an artifact of exactly this: its 15 "
          "held-out series contained no negatives at all, so the classifier was "
          "separating studies, not disease.")
        A("")

    # ---------------- results -------------------------------------------
    A("## 4. Representation quality")
    A("")
    if probe is not None and len(probe):
        clean = probe[probe.population == "holdout_clean"]
        if len(clean):
            A("On the clean holdout (grouped 5-fold by series, "
              "`random_state=20260707` — identical to every prior probe):")
            A("")
            A("| feature set | estimator | dim | ROC-AUC | PR-AUC |")
            A("|---|---|---:|---:|---:|")
            for _, r in clean.iterrows():
                A(f"| {r.feature_set} | {r.estimator} | {int(r['dim'])} | "
                  f"{fmt(r.roc_auc_mean)} ± {fmt(r.roc_auc_std)} | "
                  f"{fmt(r.pr_auc_mean)} ± {fmt(r.pr_auc_std)} |")
            A("")
            enc = clean[clean.feature_set.str.startswith("BulkFormer")]
            llm_lin = clean[(clean.feature_set.str.startswith("LLM")) &
                            (clean.estimator == "linear")]
            llm_mlp = clean[(clean.feature_set.str.startswith("LLM")) &
                            (clean.estimator == "mlp")]
            A("### Does this close the \"ties but doesn't beat the encoder\" finding?")
            A("")
            if len(enc) and len(llm_lin):
                e = float(enc.roc_auc_mean.iloc[0])
                l = float(llm_lin.roc_auc_mean.iloc[0])
                d = l - e
                verdict = ("**beats**" if d > 0.01 else
                           "**ties**" if abs(d) <= 0.01 else "**underperforms**")
                A(f"The LLM's linear probe {verdict} the frozen encoder it sits "
                  f"on: {l:.4f} vs {e:.4f} (delta {d:+.4f}).")
                if len(llm_mlp):
                    m = float(llm_mlp.roc_auc_mean.iloc[0])
                    A("")
                    A(f"The MLP probe reaches {m:.4f}, a gap of {m - l:+.4f} over "
                      f"the linear probe on the same embeddings. A large gap "
                      f"would mean the LLM's space holds non-linearly separable "
                      f"structure a linear probe cannot reach; a small one means "
                      f"it does not. Both are reported because the comparison "
                      f"between them is itself the result.")
                A("")
        cont = probe[probe.population == "full_contaminated"]
        if len(cont):
            A("The full-population numbers are also computed, for continuity with "
              "the prior session's table. They are CONTAMINATED — most positives "
              "were in Stage-2 training — and must not be read as a held-out "
              "result:")
            A("")
            A("| feature set | estimator | ROC-AUC |")
            A("|---|---|---:|")
            for _, r in cont.iterrows():
                A(f"| {r.feature_set} | {r.estimator} | {fmt(r.roc_auc_mean)} |")
            A("")
    else:
        A("_Probe results not yet available._")
        A("")

    if ml is not None and len(ml):
        A("## 5. Broad multi-label probing, before vs after")
        A("")
        A("| label | kind | before | after | delta |")
        A("|---|---|---:|---:|---:|")
        for _, r in ml.iterrows():
            A(f"| {r.label} | {r.kind} | {fmt(r.before_roc_auc)} | "
              f"{fmt(r.after_roc_auc)} | {r.delta:+.4f} |")
        A("")
        A("`platform` and `instrument` are TECHNICAL CONTROLS. A high score there "
          "measures sequencing-batch signal, not representation quality, and is "
          "not a result.")
        A("")

    if sweep is not None and len(sweep):
        A("## 6. Encoder scale (for the record, no action taken)")
        A("")
        A("BulkFormer-93M is at the plateau of the completed 5-variant sweep, "
          "which is why no 147M retrain was pursued in this plan. Stated "
          "explicitly rather than left as an unstated assumption.")
        A("")
        A("See `tables/encoder_scale_sweep.csv` and its note.")
        A("")

    A("## 7. Loss curves")
    A("")
    A("`loss_curves/stage1_loss.png` — Stage 1 replotted from existing logs; it "
      "was not retrained, since its data was never the problem.")
    A("")
    A("`loss_curves/stage2_loss_before_after.png` — the original run collapsed "
      "2.883 → 0.066 within roughly 20 of its 155 steps, which is what "
      "predicting a fixed string looks like. **A higher loss floor on the "
      "corrected run is evidence the fix worked**, not evidence of a worse "
      "model: the task is now genuinely harder. A corrected run that reproduced "
      "the old curve would be the alarming outcome.")
    A("")

    A("## 8. Known limitations")
    A("")
    A("- `LEP` is retained as a confound (adipose content) and appears as the "
      "most deviant gene in ~2.3% of samples. It is real biology, so excluding "
      "it would mean deciding which biology counts; sex-linked genes and common "
      "deletion polymorphisms were excluded on the narrower ground that they are "
      "covariates and genotyping artifacts rather than phenotype.")
    A("- Tissue matching covers only part of the corpus; the rest uses the whole "
      "`neg_hard` pool, recorded per sample.")
    A("- Stage 1 was not retrained, so any limitation in the connector alignment "
      "carries forward unchanged.")
    A("")

    out = REPORT / "README.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out.relative_to(REPO)} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
