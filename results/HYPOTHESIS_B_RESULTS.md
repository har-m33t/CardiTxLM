# Hypothesis B — Results

**Date:** 2026-09-02 · **Encoder:** BulkFormer-93M (locked, frozen) ·
**Backbone:** Vicuna-7B v1.5 + LoRA r=128/α=256 (unchanged)
**Hardware:** 4× NVIDIA L40S 46 GB · **Spend: $5.75 GPU + $0.25 DeepSeek**

Criteria for what would count as support were fixed in
`.claude/hypothesis_b_interpretation_criteria.md` **before any Phase 4 number
existed** (commit `f6daba9`, which contains no result artifact). What follows is
those criteria applied.

---

## 0. The one-paragraph version

**Hypothesis B is supported on its primary (pooled) metric, does NOT survive the
stricter batch-controlled metric, and the most important finding is the one it
turned up by accident.** Adding a genuine
disease-vs-control discriminative task moved the LLM's latents from *tying* the
frozen encoder to *beating* it — 0.6664 → 0.7034 against the encoder's 0.6680,
improving on 5 of 5 folds and surviving a dimensionality-matched control. That
is the first time in this project the LLM's representation has beaten the
encoder it sits on. **But the model cannot use what it now represents.**
Shuffling the transcriptomic vectors across samples leaves its forced-choice
answers essentially unchanged (AUC 0.4763 → 0.4732, input sensitivity 0.088),
so its output head is ~91% determined by the question text alone. The disease
signal is in the representation and the readout does not reach it — which is
direct evidence for **Hypothesis A**, the connector bottleneck, tested here only
by accident.

---

## 1. Primary metric — three-way probe on the clean 92-series holdout

Same folds, same seed, same holdout, same estimator across all conditions.
`d = LLM − encoder`; the encoder is identical in both runs (asserted, not
assumed).

| | encoder | LLM linear | **d** | LLM PCA-515 | d (matched) | LLM MLP |
|---|---|---|---|---|---|---|
| data-fix | 0.6680 | 0.6664 | **−0.0016** | 0.6625 | −0.0055 | 0.6823 |
| **discrim** | 0.6680 | **0.7034** | **+0.0353** | **0.6920** | **+0.0240** | 0.6700 |
| change | — | +0.0370 | **+0.0370** | +0.0295 | +0.0295 | −0.0123 |

Per-fold, `d` improved in **5 of 5 folds** (criterion required ≥4), and the LLM
beats the encoder outright in 4 of 5:

| fold | encoder | LLM data-fix | LLM discrim | d data-fix | d discrim |
|---|---|---|---|---|---|
| 0 | 0.7958 | 0.7562 | 0.7680 | −0.0397 | −0.0278 |
| 1 | 0.6031 | 0.6419 | 0.7170 | +0.0388 | +0.1139 |
| 2 | 0.6181 | 0.5988 | 0.6263 | −0.0193 | +0.0082 |
| 3 | 0.6894 | 0.7452 | 0.7555 | +0.0558 | +0.0661 |
| 4 | 0.6336 | 0.5899 | 0.6500 | −0.0438 | +0.0164 |

**Verdict: SUPPORTED.** +0.0370 clears the pre-set 0.03 threshold, 5/5 folds
clears the ≥4 requirement, and the PCA-515 control (+0.0295) shows it is not a
4096-vs-515 dimensionality artifact.

**The MLP probe went the other way** (0.6823 → 0.6700). Read together, the two
say something specific and worth keeping: discriminative training made the
disease signal more **linearly accessible** without making the representation
more **informative** overall. That is exactly what training a linear
discriminative objective should do, and it is a narrower claim than "the
representation got better".

## 1b. Batch-controlled comparison — the advantage largely disappears

Added 2026-09-02 after the original write-up, on a second pod. §1's numbers are
pooled across the holdout; because no non-holdout series carries both classes,
pooled AUC can reward batch signature. All 92 holdout series ARE mixed, so
computing AUC *inside* each series holds batch, platform, lab and largely tissue
fixed. This is the stricter test, and it does not support the §1 headline.

| features | pooled OOF | within (weighted) | within (unweighted) | median | series >0.5 |
|---|---|---|---|---|---|
| BulkFormer-93M | 0.6429 | 0.7645 | **0.7393** | **0.7987** | **68**/84 |
| LLM — data-fix | 0.6829 | 0.7717 | 0.7039 | 0.7500 | 66/84 |
| LLM — discrim* | 0.6802 | **0.7795** | 0.7090 | 0.7639 | 65/84 |

**Read this as a wash, not a win.** The weighted mean favours the LLM (+0.0150),
but the unweighted mean (−0.0303), the median (−0.0348) and the number of series
scoring above chance (65 vs 68) all favour the **encoder**. Weighted mean is
dominated by the largest series; the other three say the encoder is at least as
good per-series. **§1's +0.0353 does not survive holding batch fixed.**

Also note the two pooling conventions disagree materially on this population —
encoder 0.6429 pooled-OOF against 0.6680 mean-of-folds, data-fix LLM 0.6829
against 0.6664. Which convention is quoted changes the story, so it should
always be named.

**\* The discrim row is from a RECONSTRUCTED checkpoint and is indicative, not
measured.** Only the LoRA adapter, connector and configs were kept before the
training pod was deleted; the frozen base `language_model` and `vision_tower`
were rebuilt from source. Training ran `--bf16`, so the trainer saved
bf16-rounded base weights and the adapters were fitted against *those*, while
the rebuild writes higher-precision weights from HuggingFace — same
architecture, different weights. The rebuild scores **0.6617 ± 0.0836** on the
§1 metric where the original scored **0.7034 ± 0.0563** (fold 0: 0.501 vs
0.768), so it is demonstrably a different model. That check is the only reason
this is labelled rather than reported as fact. A faithful within-series number
requires a retrain with full checkpoint retention (~$5).

**Operational lesson:** a 629 MB adapter is not a reproducible checkpoint for a
run trained in bf16. Keep the trainer's own `language_model/` and
`vision_tower/`, or accept that the model cannot be rebuilt.

## 2. The finding that matters more — the readout ignores the input

The binary CVD evaluation returned pooled ROC-AUC **0.512** for a model
explicitly trained on that task, swinging 0.461–0.512 across prompt phrasings,
some of it below chance. Rather than report that as a model result, it was
tested: the eval author had flagged that `TinyLlavaScorer.score` had never
executed against a real model.

Scoring samples the model was **trained on**, with their own training questions:

| | AUC |
|---|---|
| real embeddings | 0.4763 |
| **shuffled embeddings** | **0.4732** |
| mean \|Δscore\| when input is swapped | 0.309 |
| std of scores across samples | 3.494 |
| **input sensitivity** (Δ/spread) | **0.088** |

Replacing a sample's transcriptomic profile with a different sample's changes
the answer by 9% of the score's own spread. **The forced-choice head is not
conditioning on the profile.** Training-set performance is not evidence of
generalization and is not used as such here — this is a functional test of the
readout path, and it fails.

Two consequences:

1. **The binary evaluation cannot be read as a model-quality measure.** Its
   0.512 measures the readout, not the representation. It is reported for
   completeness and excluded from the verdict.
2. **This is direct evidence for Hypothesis A.** The connector emits a single
   token; the probe shows disease information is present in the latents at that
   position and *more* linearly separable than in the encoder; the generation
   path does not use it. That is the bottleneck Hypothesis A describes, and this
   experiment tested it without meaning to.

## 3. Supporting evidence

**Multi-label probing — flat, including the controls.** Nothing moved beyond
fold noise, and critically neither did the technical controls, so the probe gain
is not batch signature leaking in:

| label | data-fix | discrim | kind |
|---|---|---|---|
| disease_category | 0.7885 | 0.7847 | scientific |
| cvd_subtype | 0.8494 | 0.8473 | scientific |
| tissue | 0.8536 | 0.8627 | scientific (2 folds scored only) |
| platform | 0.7071 | 0.7045 | **technical control** |
| instrument | 0.7054 | 0.7024 | **technical control** |

**Loss floors — not evidence, recorded for completeness.**
0.0724 (original) → 0.4681 (data-fix) → 0.4435 (discrim). The dip is mixture
arithmetic from adding an easy one-bit target and was named as non-evidence in
advance. It is smaller than a memorized target would give (~0.354), implying the
new category carried ~0.37 loss of its own.

**The encoder baseline was understated.** Within-series AUC on the 92 mixed
holdout series: encoder **0.772** against its pooled 0.658. The encoder is not
batch-shortcutting — its pooled figure is *deflated* by cross-series offsets.
Every prior comparison in this project used 0.668, so the bar was lower than it
should have been.

## 4. Limitations — including the ones pre-committed before the result

- **The corpus grew 32% and sample diversity 60%** (7,212 → 11,572 samples,
  since negatives had never appeared in any Stage-2 corpus). This experiment
  **cannot separate** "the discriminative task helped" from "more and more
  varied data helped". This caveat was written down before the numbers and it
  applies: the verdict is that the *condition* improved the representation, not
  that the discriminative objective specifically did.
- **One seed.** No run-to-run variance is available. A +0.037 shift with fold
  std ~0.06 is above the pre-set threshold but is one draw.
- **The series shortcut is irreducible in training data.** Zero non-holdout
  series contains both classes, so series identity alone scores AUC 0.9996. It
  was diluted (top series 20.89% → 0.573%, across 1,541 series) and *detected*
  at eval, never prevented. The probe is protected by grouped CV — validation
  series are unseen — and the difference between conditions cancels any effect
  common to both.
- **The tissue-only baseline for the binary eval was not computed** (no tissue
  labels were exposed to that harness). The tissue confound is therefore
  unmeasured on that metric, not ruled out. It *was* measured and neutralized in
  the training data (0.691 → 0.470 ungrouped after exact per-bucket matching).
- **Generative answer quality is still unevaluated**, carried over from the
  previous run.

## 5. What was verified rather than assumed

- Negative cache entries checked against a live encoder forward from the raw H5
  (max_abs_diff 4.8e-07, rolled control 3.11) — the population the cache builder
  could not verify.
- 8,720/8,720 discriminative answers polarity-verified against the label frame,
  by a checker whose negative controls (inversion, ambiguity, overclaim) caught
  four rule gaps and one false rejection during development.
- The previous 26,972 corpus items preserved **bit-identically**, so the
  comparison has one changed variable.
- Trainable parameters 321,929,216 — identical to both prior runs.
- Zero holdout leak, asserted independently at plan time, bundle time, and on
  the pod.
- Encoder folds asserted identical between the two probe runs before any
  difference was computed.
