# Hypothesis B — interpretation criteria, fixed BEFORE the results

Written 2026-09-02, after the retrain finished but **before any Phase 4 probe or
binary-eval number existed**. Committed at that point on purpose. Deciding what
counts as support after seeing the numbers is how a null result gets talked into
looking like a positive one, and this project has a standing commitment to
report the outcome honestly whichever way it falls.

Git history is the evidence that this file predates the results: the commit that
adds it contains no Phase 4 artifact.

---

## What is being tested

Does adding a genuine disease-vs-no-disease discriminative task to Stage 2
improve the **LLM's representation quality relative to the frozen encoder it
sits on**?

Not "does the model get good at the binary task" — it was trained on it, so
that is nearly circular. The claim under test is about the *representation*.

## The primary metric

**Three-way probe on the 92-series clean holdout, LLM latents vs BulkFormer-93M,
same grouped 5-fold, same PCA-matched dimensionality control.**

Prior conditions, for reference:

| condition | encoder | LLM linear | LLM MLP |
|---|---|---|---|
| data-fix | 0.6680 ± 0.0703 | 0.6664 ± 0.0711 | 0.6823 ± 0.0689 |

The standing finding is a **tie**: the LLM's latents did not beat the frozen
encoder they are built from.

## Decision rules

Let `d = LLM_linear − encoder_linear` on the clean holdout, with fold std ~0.07,
so the fold-to-fold noise floor on a 5-fold mean is roughly ±0.03.

1. **SUPPORTED** — `d` improves by **more than 0.03** over the data-fix
   condition's `d` (which was −0.0016), AND the direction is consistent across
   at least 4 of 5 folds. Anything smaller is inside noise.
2. **NOT SUPPORTED** — `d` is unchanged within ±0.03, or moves negative.
   This rules out Hypothesis B and points back to Hypothesis A, the
   single-token connector bottleneck. That is a real, useful result and gets
   reported as the headline, not buried.
3. **AMBIGUOUS** — `d` improves by more than 0.03 **but** the binary eval shows
   the series shortcut was taken (see below). Then the gain is not attributable
   to disease biology and must not be reported as support.

## The shortcut check that can invalidate a positive result

Zero non-holdout series contains both classes, so a model can score on the
discriminative task by recognizing batch signature. All 92 holdout series are
mixed, so:

- **pooled AUC high AND within-series AUC ≈ 0.5** → shortcut taken. Any
  representation gain is suspect and the verdict is AMBIGUOUS at best.
- **within-series AUC meaningfully above 0.5** → real per-sample signal.

The bar is the **frozen encoder's own within-series AUC: 0.765** (weighted;
0.739 unweighted), measured on real data before the LLM was trained. It is NOT
the 0.668 pooled figure this project has been quoting — the encoder's pooled
number is deflated by cross-series offsets, so the baseline was understated.

An LLM binary eval below 0.765 within-series does not beat the frozen encoder
at the task the model was explicitly trained on and the encoder was not.

## What is NOT evidence, and will not be cited as such

- **The loss floor.** Condition 3 adds an easy one-bit target, so its floor
  should fall below condition 2 from mixture arithmetic alone. Measured:
  0.4681 → 0.4435. This says nothing about representation quality.
- **Binary-eval accuracy on its own**, without the within-series breakdown.
- **Any metric computed over series containing a single class** — the standing
  batch-signature guard, asserted against rather than assumed away.
- **Multi-label probe movement inside its own fold noise.**

## Pre-committed caveats that apply to a positive result

- The corpus grew 32% (26,972 → 35,692) and sample diversity 60% (7,212 →
  11,572 samples). A representation gain could come from *more and more varied
  data* rather than from the discriminative task specifically. This experiment
  cannot separate those two, and any positive result must say so.
- Only one seed was run. No claim about run-to-run variance is available.
