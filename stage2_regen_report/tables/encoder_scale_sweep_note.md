# Encoder scale sweep: is BulkFormer-93M at the plateau?

**Yes.** Across the existing 5-variant sweep (`linear_probe/results/`, 5-fold
series-grouped CV, 8,725 CVD positives), 93M is the *best* variant on both
negative pools, and everything above it is flat-to-worse — so there is no
accuracy case for retraining at 147M.

- **neg_whole_corpus ROC-AUC:** 37M 0.925±0.037, 50M 0.928±0.036,
  **93M 0.944±0.023**, 127M 0.943±0.027, 147M 0.941±0.024. The 93M→147M spread
  is 0.0034, roughly 1/3 of a single fold-mean standard error (0.023/√5 ≈ 0.010),
  i.e. the top three variants are statistically indistinguishable. The curve
  rises from 37M to 93M and is flat thereafter — a textbook plateau at 93M.
- **neg_hard ROC-AUC:** 37M 0.781±0.105, 50M 0.724±0.110, **93M 0.801±0.187**,
  127M 0.720±0.215, 147M 0.773±0.180. 93M is again nominally highest, but the
  fold-to-fold variance on this pool is so large (one 93M fold scores 0.44)
  that no variant ordering here is meaningful. Treat neg_hard as non-informative
  for the scale question rather than as support for 93M.
- Parameter count buys nothing after 93M despite a 1.75× jump in trainable
  parameters (75.9M → 132.1M) and a wider embedding (515 → 643 dims).

**Conclusion (stated explicitly per Phase 1b, and deliberately not acted on):**
BulkFormer-93M sits at or past the knee of the scale curve on this task. A 147M
retrain is not justified by the sweep, and the 93M lock is retained.
