# Hypothesis B — Phase 1 Data Plan

Built from `hypothesis_b_prelim_report.md` / `scripts/hypothesis_b/phase0_stats.json`.
Executed by `scripts/hypothesis_b/build_discriminative_plan.py`; every number
below is reproduced by that script into `discriminative_plan_stats.json`.

---

## 1. Target ratio — **1:1**, exact, per tissue bucket

Not the raw 1:2.85 available, and not a "realistic base rate".

- **Against 1:2.85** — that ratio is a curation artifact (how many samples
  happen to carry `tissue_only_disease_unconfirmed`), not a disease prevalence.
  It is not more realistic, only differently arbitrary.
- **Against a true base rate (~5-10%)** — it would make "always answer no"
  near-optimal, reintroducing the degenerate-answer failure the previous
  regeneration existed to remove.
- **For 1:1** — the holdout evaluation prior is 1,341:1,266 ≈ 1:0.94, and
  Phase 4's forced-choice log-probability extraction is prior-sensitive. A
  training prior far from the evaluation prior moves the operating point for
  reasons unrelated to representation quality, which is the thing under test.

## 2. Negative selection — and positive selection

Phase 0 recommended capping negatives only, which yields 6,098/class. **This
plan caps both classes**, at 50 samples per GEO series, and takes an exact
per-bucket match. That costs 1,738 pairs and buys the following:

| | Phase 0 proposal | This plan |
|---|---|---|
| Per class | 6,098 | **4,360** |
| Cap applied to | negatives only | **both classes** |
| Max single-series share | 20.89% (uncapped positives) | **0.573%** |
| Distinct series | — | **1,541** (383 pos / 1,158 neg) |
| Share of final corpus | 31.1% | **24.4%** |

Capping only negatives leaves the identical shortcut sitting on the positive
side — the positive pool's top series is 15.85% and its effective series count
is 28.4 against a nominal 388. One decision fixes both, and incidentally brings
the new category's corpus share down to a level that cannot dominate
supervision (see §4).

Within each bucket, sampling is **round-robin across series**, not a flat random
draw: a flat draw reproduces the pool's own concentration, which is the thing
being diluted. Seed 20260901, recorded.

## 3. The confound this is designed against — measured, not argued

**Tissue.** A classifier given only the normalized `source_name_ch1` string
scores **ROC-AUC 0.691 ± 0.168** against **0.800** for the full 515-d embedding
on the same folds — a tissue string reproduces ~64% of the above-chance signal
and beats the transcriptome outright on fold 0. Exact per-bucket matching makes
the per-bucket positive rate identical everywhere, so tissue carries *exactly*
zero information about the label by construction. Verified empirically:

| | grouped 5-fold | ungrouped 5-fold |
|---|---|---|
| before matching | 0.691 | — |
| after matching | 0.361 | **0.470** |

The ungrouped number is the one that answers the question, and 0.470 ≈ chance
says the shortcut is gone. The grouped number falls *below* 0.5 without any
residual confound: whole-series folds plus single-class series make per-bucket
rates invert between train and validation. Both are reported so neither can be
quoted as the other.

**Series — not fixable, and this plan does not claim to fix it.**
Zero non-holdout series contain both classes (388 positive-only, 1,174
negative-only, no overlap). This is structural: the holdout was *defined* as
every mixed-class series. Series identity alone scores **AUC 0.9996** on the
selection. No sampling scheme can remove this; the cap makes the shortcut
expensive (1,541 series to memorize instead of a handful) rather than free.

**It is therefore detected, not prevented.** Phase 4 computes **within-series
AUC** on the 92 holdout series, all of which are mixed by construction.
Comparing samples inside one series holds batch, platform, lab and tissue
essentially fixed, so only per-sample biology separates them. A model that took
the shortcut shows high pooled AUC and ~0.5 within-series AUC. **No headline
from this category should be believed without that number beside it.**

## 4. Corpus integration — append, at 24.4%

| Category | Items | Before | After |
|---|---|---|---|
| comparative_differential_reasoning | 7,208 | 26.7% | 20.2% |
| gene_driver_reasoning | 7,210 | 26.7% | 20.2% |
| magnitude_reasoning | 7,202 | 26.7% | 20.2% |
| disease_subtype_classification | 5,352 | 19.8% | 15.0% |
| **cvd_presence_discrimination** (new) | **8,720** | — | **24.4%** |
| **Total** | **35,692** | | |

The existing four are left **bit-identical**, so the Phase 3 loss curve and the
Phase 4 probes compare against the previous run without a second confounded
variable. One item per sample, the minimum, so the category's weight comes
entirely from breadth of samples rather than repetition.

The new category also introduces **4,360 samples the model has never seen** —
negatives were absent from every prior Stage-2 corpus. Sample diversity rises
from 7,212 to 11,572 (+60%), which is a larger change than the item count
suggests.

**A deliberate exception to the standing degeneracy gate.** This category's
answer is one bit, so its distinct-answer ratio is ~2/N — numerically identical
to the degeneracy the last regeneration removed. It is not the same defect: the
old failure was a *constant* string independent of the sample, whereas this
answer is a verified per-sample fact. But the `MIN_DISTINCT_RATIO = 0.90` gate
in `build_stage2_bundle.py` must not be applied to it, and is not. It is gated
instead on **label balance** and **per-sample label correctness against
`probe_sample_labels.parquet`**, which are the properties that actually matter
for a binary target.

## 5. Leakage check — before generation, not after

Asserted inside `build_discriminative_plan.py`, and the run fails rather than
warns:

- `no_holdout_series` — zero selected samples in any of the 92 holdout series ✅
- `no_duplicate_samples` ✅
- `all_have_encoded_vector` — every selection has its 515-d cache entry ✅
- `series_cap_respected` ✅
- `tissue_uninformative` — per-bucket positive rate constant ✅

Re-asserted independently at bundle time in Phase 2 against
`holdout_series.json`, never inherited from this phase.
