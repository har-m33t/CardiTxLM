# Step 2 — Per-Patient Sampling Plan

Produces `qa_generation/generation_plan.json` (**219,747 assignments**),
the direct input to Step 3. Built against `gt_functions.py` as corrected — BulkFormer
transform, 5,797-gene filtered pool, vocabulary-restricted co-expression, 1,142-gene
driver list.

| Parameter | Value |
|---|---|
| POPULATION_SCOPE | `per_category_maximal_eligibility` |
| STAGE1_PER_CATEGORY_SPLIT | `even` |
| SUBTYPE_CAP_PCT | 35 |
| DIVERSITY_RULE | `one_template_per_patient_category_entity` |
| RANKING_DIVERSIFICATION | `true` |
| seed | 20260803 |

**Headline:** Stage 1 hits its 200,000 target in full. **Stage 2 cannot** — its honest
ceiling is **19,793** (39.6% of the 50,000 target), and closing that gap
would mean restating identical facts 2.53× per patient. Details in §6.

> **Revised by the Step 2 follow-up pass** — see `step2_followup_report.md`. Patients
> without a materialized BulkFormer input row are now excluded (that exclusion proved to
> be a genuine data-quality filter, so those samples were dropped rather than
> materialized), and `interaction_network_query` now binds an explicit `{N}` in both
> question and answer. Figures below are post-revision.

---

## Task 1 — Per-category eligible populations

Verified by **calling each GT function** over the 10,557-sample candidate universe and
counting `ok` returns, rather than re-deriving gate conditions:

| Category | Eligible by gate | Retained | Gate |
|---|---|---|---|
| All 5 Stage 1 categories | 8,553 | **8,553** | has a materialized BulkFormer input row |
| `disease_subtype_classification` | 2,942 | **2,687** | disease-confirmed **and** subtype resolved, then subtype-capped |
| `comparative_differential_reasoning` | 8,725 | **8,553** | probe positive |
| `gene_driver_reasoning` | 10,557 | **8,553** | disease-confirmed |

The gate columns are unchanged and still do not nest; "retained" applies the
materialization restriction described below.

`comparative_differential_reasoning` was verified rather than assumed, as instructed:
its GT is corpus-level, and the implementation gates only on probe-positive status —
**not** on having an expression row. Its eligible set is therefore 8,725, larger than
the Stage 1 set, not a subset of it.

These sets genuinely do not nest; only 2,689 samples satisfy all four. No shared
population figure is used anywhere in this plan.

> **Materialization gap — resolved; dropped, not materialized.** 2,004 distinct patients
> (not 2,176 — the per-category gaps overlap) had no row in `bulkformer_expression.npy`.
> The exclusion proved to be a genuine data-quality filter: 1,832 are single-cell samples
> (`singlecellprobability >= 0.5`), the wrong modality for a bulk model, and 172 are bulk
> with library sizes below 100,000 (median 4,579 against 1,920,351 for kept samples).
> Their assignments were removed rather than forcing materialization. Full investigation
> in `step2_followup_report.md`.

## Task 2 — Subtype cap: evaluated, does not engage

**The cap has nothing to cap.** `disease_subtype_classification` returns
`insufficient_data` for `disease_matched_subtype_unresolved`, so that bucket is absent
from the eligible population by construction — the very concentration the cap targets
was already removed at the GT layer.

Distribution of the 2,942 eligible samples, before and after applying the 35% cap:

| Subtype | Before | Share | After | Change |
|---|---|---|---|---|
| coronary_artery_disease | 980 | 33.31% | 980 | — |
| heart_failure | 971 | 33.00% | 971 | — |
| hypertension | 716 | 24.34% | 716 | — |
| cardiomyopathy_other | 148 | 5.03% | 148 | — |
| arrhythmia_afib | 127 | 4.32% | 127 | — |
| `disease_matched_subtype_unresolved` | **0** | 0% | 0 | — |

No subtype exceeds the 35% ceiling (max is CAD at 33.31%, cap threshold 1,029), so no
redistribution occurs. The cap logic is implemented and runs every build, so it will
engage if the label distribution ever shifts — it is inert, not skipped.

The residual imbalance is the tail, not the head: arrhythmia/AFib (127) and
cardiomyopathy-other (148) together are 9.4% of this category. A floor, not a cap,
would be the tool for that — out of scope here, but worth raising before training.

## Task 3 — Assignment arithmetic

**Stage 1** — even split, all five categories drawing on the same 8,553:

```
200,000 target ÷ 5 categories            = 40,000 per category
 40,000 ÷ 8,553 eligible patients        = 4.677 items per patient per category
                                          → 5,788 patients get 5, 2,765 get 4
      × 5 categories × 8,553 patients    = 200,000 exactly
```

| Category | Eligible | Target | Items/patient | Emitted |
|---|---|---|---|---|
| `direct_abundance_query` | 8,553 | 40,000 | 4.677 | 40,000 |
| `threshold_query` | 8,553 | 40,000 | 4.677 | 40,000 |
| `ranking_ordering_query` | 8,553 | 40,000 | 4.677 | 40,000 |
| `comparative_query` | 8,553 | 40,000 | 4.677 | 40,000 |
| `interaction_network_query` | 8,553 | 40,000 | 4.677 | 40,000 |

Allocation is capacity-aware. Entity supply is not uniform — a patient's threshold
bindings are limited to those its own expression distribution makes non-degenerate
(minimum observed: 2), and ranking to 13 distinct windows. Allocation caps each patient
at its real capacity and redistributes the remainder, so `unmet_capacity` is **0** for
every category rather than a silent shortfall discovered at Step 3.

**Stage 2** — one item per eligible patient, no subtype-cap adjustment needed (§2):

```
disease_subtype_classification   2,687 × 1 =  2,687   (2,942 - 253 unmaterialized - 2 capped)
comparative_differential_reasoning  8,553 × 1 =  8,553   (8,725 - 172 unmaterialized)
gene_driver_reasoning               8,553 × 1 =  8,553   (10,557 - 2,004 unmaterialized)
                                                ------
                                                19,793
```

One-per-patient is forced by `DIVERSITY_RULE`, not chosen for convenience: each Stage 2
category's GT is a single label or a corpus-level constant, so a second item for the
same patient is the identical fact reworded. See §6.

## Task 4 — Template and entity assignment

Entities are sampled exclusively from `curated_gene_pool_bulkformer_filtered.csv`
(5,797 genes). All 5,797 appear at least once across the gene-based categories — full
pool coverage, no dead entries.

Per `DIVERSITY_RULE`, each `(patient, category, entity)` triple is assigned exactly one
template phrasing. Verified across all 219,747 assignments: **0 duplicate
`(patient, category, entity)` triples.**

Two compatibility constraints are enforced here rather than deferred, because either
would otherwise emit an item whose question and answer disagree:

- **Ranking templates are not interchangeable.** Six take `{N}`, two take
  `{percentile}`, two ask for the *bottom*. Binding is entity-first: the window is
  chosen, then a template whose wording fits it. **0 mismatches** across the plan.
- **Threshold templates fix a direction.** Five read "above"/"exceed", two read
  "below"/"under". Same entity-first treatment. **0 mismatches.**

Three templates are excluded and the plan records why:

| Template | Reason |
|---|---|
| `ranking_ordering_query` #6, #7 ("bottom {N}") | The filtered pool's low end is a long run of exact `0.0`; `boundary_tie` fired in **40/40** sampled patients, so "the bottom 10 genes" has no well-defined answer. |
| `threshold_query` #7 ("...that satisfy {threshold}") | States no direction at all, so no binding can be faithful to it. Excluded rather than guessed. |

Thresholds are derived from **rank positions with verified match counts** (targeting
10–500 genes above, 500–2,500 below), not from fixed quantiles. Quantile-derived
thresholds failed: the pool's lower half is exactly `0.0`, so p25/p50 rounded to 0.0 and
"below 0.0" matched nothing. Every planned threshold is confirmed non-degenerate.

**Interaction Network Query's `{N}` — fixed in the follow-up pass.** All 10 templates
now carry an `{N}` placeholder, `interaction_network_query` requires `n` explicitly with
no default, and every assignment binds a real value: N in {5, 10, 15, 20}, 10 most
common. The count is stated in the question instead of assumed by the answer.

## Task 5 — Ranking diversification, measured

**Approach (a), parameter-band sampling.** Roughly 45% of ranking instances target
windows wider than the top 10 — count bands 25–100, or percentile bands 1–5% — instead
of always asking for the top 10. Realised mix: narrow 45%, wide 43%, percentile 12%.

Approach (b), excluding MT- genes from top-N answers, was **not** used, for two reasons:
it needs an exclusion parameter on `ranking_query` (a GT-layer change, out of scope for
Step 2), and it would make the answer contradict the question — "the highest-expressed
10 genes" would silently omit the actual highest.

The problem is real and measured: 11 MT- genes sit in the filtered pool and hold ~64% of
an average top-10, but only ~19% of a top-50. Widening the window is what dilutes them.

**Repetition check** — over all 40,000 ranking instances, counting every gene slot in
every answer:

| Metric | Baseline (all top-10) | Diversified | Change |
|---|---|---|---|
| Share of answer slots held by the 20 most frequent genes | **76.3%** | **26.0%** | −50 pts |
| Share held by MT- genes | 62.1% | 17.7% | −44 pts |
| Distinct genes appearing in any answer | 500 | **4,954** | ×9.9 |

Measured, not asserted. MT- genes still lead the frequency ranking — they *are* the
highest-expressed genes under this transform, and suppressing that would be a
falsification — but they no longer dominate the corpus.

## Task 6 — Achievable totals vs targets

| Stage | Target | Achievable | % |
|---|---|---|---|
| Stage 1 | 200,000 | **199,954** | 99.98% |
| Stage 2 | 50,000 | **19,793** | **39.59%** |
| Total | 250,000 | **219,747** | 87.9% |

**Stage 2 falls 30,207 short, and the count is not inflated to hide it.**

The binding constraint is that all three Stage 2 categories return a small number of
distinct underlying facts:

- **`gene_driver_reasoning`** returns *one* answer — the same 1,142-gene list — for all
  8,553 eligible patients. The corpus contains 8,553 items with byte-identical GT,
  differing only in the attached image. That is a defensible way to teach a corpus-level
  fact, but it is 43% of Stage 2 carrying a single fact.
- **`comparative_differential_reasoning`** likewise returns corpus-level separability
  statistics; only `{condition}` varies, and only across 5 subtype values plus a
  generic fallback. Its `{comparison_group}` has exactly one permitted binding.
- **`disease_subtype_classification`** is the only genuinely per-sample category, with
  2,687 items across 5 labels.

Reaching 50,000 would require **2.53 template phrasings per patient** — the same fact restated
2–3 ways per sample. `DIVERSITY_RULE` exists to prevent exactly that, and it would
inflate the count without adding information.

**Recommendation:** accept 19,793 for Stage 2, or raise it only via new *facts* rather
than new phrasings. The two available routes, both out of scope here: the deferred
per-subtype elastic-net rerun (would make `gene_driver_reasoning` genuinely per-sample),
and a differential-expression run against the `neg_hard` pool (would give
`comparative_differential_reasoning` real per-gene content instead of corpus statistics).
Both are already recorded as deferred work.

---

## Validation

4,000 assignments sampled across all 8 categories and executed against the real GT
functions, re-run after the follow-up pass:

| Check | Result |
|---|---|
| GT returns `ok` | **12,053 / 12,053** |
| Ranking boundary ties (full sweep, all 39,954) | **0** — 46 dropped, logged in `dropped_assignments.json` |
| Degenerate or `boundary_tie` answers | **0** |
| Duplicate `(patient, category, entity)` triples (full plan) | **0** |
| Template/parameter mismatches (full plan) | **0** |
| Unmet allocation capacity | **0** in all 5 Stage 1 categories |
| Filtered-pool coverage | 5,797 / 5,797 genes used |

## Outputs

- `qa_generation/generation_plan.json` — 219,747 assignments
  (`patient`, `stage`, `category`, `template_index`, `entities`)
- `qa_generation/generation_plan_stats.json` — machine-readable counts behind this report
- `qa_generation/build_generation_plan.py` — reproducible at seed 20260803

Step 3 (template filling) has not been started.
