# Step 2 Follow-Up — Materialization Gap and Interaction Query's Unstated N

Resolves the two items flagged at the end of Step 2, before Step 3 begins.

| | Outcome |
|---|---|
| Task 1 — 2,004-sample materialization gap | **Real data-quality exclusion. Assignments removed, not materialized.** |
| Task 2 — interaction_network_query's unstated N | **Fixed.** `{N}` added to all 10 templates, `n` now required in GT. |
| Task 2.4 — same bug class elsewhere | **1 further instance found** (`gene_driver_reasoning`) — flagged, not fixed. |
| Task 3 — re-validation | 4,000 assignments executed, **0 failures**; full structural pass on all 219,793. |

**Correction to the Step 2 report:** the gap was reported as 2,176 samples. That
double-counted — the true figure is **2,004 distinct patients**. The
`comparative_differential_reasoning` gap (172) and the `disease_subtype_classification`
gap (253, not previously reported at all) are both subsets of the
`gene_driver_reasoning` gap (2,004), not additions to it. 2,429 assignments were
affected across the three categories.

---

## Task 1 — Investigation: the exclusion is real, not incidental

### Why the 2,004 were excluded

They were never dropped by `build_bulkformer_matrix.py`. That script materializes
whatever is in `cvd_only_sample_index.npy`, and the filtering happened upstream in
`materialize_cvd_matrix.py`, whose manifest records two sequential filters applied to
the 10,557 disease-confirmed samples:

| Filter | Removed | Reason |
|---|---|---|
| `singlecellprobability >= 0.5` | **1,832** | Single-cell samples. BulkFormer is a **bulk** transcriptome model — wrong data modality. |
| library size < 100,000 | **172** | Degenerate sequencing depth. |
| | **2,004** | = the entire gap |

Both were verified independently rather than taken from the manifest:

- All 1,832 carry `is_bulk == False` in `probe_sample_labels.parquet`, consistent with
  the ARCHS4 single-cell probability threshold.
- The 172 bulk samples' library sizes were recomputed directly from the H5:
  **median 4,579, min 2, max 96,374**, with 66 below 1,000 total counts — against a
  median of **1,920,351** for retained samples, a ~400× difference. Every one is below
  the 100,000 threshold. TPM normalization divides by a per-sample total; at a library
  size of 2 the resulting vector is noise, not measurement.

**Conclusion: a real data-quality reason, in both cases.** Per the guardrail,
materialization was **not** extended. `build_bulkformer_matrix.py` and
`bulkformer_input/` are unchanged — the matrix remains 8,553 × 20,010.

Forcing these in would mean feeding a bulk-transcriptome model single-cell profiles
and near-empty libraries, then training a captioning objective on the resulting
numbers as though they were real measurements.

### Resolution and impact

The 2,004 patients are now excluded at the source — `eligible_populations()`
intersects every category with the materialized set, so the plan cannot reference a
patient that has no image.

| Category | Eligible by gate | No materialized input | Retained |
|---|---|---|---|
| Stage 1 (all five) | 8,553 | 0 | **8,553** |
| `disease_subtype_classification` | 2,942 | 253 | **2,689** |
| `comparative_differential_reasoning` | 8,725 | 172 | **8,553** |
| `gene_driver_reasoning` | 10,557 | 2,004 | **8,553** |

**Impact on the two thinnest categories, quantified:**

| Category | Before | After | Change |
|---|---|---|---|
| `gene_driver_reasoning` | 10,557 | 8,553 | **−2,004 (−19.0%)** |
| `comparative_differential_reasoning` | 8,725 | 8,553 | −172 (−2.0%) |
| `disease_subtype_classification` | 2,942 | 2,689 → **2,687** | −255 (−8.7%) |
| **Stage 2 total** | **22,224** | **19,793** | **−2,431 (−10.9%)** |

Stage 2 therefore falls from 44.4% to **39.6%** of its 50,000 target. Stage 1 is unaffected
(200,000, unchanged) because it always drew on the materialized 8,553.

`gene_driver_reasoning` takes the whole loss — it was the only category reaching beyond
the bulk-QC'd population, and 19% of its reach was single-cell samples. Note this costs
**no distinct facts**: that category returns the identical 1,142-gene list for every
patient, so the loss is 2,004 repetitions of an answer that remains in the corpus 8,553
times.

### Side effect: the subtype cap now engages

Restricting to materialized samples shifted the subtype distribution enough to cross the
35% ceiling — in Step 2 it was inert.

| Subtype | Before | Share | After | Share |
|---|---|---|---|---|
| coronary_artery_disease | 943 | 35.07% | **941** | 35.00% |
| heart_failure | 804 | 29.90% | 804 | 29.92% |
| hypertension | 679 | 25.25% | 679 | 25.27% |
| cardiomyopathy_other | 141 | 5.24% | 141 | 5.25% |
| arrhythmia_afib | 122 | 4.54% | 122 | 4.54% |
| **total** | **2,689** | | **2,687** | |

CAD exceeded the 941-sample ceiling by 2, and those 2 are dropped.

**The freed budget is not redistributed, and the report does not claim it was.** Step 2's
cap function computed a proportional redistribution; that is not achievable here. Each
patient contributes exactly one item to this category, and every other subtype already
contributes all of its patients — there is no spare capacity to move the surplus into.
Redistributing would mean asking the same patient the same question twice, which
`DIVERSITY_RULE` forbids. The cap function now applies the cap to the actual assignments
and reports `freed_budget_redistributed: 0` with the reason, rather than reporting a
redistribution the plan never performed.

---

## Task 2 — Interaction Network Query's unstated N

### The defect

All 10 templates asked for "the genes most co-expressed with `{gene}`" while the plan
silently bound `n = 10`. The stored edge list holds 100 partners per gene, so the
question identified no count and the answer asserted one it never requested.

### The fix, in three places

**1. Templates** — `{N}` added to all 10, following the pattern
`ranking_ordering_query` already uses for the same reason. The YAML records why, next to
the templates, matching this project's convention of keeping constraints with the data:

> `"What is the expression level of the top {N} genes most co-expressed with {gene} in this sample?"`

(The file is `qa_generation/templates/stage1.yaml`; the task referred to it as
`stage1_templates.yaml`.)

**2. GT function** — `interaction_network_query(sample_id, gene, n)`. `n` has no default
and is validated: `None`, non-integer, `< 1`, or `> 100` (beyond stored edge depth) all
return `unusable_partner_count:<value>`. It is refused, never guessed — the same
contract `ranking_query` has for its size bound.

**3. Plan** — every one of the 40,000 assignments binds a real `N`, varied for diversity
per `DIVERSITY_RULE`, with 10 kept as the most common value:

| N | Instances |
|---|---|
| 5 | 8,073 |
| 10 | 15,945 |
| 15 | 8,034 |
| 20 | 7,948 |

Verified: all 40,000 have an integer `n`, every chosen template contains `{N}`, and the
returned partner count equals the bound `N` in every executed sample.

Test coverage added: `test_interaction_requires_an_explicitly_bound_valid_n`
(7 invalid inputs) and `test_interaction_n_is_mandatory_in_the_signature`.
Suite is **88 passing**, up from 80.

### Task 2.4 — One further instance, flagged not fixed

Every category was checked by comparing its template placeholders against its GT
function's answer-shaping parameters:

| Category | Placeholders | GT parameters | Verdict |
|---|---|---|---|
| `direct_abundance_query` | `gene` | `gene` | ok |
| `threshold_query` | `threshold` | `threshold`, `direction` | ok — direction is in the wording, compatibility-enforced |
| `ranking_ordering_query` | `N`, `percentile` | `n_or_percentile`, `direction` | ok — same |
| `comparative_query` | `gene_A`, `gene_B` | `gene_a`, `gene_b` | ok |
| `interaction_network_query` | `gene`, **`N`** | `gene`, `n` | **fixed this pass** |
| `disease_subtype_classification` | — | — | ok |
| `comparative_differential_reasoning` | `condition`, `comparison_group` | `comparison_group` | ok — `condition` bound from the sample's own label |
| `gene_driver_reasoning` | **none** | **`top_n`** | ⚠️ **see below** |

**`gene_driver_reasoning` has the same class of defect.** It accepts `top_n`, the plan
leaves it unbound, and the answer is therefore all **1,142 genes** — while its templates
use bounded superlative phrasing:

> "Determine the **top** molecular signals contributing to cardiovascular disease classification."
> "Identify the **highest-ranking** signal genes distinguishing this cardiovascular sample."

It is milder than the interaction bug — the answer asserts no count the question
contradicts — but a 1,142-gene answer to "the top molecular signals" is not a usable
training target, and the question gives no bound to justify any truncation.

**Not fixed, per the guardrail.** The remedy is the same shape as this pass
(add `{N}`, require `top_n`, bind per instance) but it touches a Stage 2 category and
would change every one of its 8,553 answers. Your call whether it belongs in this pass.

---

## Task 3 — Re-validation

Structural checks across **all 219,793** assignments, plus 4,000 executed against the
real GT functions (2.5× the original 1,600 scale, 500 per category):

| Check | Result |
|---|---|
| GT returns `ok` | **4,000 / 4,000** |
| Degenerate or `boundary_tie` answers | **0** |
| Duplicate `(patient, category, entity)` triples | **0** |
| Template/parameter mismatches | **0** |
| Assignments referencing a patient with no materialized input | **0** |
| Chosen template has an unbound placeholder | **0** |
| `interaction_network_query`: returned partner count == bound `N` | **all** |
| `gt_functions` test suite | **88 passed** |

Neither fix introduced a new mismatch.

---

## Final totals

| Stage | Target | Step 2 | Now | % of target |
|---|---|---|---|---|
| Stage 1 | 200,000 | 200,000 | **200,000** | 100% |
| Stage 2 | 50,000 | 22,224 | **19,793** | **39.6%** |
| Total | 250,000 | 222,224 | **219,793** | 87.9% |

Stage 2's shortfall is now 30,207; hitting the target would take 2.53 phrasings per
patient. The Step 2 recommendation stands: raise it with new *facts* — the deferred
per-subtype elastic-net rerun, or a differential-expression run against `neg_hard` —
rather than new phrasings.

## Files changed

| File | Change |
|---|---|
| `qa_generation/templates/stage1.yaml` | `{N}` added to all 10 interaction templates, with rationale |
| `qa_generation/gt_functions.py` | `interaction_network_query`'s `n` now required and range-validated |
| `qa_generation/build_generation_plan.py` | Eligibility intersected with materialized input; `N` bound per instance; subtype cap now applied to assignments, not just reported |
| `qa_generation/tests/test_gt_functions.py` | 80 → 88 tests |
| `qa_generation/generation_plan.json` | 222,224 → 219,793 assignments |
| `qa_generation/sampling_plan_report.md` | Updated for the corrected figures |
| `qa_generation/bulkformer_input/` | **Unchanged** — materialization deliberately not extended |

Step 3 (template filling) has not been started.
