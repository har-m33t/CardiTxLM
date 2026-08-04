# Step 2 Final Cleanup — gene_driver_reasoning's Unbound `top_n`

Closes the last known question/answer-mismatch defect flagged in
`step2_followup_report.md`: `gene_driver_reasoning` accepted `top_n` but the plan left
it unbound, so all 8,553 assignments returned the full **1,142-gene** stable list —
while every template in that category asks for "the top molecular signals" or "the
highest-ranking signal genes".

| | Outcome |
|---|---|
| Task 1 — bind `top_n` in the plan | **Done.** All 8,553 assignments carry an explicit value. |
| Task 2 — remove the silent default | **Done.** `top_n` is required; `None`/invalid rejected as `unusable_top_n`. |
| Task 3 — template compatibility check | **Confirmed: no rewording needed.** No template implies "all". |
| Task 4 — test coverage | 88 → **100 tests**, all passing. |
| Task 5 — re-validation | Full structural pass on all 219,793; **all 8,553** driver assignments executed. |

Totals are unchanged: Stage 1 200,000, Stage 2 19,793, total **219,793**. This pass changed
what those assignments *ask for*, not how many there are.

---

## Task 3 — Template compatibility: checked, not assumed

Your assessment is correct — **no template needs rewording.** All 10 were read
individually and pattern-checked for unbounded phrasing (`all`, `every`, `complete`,
`entire`, `full list`, `list all`):

| # | Template | Bounded phrasing | Implies "all"? |
|---|---|---|---|
| 0 | "Identify the genes **most indicative** of cardiovascular disease…" | superlative | no |
| 1 | "Which genes are **most associated** with distinguishing this cardiovascular sample…" | superlative | no |
| 2 | "Determine the **top** molecular signals contributing to…" | explicit "top" | no |
| 3 | "…identify the **most informative** genes for cardiovascular disease." | superlative | no |
| 4 | "Report the genes that contribute **most strongly** to…" | superlative | no |
| 5 | "Analyze the **key** genes driving the cardiovascular transcriptomic signature." | superlative | no |
| 6 | "Which genes have the **strongest** association with cardiovascular disease status?" | superlative | no |
| 7 | "Identify the **highest-ranking** signal genes distinguishing…" | explicit rank | no |
| 8 | "Determine the genes **most responsible** for the observed…" | superlative | no |
| 9 | "What are the **primary** gene-level features supporting…" | superlative | no |

**0 of 10 imply a complete or unbounded list**, and none states a specific count. Any
bounded list therefore satisfies the question as written, so — unlike
`interaction_network_query`, whose fix required adding `{N}` — this is a GT-binding
fix only. `stage2.yaml` is **unmodified**.

### One consequence that shaped the binding

Because the phrasing carries no count, `top_n` cannot vary *within* a phrasing. Each of
the 10 templates is used ~855 times; if "Determine the top molecular signals…" returned
10 genes in some instances and 20 in others, identical question text would carry
different answers — which is the same defect this pass exists to remove, reintroduced in
a new form.

So the bound is **keyed to the template index** rather than drawn at random:

```
GENE_DRIVER_TOP_N = {i: (10, 15, 20)[i % 3] for i in range(10)}
```

This still varies across the corpus per `DIVERSITY_RULE` — three distinct answer sizes,
assigned across different patients — while keeping every individual phrasing internally
consistent. Verified: **0 templates map to more than one `top_n`**.

## Task 1 — `top_n` distribution

All **8,553 / 8,553** assignments carry an explicit integer `top_n`. None is unbound or
defaulted.

| `top_n` | Assignments | Share | Templates |
|---|---|---|---|
| 10 | 3,481 | 40.7% | 0, 3, 6, 9 |
| 15 | 2,483 | 29.0% | 1, 4, 7 |
| 20 | 2,589 | 30.3% | 2, 5, 8 |

The 40/30/30 split follows from four templates mapping to 10 and three each to 15 and 20,
with templates drawn uniformly per patient.

## Task 2 — Explicit rejection, mirroring `interaction_network_query`

`gene_driver_reasoning(sample_id, top_n)` — no default. Rejected with
`unusable_top_n:<value>`:

| Input | Rejected |
|---|---|
| `None` (previously meant "all 1,142") | ✓ |
| `0`, negative | ✓ |
| `> 1142` (beyond the stable list) | ✓ |
| non-integer (`"ten"`, `10.5`) | ✓ |
| `True` (bool is not a count) | ✓ |

Exactly the contract `interaction_network_query`'s `n` received in the previous pass.

## Task 4 — Test coverage

88 → **100 tests**, all passing. Added, mirroring the interaction pattern:

- `test_gene_driver_requires_an_explicitly_bound_valid_top_n` — 7 invalid inputs
- `test_gene_driver_top_n_is_mandatory_in_the_signature` — no default in the signature
- `test_gene_driver_returns_exactly_the_requested_count` — for 10, 15, 20
- `test_gene_driver_counts_are_nested_prefixes` — top-20 extends top-10 rather than
  reordering it, so the three answer sizes stay mutually consistent

## Task 5 — Re-validation at full scale

**Structural check across all 219,793 assignments** (not a sample):

| Check | Result |
|---|---|
| Duplicate `(patient, category, entity)` triples | **0** |
| Template/parameter mismatches | **0** |
| Unbound placeholders / unbound `top_n` | **0** |
| Assignments referencing an unmaterialized patient | **0** |
| `gene_driver` assignments carrying an explicit `top_n` | **8,553 / 8,553** |
| Templates mapping to more than one `top_n` | **0** |

**Execution:**

| Set | Executed | Result |
|---|---|---|
| `gene_driver_reasoning` — **every** assignment | **8,553** | 0 failures; returned gene count == bound `top_n` in all 8,553 |
| All other categories — 500 each | 3,500 | 0 failures |
| `gt_functions` suite | 100 tests | all passing |

Nothing else was disturbed: totals, eligibility, the subtype cap, and every other
category's bindings are byte-identical to the previous pass.

---

## Flagged, not fixed: residual ranking boundary ties

Executing at full scale surfaced something sampling had missed, in a category outside
this task's scope.

**46 of 40,000 `ranking_ordering_query` assignments (0.115%) have `boundary_tie: True`** —
the expression value at rank N equals the value at rank N+1, so "the top N genes" has no
uniquely defined membership at the cut.

| Spec | Ties |
|---|---|
| 100 | 13 |
| 75 | 8 |
| 50 | 6 |
| 1.0% / 2.0% / 5.0% | 5 each |
| 40 | 4 |
| (remainder) | 5 |

This is **pre-existing and not caused by this pass** — no ranking assignment was touched.
It is also a correction to my own earlier reporting: the Step 2 and follow-up passes both
stated "0 degenerate answers", but that came from 500-item samples. At a 0.115% rate,
a 500-item sample misses it roughly 56% of the time. The full-set figure is 46, not 0.
Thresholds are genuinely clean: **0 / 40,000** degenerate.

It is a different defect class from this pass's (an *ambiguous* answer, not a
*mismatched* one), it affects a category this task was scoped out of, and the fix was a
real choice — drop the 46, or nudge each to the nearest untied cut.

**Resolved: dropped.** See §Closure below.

Note `gt_functions` already surfaces this correctly: `ranking_query` sets
`boundary_tie: True` on exactly these items, so Step 3 can filter them without any GT
change. Dropping all 46 would cost 0.02% of Stage 1.

---

## Status

With `top_n` bound, **every question/answer-mismatch defect identified in Step 2 is
closed.** All eight categories were re-checked this pass: each category's answer-shaping
parameters are now either present in its template text (`interaction_network_query`'s
`{N}`, `ranking_ordering_query`'s `{N}`/`{percentile}`, `threshold_query`'s
`{threshold}`) or explicitly bound and consistent within each phrasing
(`gene_driver_reasoning`'s `top_n`). No new mismatch pattern was found.

**Step 2 is formally closed.** See §Closure. Step 3 (template filling) has not been
started.

---

## Closure — the 46 tied ranking assignments removed

The 46 boundary-tie assignments are dropped. The filter lives in
`build_generation_plan.py`, applied at emit time, so it is reproducible rather than a
one-off edit to the JSON — re-running the builder at seed 20260803 reproduces both the
plan and the removals exactly.

An item is dropped when the value at rank N equals the value at rank N+1. At that cut
"the top N genes" has no unique membership: the gene at rank N and the one at N+1 are
indistinguishable on the only criterion the question states, so no single gene set is
the answer. Emitting one would bake an arbitrary alphabetical tie-break into ground
truth and present it as fact.

Every removal is logged with its evidence in `qa_generation/dropped_assignments.json`:

```json
{
  "patient": "GSM1201749",
  "stage": 1,
  "category": "ranking_ordering_query",
  "template_index": 4,
  "entities": {
    "spec": 50,
    "direction": "top",
    "shape": "count",
    "band": "wide"
  },
  "reason": "boundary_tie",
  "evidence": {
    "cut_at_rank": 50,
    "value_at_rank_n": 1.753923,
    "value_at_rank_n_plus_1": 1.753923,
    "gene_at_rank_n": "MMP14",
    "gene_at_rank_n_plus_1": "PKN1",
    "units": "log1p(TPM)"
  }
}
```

All 46 log entries carry `value_at_rank_n == value_at_rank_n_plus_1` — verified, not
asserted. Distribution by requested size: 100 (13), 75 (8), 50 (6), 1.0%/2.0%/5.0%
(5 each), 40 (4), remainder (5). Wider cuts tie more often, since the value
distribution flattens away from the top.

### Final adjusted totals

| Stage | Target | Achievable | % of target |
|---|---|---|---|
| Stage 1 | 200,000 | **199,954** | 99.98% |
| Stage 2 | 50,000 | **19,793** | 39.59% |
| **Total** | 250,000 | **219,747** | 87.9% |

Per category:

| Category | Assignments |
|---|---|
| `direct_abundance_query` | 40,000 |
| `threshold_query` | 40,000 |
| `ranking_ordering_query` | **39,954** (−46) |
| `comparative_query` | 40,000 |
| `interaction_network_query` | 40,000 |
| `disease_subtype_classification` | 2,687 |
| `comparative_differential_reasoning` | 8,553 |
| `gene_driver_reasoning` | 8,553 |

The removal costs 0.023% of Stage 1. Stage 2 is unaffected.

### Final validation — full scale, no sampling

| Check | Scope | Result |
|---|---|---|
| Duplicate `(patient, category, entity)` triples | all 219,747 | **0** |
| Template/parameter mismatches | all 219,747 | **0** |
| Unbound placeholders | all 219,747 | **0** |
| Assignments referencing an unmaterialized patient | all 219,747 | **0** |
| **Ranking boundary ties** | **all 39,954** | **0** |
| **Threshold degenerate answers** | **all 40,000** | **0** |
| GT execution | 12,053 items | **0 failures**, 0 degenerate |
| `gene_driver_reasoning` count == bound `top_n` | all 8,553 | **all match** |
| `interaction_network_query` count == bound `N` | 500 sampled | **all match** |
| `gt_functions` suite | 100 tests | **all passing** |

The degeneracy sweep is now exhaustive rather than sampled — every ranking and every
threshold assignment was checked, which is what caught the 46 in the first place.

## Step 2: closed

Every defect raised across the three passes is resolved:

| Issue | Pass | Resolution |
|---|---|---|
| Materialization gap (2,004 patients) | follow-up | Real data-quality exclusion; assignments dropped |
| `interaction_network_query` unstated `N` | follow-up | `{N}` added to all 10 templates; `n` required |
| `gene_driver_reasoning` unbound `top_n` | cleanup | `top_n` required, keyed per template |
| Ranking boundary ties (46) | closure | Dropped with logged evidence |

`generation_plan.json` — **219,747 assignments** — is final and ready for Step 3.

## Files changed

| File | Change |
|---|---|
| `qa_generation/gt_functions.py` | `gene_driver_reasoning`'s `top_n` required and range-validated |
| `qa_generation/build_generation_plan.py` | `GENE_DRIVER_TOP_N` keyed by template index; bound per assignment |
| `qa_generation/tests/test_gt_functions.py` | 88 → 100 tests |
| `qa_generation/generation_plan.json` | All 8,553 driver assignments now carry `top_n` |
| `qa_generation/templates/stage2.yaml` | **Unchanged** — Task 3 found no rewording needed |
