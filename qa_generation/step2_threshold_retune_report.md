# threshold_query — Threshold Retune

Step 3 validation found `threshold_query` answers running to hundreds or thousands
of genes: true answers, but data dumps rather than learnable ones, and 23,428 of
40,000 exceeded Step 4's configured `max_tokens: 512`, where truncation would turn
a correct answer into a wrong one.

This pass retunes those 40,000 assignments' bound thresholds so every answer lists
**5–30 genes**. Nothing else in the plan is touched — verified by per-category
checksum.

| | Before | After |
|---|---|---|
| Match count, median | **258** | **17** |
| Match count, max | **3,158** | **30** |
| Out of the 5–30 range | **31,758 / 40,000 (79.4%)** | **0 / 40,000** |
| Rendered answer, median | 4,165 chars (~1,041 tok) | **348 chars (~87 tok)** |
| Rendered answer, max | 48,807 chars (~12,201 tok) | **564 chars (~141 tok)** |
| Answers over `max_tokens: 512` | 23,428 | **0** |
| Category answer text | 467.3M chars | **~13.9M chars** |

Assignment count unchanged at 40,000; total plan unchanged at 219,747.

---

## 1. Why 5–30

Grounded rather than picked round. At ~18 characters per `GENE (value)` entry, 30
genes renders to roughly 600 characters (~150 tokens) — inside `max_tokens: 512`
with headroom, and in line with the list categories already in the corpus
(`ranking_ordering_query` median 495 chars, `interaction_network_query` 264). The
floor of 5 keeps the answer a genuine list rather than a near-singleton. Measured
outcome: p95 = 535 chars, max = 564, so the whole category now sits below the
smallest budget any other list category needs.

## 2. Scope of the original problem

| Direction | Assignments | Median match | Max | In range (5–30) |
|---|---|---|---|---|
| `above` | 24,880 | 90 | 570 | 8,242 |
| `below` | 15,120 | 1,951 | 3,158 | **0** |
| **total** | **40,000** | **258** | **3,158** | **8,242** |

## 3. The `below` direction was structurally infeasible

Not merely badly tuned — **incapable** of hitting the range at any threshold.

The filtered pool's lower tail is a mass of exact zeros: a median of **422 genes
(max 5,534)** sit at exactly `0.0` in a given sample. So any threshold above zero
already matches that entire mass, and any threshold at or below zero matches
nothing — there is no value in between. Only **1 sample in 200** has 30 or fewer
genes at the minimum. That is why `below` scored **0 / 15,120** in range, with a
median of 1,951 matches.

Per the task's direction clause, the **15,120 `below` assignments were rebound to
`above`**, with `template_index` moved to an above-worded template so question and
answer still agree. This is the one substantive semantic change in this pass, so
stating it plainly:

- **All 40,000 threshold assignments are now `above`.** The category no longer
  contains a "below threshold" question.
- Templates 5 ("Which genes have measured expression below {threshold}?") and 6
  ("Find genes whose transcript abundance falls under {threshold}") are now
  **unused**. Template 7 remains excluded from earlier (states no direction).
  Usable phrasings for this category drop from 8 to 6.

The alternative was to flag all 15,120 as infeasible and leave them broken, which
would have cost 38% of the category. Reversible if you would rather widen the range
for `below` specifically — but note no range below ~450 genes is reachable for it.

## 4. Method

Per-sample, percentile-style derivation rather than a fixed absolute cutoff, since
one absolute number cannot hold a match count steady across samples with different
distributions.

For a target of *m* genes, the threshold is placed **midway between the m-th and
(m+1)-th largest values** in that sample, so exactly *m* genes sit strictly above
it. The midpoint is rounded to 2 dp and the resulting count re-verified; if
rounding leaves the interval, or the two values are tied, that target is skipped
and another tried.

Each patient's *m* values are drawn distinct, so its thresholds stay distinct and
`DIVERSITY_RULE` still holds. Above-worded templates are also sampled distinct
within a patient.

**Infeasible assignments: 0.** Every patient supplied enough distinct in-range
thresholds; no assignment needed flagging or dropping.

## 5. Validation — all 40,000 executed against real GT

| Check | Result |
|---|---|
| Executed against `gt_functions.threshold_query` | **40,000 / 40,000** |
| GT failures | **0** |
| Degenerate answers | **0** |
| Match count within 5–30 | **40,000 / 40,000** |
| Planned `n_matching` == actual GT count | **40,000 / 40,000** |
| Template direction matches bound direction | **0 mismatches** |
| Duplicate `(patient, entity)` within the category | **0** |

Achieved distribution — min **5**, p25 **11**, median **17**, p75 **24**, max **30** —
spread evenly across the range (1,420–1,644 assignments at each of the 26 values),
so answer length varies naturally rather than clustering at one size.

Example rendered answer:

> 6 genes in this sample have expression above 6.65 (log1p(TPM)): S100A9 (7.854),
> LYZ (7.5047), TMSB4X (7.4114), B2M (7.0444), SRGN (6.9816), FCGR3B (6.7856).

## 6. Everything else is byte-identical

Per-category SHA-256 over the serialized entries, in order, before and after:

| Category | Status |
|---|---|
| `direct_abundance_query` | **UNCHANGED** |
| `ranking_ordering_query` | **UNCHANGED** |
| `comparative_query` | **UNCHANGED** |
| `interaction_network_query` | **UNCHANGED** |
| `disease_subtype_classification` | **UNCHANGED** |
| `comparative_differential_reasoning` | **UNCHANGED** |
| `gene_driver_reasoning` | **UNCHANGED** |
| `threshold_query` | CHANGED (intended) |

Total assignments unchanged at **219,747**; `threshold_query` unchanged at
**40,000**. The script aborts before writing if any other category's checksum
moves.

## 7. Knock-on effect on Step 4's cost model

Not re-estimated here (out of scope), but the driver of the overrun is gone. Total
rendered answer text across the corpus falls from ~528M chars (~132M tokens) to
roughly **75M chars (~19M tokens)** — `threshold_query` was 88.4% of the old total
and is now ~19% of the new one. That is close to the ~35M tokens the original
estimate modelled, so the $7–9 figure is plausible again, but it should be
re-measured against the refilled corpus rather than assumed.

## Files

| File | Change |
|---|---|
| `qa_generation/generation_plan.json` | `threshold_query` entities and template indices retuned; all else untouched |
| `qa_generation/retune_thresholds.py` | **New** — reproducible at seed 20260803, `--dry-run` supported |
| `qa_generation/threshold_retune_stats.json` | Machine-readable before/after and checksums |

## Next

Step 3 must be re-run **for `threshold_query` only** — the other 179,747
assignments' filled pairs are still valid, since their plan entries are unchanged.
`fill_templates.py --resume` keys on `assignment_index`, so removing the 40,000
threshold records from `filled_pairs_stage1.jsonl` and resuming would refill exactly
those. Not done here; that is the next scoped step.
