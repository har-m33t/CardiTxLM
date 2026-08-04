# Step 3 — threshold_query Refill

Re-filled exactly the 40,000 `threshold_query` pairs after the threshold retune.
The other 179,747 filled pairs were left in place and are verified byte-identical.

| | Result |
|---|---|
| Pre-flight | **PASS** — retuned plan confirmed, other categories' checksums unchanged |
| Deleted | 40,000 `threshold_query` lines, indices matching the retuned set exactly |
| Re-filled | **40,000**, 0 excluded, 9.7 s |
| Stage 1 total | **199,954** — restored, 0 duplicate `assignment_index` |
| Answer size | 11,682 → **347 chars** mean (33.7× smaller) |
| Cost estimate | **$8.98 → ~$16.6** — needs updating, but not for the original reason |

---

## Pre-flight

Both required checks ran before anything was touched:

| Check | Result |
|---|---|
| `threshold_query` count == 40,000 | ✓ |
| All `n_matching` within 5–30 | ✓ (min 5, max 30) |
| All directions == `above` | ✓ (40,000 / 40,000) |
| Other 7 categories' SHA-256 unchanged vs pre-retune | ✓ all 7 |

The checksum comparison used `threshold_retune_stats.json`'s recorded
`checksums_before`, so it verifies against the genuine pre-retune state rather
than a re-derivation.

## Task 1 — Clean deletion

| | |
|---|---|
| Lines before | 199,954 |
| Removed | **40,000** (all `category == "threshold_query"`) |
| Removed indices == the 40,000 retuned assignment indices | **exact match** |
| Lines kept | **159,954** |
| SHA-256 over kept lines, before vs after rewrite | **identical** |
| Other categories' line counts | unchanged (40,000 / 39,954 / 40,000 / 40,000) |

The deletion was verified by hashing the surviving lines during the filter and
re-hashing the written file — not merely by counting.

## Task 2 — Targeted re-fill

`fill_templates.py --resume` reported `resuming — 179747 assignments already
written`, attempted exactly **40,000**, wrote 40,000, excluded 0. Stage 2 was not
opened for writing at all (`stage2_written: 0`).

Its pre-flight ran again inside the fill and passed, so the refill could not have
executed against a stale plan.

## Task 3 — Spot-check against direct GT execution

1,500 refilled entries were re-executed through `gt_functions.threshold_query`
directly, and the rendered answer text parsed back and compared field by field —
not trusting the fill pipeline's own report:

| Check | Mismatches |
|---|---|
| Gene count stated in answer vs GT `n_matching` | **0** |
| Gene list in answer vs GT `genes` (order and membership) | **0** |
| Threshold stated in answer vs bound threshold | **0** |
| Gene-list size within 5–30 | **0 violations** (min 5, median 17, max 30) |

Full sweep across all 40,000 refilled answers: **40,000 / 40,000** state a gene
count within 5–30 (min 5, median 17, max 30).

Example:

> 8 genes in this sample have expression above 7.6 (log1p(TPM)): MMP1 (8.1045),
> MT-CO1 (8.081), CXCL6 (7.9124), COL3A1 (7.9059), S100A6 (7.8924), FN1 (7.8689),
> TFPI2 (7.698), SERPINE1 (7.6821).

## Task 4 — Final counts and dedup

| Check | Result |
|---|---|
| `filled_pairs_stage1.jsonl` total lines | **199,954** ✓ |
| Per category | 40,000 / 40,000 / 39,954 / 40,000 / 40,000 |
| Duplicate `assignment_index` in Stage 1 | **0** |
| Non-`threshold_query` lines still byte-identical | ✓ (SHA-256 match) |
| `filled_pairs_stage2.jsonl` | **19,793**, untouched |
| Union of all indices == 219,747 exactly once | ✓ |

## Task 5 — Cost re-measured

### Calibration

The previous flag used a crude 4-chars-per-token assumption. This re-measure
calibrates against the cost doc's own reference call: it recorded
`threshold_query` at 896 input / 101 output tokens for **"an 8-gene list"** — an
answer shape the retuned corpus now produces. Taking 200 real 8-gene entries:

- Given-block median 331 chars ÷ 128 measured variable input tokens = **2.59 chars/token**
- Question+answer median 253 chars ÷ 101 measured output tokens = **2.50 chars/token**

Far denser than prose (~4), because gene symbols and 4-decimal values tokenize
poorly. The earlier estimate understated volume partly for this reason.

### Result

| | Doc model | Re-measured |
|---|---|---|
| Calls | 250,000 (assumed) | **219,747** (actual) |
| Input, cache hit | 204.80M | 173.83M |
| Input, cache miss | 26.18M | **42.51M** |
| Output | 16.92M | **36.14M** |
| **Total tokens** | **247.90M** | **252.48M** |
| **Cost** | **$8.98** | **~$16.56** |

Sensitivity to the chars/token ratio, since it is the main uncertainty:

| chars/token | Total tokens | Cost |
|---|---|---|
| 2.20 | 266.4M | $19.41 |
| **2.59 (calibrated)** | **252.5M** | **$16.56** |
| 3.00 | 241.7M | $14.36 |
| 4.00 | 224.8M | $10.89 |

### Does the estimate hold? No — but the original failure mode is fixed

**Total token volume is now essentially as modelled** (252M vs 248M). The
threshold blow-up is gone: pre-retune, this corpus projected to **~609M tokens and
~$92**. The retune removed that.

**The cost is still ~1.8× the doc's figure, for a different reason.** Volume sits
in the expensive tiers: cache-miss input is 42.5M against 26.2M modelled, and
output 36.1M against 16.9M. The cached prefix (billed at 1/50th) is close to
plan, so the mix shifted rather than the total. Two contributors:

1. The doc measured one short call per category and treated it as the mean; real
   per-category medians are longer.
2. The chars/token ratio for gene-list content is ~2.5, not ~4.

**Recommendation:** update `deepseek_cost_estimate.md` to ~$16–17 with the
sensitivity band. It remains a small absolute number, and the 5M-token signup
grant still offsets part of it — this is a correction to a stale figure, not a
budget problem.

---

## Flagged, not fixed: `ranking_ordering_query` is now the constraint

The retune was scoped to `threshold_query`, and `threshold_query` is now fully
clean — **0 of 40,000** answers exceed `max_tokens: 512` (p95 236 tokens, max
253). But at the calibrated 2.50 chars/token, the truncation exposure has moved:

| Category | p50 out | p95 | Max | Over 512 |
|---|---|---|---|---|
| `ranking_ordering_query` | 223 | 809 | **1,959** | **8,839 / 39,954** |
| all others | ≤ 233 | ≤ 244 | 253 | **0** |

**8,839 answers (4.0% of the corpus) are still projected to exceed `max_tokens`**,
now entirely in `ranking_ordering_query` — whose "wide" band goes to 100 genes and
whose percentile band reaches 290. Truncation there is the same correctness
failure as before: a cut-off ranked list becomes a wrong answer.

`ranking_ordering_query` is also now the single largest cost line (15.2M
cache-miss input + 14.4M output, ~35% of the billable total), ahead of
`threshold_query`.

Two options, both out of this task's scope: raise `max_tokens` (cost only), or
retune the ranking band the way thresholds were just retuned (cost and answer
quality). The ranking diversification introduced in Step 2 deliberately widened
those windows to break mitochondrial dominance, so narrowing them trades against
that — it needs its own decision rather than a silent change.

## Files

| File | Change |
|---|---|
| `qa_generation/filled_pairs_stage1.jsonl` | 40,000 `threshold_query` entries replaced; other 159,954 byte-identical |
| `qa_generation/filled_pairs_stage2.jsonl` | **Untouched** (19,793) |
| `qa_generation/step3_excluded.jsonl` | Still empty — no rejections |
| `qa_generation/step3_stats.json` | Refreshed by the resume run |

Step 4 (DeepSeek generation) has not been started.
