# Pilot Batch — Truncation Fix and Real-Cost Measurement

Two parts: raise the output-token cap that was risking silent truncation of ranked
gene lists, then run 300 real API calls to measure actual cost and generation
quality before committing the full corpus.

| | Result |
|---|---|
| Part 1 — `max_tokens` fix | **Per-category override added; ranking → 3,000.** 0 truncations in the pilot |
| Part 2 — pilot | **300/300 ok**, 0 errors, **$0.0320** actual billed |
| Factual accuracy | **1 real defect found** → root-caused to a prompt bug, **fixed and re-verified** |
| Extrapolated full-run cost | **~$21.07** (upper bound $23.40) |
| **Budget verdict** | **⚠ Over the $20 budget. Do not start the full run as-is.** |

---

## Part 1 — Truncation fix

### The config did support a per-category override — after a small addition

`deepseek_client.py` applied one global `self.max_tokens` to every call. Rather
than raising it globally, I added a per-category table so other categories keep a
tight cap:

```python
MAX_TOKENS_BY_CATEGORY = {"ranking_ordering_query": 3000}
```

`DeepSeekClient.max_tokens_for(category)` falls back to the global default (512)
for everything else. Verified:

| Category | max_tokens |
|---|---|
| `ranking_ordering_query` | **3,000** |
| all seven others | 512 (unchanged) |

**Step 2's ranking bands were not touched**, per the constraint — those windows
were widened deliberately to break mitochondrial dominance (Step 2 Task 5, which
cut top-20 gene concentration from 76.3% to 26.0%). The token limit moved instead.

### Why 3,000 and not the suggested 2,200–2,500

Ranking's longest question+answer is 4,918 characters. At a deliberately
pessimistic 2.0 chars/token that is 2,459 tokens — so 2,500 would have left ~2%
headroom. **The pilot proved that too tight: the real observed maximum was 2,705
output tokens**, which would have truncated at 2,500. 3,000 leaves 9.8% headroom
against real measured usage.

### Coverage confirmed

Recomputed across all 39,954 ranking entries in `filled_pairs_stage1.jsonl`:

| chars/token assumed | Max tokens | Exceeding 3,000 |
|---|---|---|
| 2.5 | 1,967 | **0** |
| 2.2 | 2,235 | **0** |
| 2.0 (pessimistic) | 2,459 | **0** |
| 1.8 (absurd) | 2,732 | **0** |

And against real API usage in the pilot: **0 of 300 responses reached their cap.**
14 of 54 ranking responses (26%) exceeded the old 512 limit — the fix was needed.

| Category | Cap | p50 out | max out | At cap |
|---|---|---|---|---|
| `ranking_ordering_query` | 3,000 | 304 | **2,705** | **0** |
| `threshold_query` | 512 | 236 | 321 | 0 |
| `interaction_network_query` | 512 | 166 | 275 | 0 |
| all others | 512 | ≤133 | ≤147 | 0 |

---

## Part 2 — Pilot batch

### Sample

300 pairs, stratified by largest-remainder so each category's pilot share matches
its corpus share:

| Category | Corpus share | Pilot |
|---|---|---|
| `direct_abundance_query` | 18.20% | 55 |
| `threshold_query` (retuned) | 18.20% | 55 |
| `ranking_ordering_query` (new cap) | 18.18% | 54 |
| `comparative_query` | 18.20% | 54 |
| `interaction_network_query` | 18.20% | 54 |
| `comparative_differential_reasoning` | 3.89% | 12 |
| `gene_driver_reasoning` | 3.89% | 12 |
| `disease_subtype_classification` | 1.22% | 4 |

Both corrected categories are represented, and the sample includes a 4,733-char
answer that exercises the new cap.

### Format, truncation, skips

| Check | Result |
|---|---|
| Status | **300 ok / 0 skip / 0 error** |
| `Question:` / `Answer:` parse anomalies | **0** |
| Responses truncated at their cap | **0** |
| SKIP responses | 0 — and 0 inputs had missing/`insufficient_data` GT, so correct |
| Median latency | 21.49 s |

No SKIPs is the right outcome: Step 3 already excluded every rejection at fill
time, so no pilot input lacked ground truth.

### Factual accuracy — one real defect, root-caused and fixed

Automated check on all 300: every numeric value and gene symbol in the GT compared
against the paraphrase.

**Gene symbols: 0 dropped, 0 invented, across all 300.**

Numeric values: 12 flagged, of which **11 were benign paraphrasing** — a spelled
numeral for the list size ("top 10" → "the ten most highly expressed", "Top 15" →
"the top fifteen") with every gene and every expression value intact.

**1 was a genuine numeric change**, and per the guardrail I stopped and
investigated before extrapolating:

> GT: `...separable at ROC-AUC 0.7806 (sd 0.1055...)`
> Response: `...approximately 0.78 ROC-AUC (standard deviation 0.11...)`

**Root cause — a prompt bug, not a client bug.** Two defects in
`stage2_generation_prompt.txt`:

1. It was **missing the no-rounding restriction** that
   `stage1_generation_prompt.txt` carries verbatim ("Do not change, round, or
   approximate any numeric value in the answer"). Stage 2 never forbade rounding.
2. Its worked example **demonstrated rounding**: gold `ROC-AUC 0.78 ± 0.11`,
   output "about 0.78 ROC-AUC" — while real GT carries `0.7806 (sd 0.1055)`.

The model rounded because the prompt both permitted and modelled it. At the
observed 1-in-12 rate this would have produced roughly **713 rounded statistics**
across the 8,553 differential items.

**Fix applied** to `stage2_generation_prompt.txt` only (Stage 1 untouched, md5
verified): added the no-rounding restriction, and corrected the example so it no
longer models rounding.

**Re-verified** by re-running all 28 Stage 2 pilot items against the patched
prompt: **12/12 `comparative_differential_reasoning` responses now preserve both
`0.7806` and `0.1055` exactly.** The 3 remaining numeric flags are the benign
spelled-numeral case, with gene lists confirmed complete.

### Real token usage and cost

Straight from the API's `usage` object, not estimated:

| | Pilot (300 calls) |
|---|---|
| Cache-hit input | 213,120 tok (69.0%) |
| Cache-miss input | 95,542 tok |
| Output | 60,162 tok |
| **Billed** | **$0.0320** ($0.000107/call) |

Warm-cache prefix measured at **768 tokens**, matching the cost doc's assumption.
21 of 300 calls (7%) were cold-start misses — a fixed startup cost that amortises
away over a full run.

### Extrapolation to 219,747 calls

Per-category, using each category's own measured per-call usage at steady state
(warm cache):

| Category | $/call | Corpus | Projected |
|---|---|---|---|
| `ranking_ordering_query` | 0.000210 | 39,954 | **$8.40** |
| `threshold_query` | 0.000099 | 40,000 | $3.96 |
| `interaction_network_query` | 0.000086 | 40,000 | $3.43 |
| `comparative_query` | 0.000056 | 40,000 | $2.23 |
| `direct_abundance_query` | 0.000025 | 40,000 | $1.01 |
| `comparative_differential_reasoning` | 0.000113 | 8,553 | $0.97 |
| `gene_driver_reasoning` | 0.000103 | 8,553 | $0.88 |
| `disease_subtype_classification` | 0.000070 | 2,687 | $0.19 |
| **Total** | | **219,747** | **$21.07** |

- **Best estimate: $21.07** (steady state, warm cache)
- **Conservative upper bound: $23.40** (flat pilot per-call rate, including its
  cold starts)

Against the earlier $16–19.50 estimate, the real number lands slightly above the
top of that range. `ranking_ordering_query` is 40% of the total — its 485
output-tokens-per-call is more than double any other category, a direct
consequence of the wide bands.

---

## ⚠ Budget verdict: over by ~$1.07. Do not start the full run yet.

| | Amount |
|---|---|
| Budget | $20.00 |
| Best estimate | **$21.07** |
| **Margin** | **−$1.07 (−5.3%)** |
| Upper bound | $23.40 (−17.0%) |

The 5M-token signup grant is worth roughly $0.70–1.40 depending on which tier it
offsets, which lands the effective cost at **$19.67–20.37** — i.e. break-even at
best, with **no** meaningful buffer, and no headroom at all for a retry storm or a
partial re-run.

### Concrete options

| Option | Action | Result |
|---|---|---|
| **A — trim ranking (recommended)** | Drop **5,082** `ranking_ordering_query` pairs (12.7% of it, 2.3% of corpus) | ~$20.00 |
| **B — trim for real margin** | Drop **14,606** ranking pairs (36.6% of it, 6.6% of corpus) | ~$18.00, 10% margin |
| **C — add credit** | Add **$1.07** to break even, or **$3.17** for a 10% buffer | full 219,747 corpus preserved |
| **D — pro-rata trim** | Cut all five Stage-1 categories by 5.6% (~12,324 pairs) | ~$20.00, even quality impact |

**Recommendation: C if the extra credit is available, otherwise A.**
`ranking_ordering_query` is the right place to trim because it is 40% of spend for
18% of the corpus, and it is the most internally redundant category — its 39,954
pairs draw on only 13 distinct window specs. Dropping 12.7% of them costs little
diversity. Trimming means removing *assignments*, **not** narrowing the bands —
narrowing would undo the mitochondrial-dominance fix.

Whichever you pick, the trim should be a scoped edit to `generation_plan.json` and
a matching filtered re-emit of `filled_pairs_stage1.jsonl`, not a replan.

---

## Files

| File | Change |
|---|---|
| `qa_generation/deepseek_client.py` | Per-category `max_tokens` override; ranking → 3,000 |
| `qa_generation/prompts/stage2_generation_prompt.txt` | No-rounding restriction added; example no longer models rounding |
| `qa_generation/pilot_batch_results.jsonl` | 300 responses with usage and cost |
| `qa_generation/pilot_stage2_recheck.jsonl` | 28 Stage 2 responses re-run against the fixed prompt |
| `qa_generation/pilot_batch_input.jsonl` | The stratified sample, for reproducibility |

Assignments, GT functions and filled pairs were **not** modified. The full Step 4
run has **not** been started.
