# DeepSeek V4-Flash — cost estimate for the Step 4 paraphrase run

**Prepared:** 2026-08-03 · **Revised:** 2026-08-03 (canonical prompts) ·
**Pricing source checked:** 2026-08-03 · **Model:** `deepseek-v4-flash`

Bottom line: **~$8.98 for the full 250K-call run, ~$8.80 net of the 5M-token
signup grant.** Token counts below are *measured*, not extrapolated — every
number came back from real API calls against the canonical generation prompts
(see `connectivity_test_result.md`).

> **Revision note.** The first version of this estimate used a prompt written
> from the description in `qa_generation_pipeline_todo.md`, because the real
> prompt text was not in the repo at the time. The canonical prompts have since
> been added at `prompts/stage{1,2}_generation_prompt.txt` and are now what the
> client sends. They are ~1.7x longer (they carry worked examples), which raises
> total volume from 152M to 248M tokens — but **lowers cost from $10.87 to
> $8.98**, because a longer fixed prefix caches proportionally better and the
> expensive cache-*miss* portion shrinks. All numbers below are re-measured
> against the canonical prompts.

---

## 1. Pricing used

From <https://api-docs.deepseek.com/quick_start/pricing>, checked 2026-08-03:

| Model | Input (cache hit) | Input (cache miss) | Output |
|---|---|---|---|
| `deepseek-v4-flash` | **$0.0028 / 1M** | **$0.14 / 1M** | **$0.28 / 1M** |
| `deepseek-v4-pro` | $0.003625 / 1M | $0.435 / 1M | $0.87 / 1M |

The `$0.14 / $0.28` figures carried in `qa_generation_pipeline_todo.md` are still
correct for cache-miss input and output. What that note was missing is the
**cache-hit tier at $0.0028/1M — a 50x discount on repeated prefix tokens**,
which is what makes this run cheap.

Two forward-looking caveats on the same page:

* A **peak/off-peak policy is announced but not yet in effect**: "During peak
  hours, prices will be 2x the regular prices, applicable to all billing items,"
  peak being 09:00–12:00 and 14:00–18:00 Beijing time (UTC+8). Effective date
  "subject to the official announcement." If this lands before the run,
  scheduling the job outside those windows halves the bill relative to running
  through them.
* DeepSeek "reserves the right to adjust" pricing. Re-check the page before
  committing budget.

---

## 2. Assumptions

| Assumption | Value | Basis |
|---|---|---|
| Stage 1 calls | 200,000 | target from `qa_generation_pipeline_todo.md` |
| Stage 2 calls | 50,000 | same |
| Category mix within a stage | uniform | **Step 2's sampling plan does not exist yet.** See §6. |
| Thinking mode | **disabled** | measured: on by default at `high` effort, costs 2.6x output tokens for no fidelity gain (§5) |
| Cached prefix per call | 768 (Stage 1) / 1024 (Stage 2) tokens | measured warm-cache `prompt_cache_hit_tokens` |
| `max_tokens` | 512 | client default; no measured call exceeded 101 output tokens |
| One call per QA pair | yes | no multi-pair batching in a single prompt |
| Retries | not priced in | measured 0 errors / 27 calls; a few % of retries moves the total by cents |

**Prompt provenance.** The prompts are the canonical files at
`prompts/stage1_generation_prompt.txt` and `prompts/stage2_generation_prompt.txt`,
loaded verbatim by `deepseek_client.py` (`STAGE1_INSTRUCTIONS` /
`STAGE2_INSTRUCTIONS` read them at import, so the two cannot drift apart). Every
token count here was measured against those exact files. Editing a prompt file
changes both the token counts and the cache prefix, so re-measure if either file
changes.

---

## 3. Measured tokens per call

Every row is a real API response with thinking disabled. "in" is
`prompt_tokens`, "out" is `completion_tokens`.

### Stage 1

| Category | Input | Output |
|---|---|---|
| `direct_abundance_query` | 835 | 35 |
| `comparative_query` | 860 | 63 |
| `ranking_ordering_query` | 875 | 75 |
| `interaction_network_query` | 891 | 77 |
| `threshold_query` | 896 | 101 |
| **Average** | **871.4** | **70.2** |

### Stage 2

| Category | Input | Output |
|---|---|---|
| `disease_subtype_classification` | 1105 | 28 |
| `gene_driver_reasoning` | 1143 | 70 |
| `comparative_differential_reasoning` | 1154 | 75 |
| **Average** | **1134.0** | **57.7** |

Of the average input, **768 tokens (Stage 1) / 1024 tokens (Stage 2) are the
cached prefix** — 88.1% and 90.3% respectively. Only the remaining ~100–110
tokens (the `Given:` block: category, template question, filled question, and
ground-truth answer) bill at the miss rate.

The output totals are modest because the answer is a restatement, not a
generation. Longest observed output across all ten categories: 101 tokens
(`threshold_query`, an 8-gene list).

---

## 4. Total volume and cost

### Stage 1 — 200,000 calls

| | Tokens | Rate | Cost |
|---|---|---|---|
| Input, cache hit | 153.60M | $0.0028/M | $0.43 |
| Input, cache miss | 20.68M | $0.14/M | $2.90 |
| Output | 14.04M | $0.28/M | $3.93 |
| **Total** | **188.32M** | | **$7.26** |

### Stage 2 — 50,000 calls

| | Tokens | Rate | Cost |
|---|---|---|---|
| Input, cache hit | 51.20M | $0.0028/M | $0.14 |
| Input, cache miss | 5.50M | $0.14/M | $0.77 |
| Output | 2.88M | $0.28/M | $0.81 |
| **Total** | **59.58M** | | **$1.72** |

### Combined

| | |
|---|---|
| Total tokens | **247.90M** (230.98M in / 16.92M out) |
| **Total cost** | **$8.98** |
| Blended rate | $0.0362 per 1M tokens |
| Less 5M signup grant (~2.0% of volume, ~$0.18) | |
| **Net cost** | **~$8.80** |

The grant barely dents this because the run is cheap in absolute terms — 5M of
248M tokens is 2.0%. Reported grant validity is ~30 days from signup, so it is
worth burning on the Step-4 pilot batch rather than saving for the full run.

---

## 5. What the measurements changed

**Thinking mode is the single biggest cost lever, and it defaults the wrong way
for this job.** `deepseek-v4-flash` has thinking **enabled by default at `high`
effort**, and reasoning tokens bill as output tokens. On identical inputs:

| | Output tokens (5 Stage 1 calls) | Mean latency |
|---|---|---|
| Thinking on (default) | 837 | 2.1–4.2 s |
| Thinking off | 318 | 1.1–1.8 s |

That is **2.63x the output tokens** for a task that is verbatim-preserving
rewriting — the reasoning traces were things like *"Need to paraphrase. Ensure
exact gene symbol NPPA, value 8.42."* Fidelity was identical with it off; every
gene symbol and decimal was preserved in both configurations.

Projected to full scale: **thinking on would cost ~$18.63 instead of $8.98 — a
$9.65 (+107%) premium.** `deepseek_client.py` therefore sends
`"thinking": {"type": "disabled"}` by default, with `--thinking` to re-enable.

(The 2.63x ratio was measured against the earlier, shorter prompt. It is applied
here to the canonical prompts' output volume; the ratio should hold, since it is
a property of the task, but it has not been re-measured against the canonical
text.)

**Cost sensitivity summary:**

| Scenario | Cost |
|---|---|
| **Recommended config** (thinking off, caching on) | **$8.98** |
| No prompt caching at all | $37.07 |
| Thinking mode left on | $18.63 |
| Peak pricing in effect *and* run overlaps peak hours | $17.95 |
| Worst case (thinking on + peak hours) | $37.27 |

Caching matters far more under the canonical prompts than it did under the
shorter draft — it now saves $28.09 (76%) rather than $13.17 (55%), because the
prompts are longer and almost all of that length is cacheable prefix. **If prefix
caching ever silently stops hitting, this job gets 4x more expensive**, so the
cumulative cache-hit rate in the run log is the number to watch.

---

## 6. Caveat on the category mix

The uniform-mix assumption is the weakest input here, because **Step 2's
sampling plan hasn't been written yet** — it decides how many pairs per patient
per category, and `qa_generation_pipeline_todo.md` already flags that Stage 2
may land short of 50K once the corrected Comparative/Differential and
Gene-Driver constraints are applied.

The exposure is small, and smaller than before the prompt correction. The
cheapest and dearest Stage 1 categories differ by only 61 input / 66 output
tokens — and because ~88% of every input is the identical cached prefix, mix
skew now moves almost nothing on the input side. Even a badly skewed mix moves
the total by roughly ±10% (±$0.90). A Stage 2 that comes in under 50K calls only
reduces the bill. **No re-estimate is needed before Step 2 lands** — this is a
$9 job under every mix that could plausibly result.

---

## 7. Reproducing these numbers

Measured token counts come from the connectivity test; see
`connectivity_test_result.md` for the raw per-call usage objects. The arithmetic
above is a straight application of §1's rates to §3's averages at §2's call
counts.
