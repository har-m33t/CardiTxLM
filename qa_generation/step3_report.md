# Step 3 — Template Filling

Filled every assignment in the closed Step 2 plan by calling the matching
`gt_functions` function with the exact bound parameters, and rendered the
deterministic answers Step 4 will paraphrase.

**Result: 219,747 attempted, 219,747 filled, 0 excluded.** The real run matches
Step 2's validated expectations exactly — no discrepancy to investigate.

| | Expected (Step 2) | Actual (this run) | Match |
|---|---|---|---|
| Stage 1 | 199,954 (200,000 − 46 dropped ties) | **199,954** | ✓ |
| Stage 2 | 19,793 | **19,793** | ✓ |
| **Total** | **219,747** | **219,747** | ✓ |
| Excluded (rejection reasons) | 0 expected | **0** | ✓ |

Run time 98 s. Outputs: `filled_pairs_stage1.jsonl` (599 MB),
`filled_pairs_stage2.jsonl` (15 MB), `step3_excluded.jsonl` (empty).

---

## Pre-flight — the plan is the closed Step 2 state

Checked before any work, and re-checked inside `fill_templates.py` on every run
(it exits non-zero rather than filling against a stale plan):

| Check | Result |
|---|---|
| 46 `boundary_tie` ranking assignments removed (39,954 remain) | ✓ |
| `gene_driver_reasoning.top_n` bound on all 8,553 | ✓ |
| `interaction_network_query.n` bound on all 40,000 | ✓ |
| Total assignments == 219,747 | ✓ |
| `dropped_assignments.json` present, 46 entries, all `boundary_tie` | ✓ |

## Task 5 — Live execution vs Step 2's expectations

Re-verified by execution, not assumed. Every category matched its planned count
exactly:

| Category | Planned | Filled | Excluded |
|---|---|---|---|
| `direct_abundance_query` | 40,000 | **40,000** | 0 |
| `threshold_query` | 40,000 | **40,000** | 0 |
| `ranking_ordering_query` | 39,954 | **39,954** | 0 |
| `comparative_query` | 40,000 | **40,000** | 0 |
| `interaction_network_query` | 40,000 | **40,000** | 0 |
| `disease_subtype_classification` | 2,687 | **2,687** | 0 |
| `comparative_differential_reasoning` | 8,553 | **8,553** | 0 |
| `gene_driver_reasoning` | 8,553 | **8,553** | 0 |

**Zero exclusions is the expected outcome, not a suspicious one.** Step 2 built the
plan by executing these same functions and filtered every rejection class at
planning time — unresolved subtypes, non-permitted comparison groups, degenerate
thresholds, boundary ties, unbound counts. `step3_excluded.jsonl` exists and is
empty; the rejection path was exercised during development and writes correctly
when a rejection occurs.

Integrity of the written output:

| Check | Result |
|---|---|
| Records with a missing required field | **0** |
| `filled_question` with an unsubstituted `{placeholder}` | **0** |
| Empty answer strings | **0** |
| `assignment_index` unique within each file, disjoint across files | ✓ |
| Union of indices == all 219,747 exactly once | ✓ |
| Stage 1 uses `deterministic_answer` | ✓ |
| Stage 2 uses `gold_answer` | ✓ |
| `comparison_group: "neg_hard"` on all 8,553 differential rows, absent elsewhere | ✓ |

Field names are taken from `prompts/stage1_generation_prompt.txt` and
`prompts/stage2_generation_prompt.txt`, which Step 4 formats with these keys.

## Task 6 — Incremental writes and resumability

Records are appended line-by-line and flushed every 2,000, never held in memory.
Observed mid-run: `filled_pairs_stage1.jsonl` at 43,575 lines, then 58,260 lines
20 s later, while the process was still running.

Resumability was tested end-to-end rather than asserted:

1. Fresh run stopped at 5,000 records (simulating an interruption).
2. `--resume` reported `resuming — 5000 assignments already written`, attempted the
   remaining 214,747, and skipped 5,000.
3. Final output: 199,954 + 19,793 = 219,747, **no duplicate `assignment_index`**,
   files disjoint, union covering all 219,747 exactly once.

Resume keys on `assignment_index` present in the existing outputs, and tolerates a
truncated final line from a hard kill.

## Answer rendering

Answers are rendered deterministically from the GT payload — Step 4 rewords them
and is forbidden from changing facts. Nothing is added that the payload does not
carry, and **no gene list is truncated**: a shortened list would present a partial
answer as a complete one.

Units are `log1p(TPM)` throughout, taken from the payload (see the flag below).

| Category | Example answer |
|---|---|
| `direct_abundance_query` | "The expression level of RAPGEF5 in this sample is 0.1086 (log1p(TPM))." |
| `comparative_query` | "…G0S2 is the higher of the two, by 4.8495 log1p(TPM) units (log2 fold change 6.9963 for G0S2 relative to SGCZ)." |
| `interaction_network_query` | "RAPGEF1's top 5 co-expressed partners are ZZEF1, WDTC1, DNM2, DGKZ, TAOK2. In this sample, their expression levels are …" |
| `disease_subtype_classification` | "disease_confirmed_subtype: coronary_artery_disease" |
| `gene_driver_reasoning` | "Top 10 stable elastic-net signal genes for cardiovascular disease vs. random bulk tissue (nonzero in all 5 CV folds): FCGR2A, GFRA1, …" |

`{comparison_group}` is filled with the readable label
"tissue-matched samples without confirmed cardiovascular disease" (the stage-2
prompt requires a real verified label in the filled question); the internal name
`neg_hard` is carried in the separate `comparison_group` field.

---

## Flagged for Step 4 — three issues found while filling

None is fixable inside Step 3's scope (the plan, GT functions and templates are
read-only here), and none blocks the filled output being correct. All three affect
Step 4.

### 1. Answer length breaks the configured `max_tokens` — blocking

`deepseek_cost_estimate.md` sets `max_tokens: 512` on the reasoning that "no
measured call exceeded 101 output tokens". That held for the short sample answers
it was calibrated on. Against the real corpus it does not:

| Category | Median answer | p95 | Max | Over ~512 tokens |
|---|---|---|---|---|
| `threshold_query` | 4,165 chars (~1,041 tok) | 39,106 | 48,807 (~12,201 tok) | **23,428 / 40,000** |
| `ranking_ordering_query` | 495 | 1,956 | 4,833 | 1,546 / 39,954 |
| all others | ≤ 443 | ≤ 443 | 498 | 0 |

**24,974 of 219,747 answers (11.4%) exceed `max_tokens: 512`** and would be
truncated mid-list. Truncation here is a correctness failure, not just a cost one:
a cut-off gene list silently becomes a wrong answer. 35,859 of the 40,000 threshold
answers exceed the 101-token figure the estimate assumed.

### 2. The cost model needs re-running

Total rendered answer text is ~528M characters (**~132M tokens**), against the
estimate's 20.68M cache-miss input plus 14.04M output. The answer appears twice per
call — once in the prompt, once restated in the output — so the variable portion is
roughly 6× the modelled input and ~9× the modelled output. `threshold_query` alone
is **88.4%** of all answer text.

The $7.26–8.98 figure will not survive contact with this corpus. Re-measure before
committing to the run.

Both issues have the same root: Step 2's threshold targets ask for 500–2,500 genes
in the "below" direction and up to 500 "above". That produces true but unusably long
answers. The fix belongs in Step 2 (retune `BELOW_TARGET_COUNTS` / `ABOVE_TARGET_COUNTS`
to bounded sizes and re-run the plan), not in Step 3 — the answers as written are
faithful to the questions as planned.

### 3. Stale units in both generation prompts

The worked examples in `stage1_generation_prompt.txt` say **`log2-CPM`**
("8.42 (log2-CPM)"). Actual GT units have been **`log1p(TPM)`** since the correction
pass. Few-shot examples carrying wrong units invite the paraphraser to reproduce
them, and the prompt simultaneously forbids changing units — a contradiction it
cannot satisfy.

`stage2_generation_prompt.txt` has a related problem: its Comparative/Differential
worked example shows a **per-gene** gold answer ("elevated NPPA and NPPB… reduced
ATP2A2"). This project's GT for that category is corpus-level and explicitly states
that no per-gene contrast exists. That example models exactly the fabrication its own
restrictions prohibit.

Prompt files are Step 4's assets and were not modified here.

---

## Outputs

| File | Contents |
|---|---|
| `qa_generation/filled_pairs_stage1.jsonl` | 199,954 records, `deterministic_answer` (599 MB) |
| `qa_generation/filled_pairs_stage2.jsonl` | 19,793 records, `gold_answer` + `comparison_group` (15 MB) |
| `qa_generation/step3_excluded.jsonl` | 0 records — no assignment was rejected |
| `qa_generation/step3_stats.json` | Machine-readable counts behind this report |
| `qa_generation/fill_templates.py` | Reproducible, `--resume`-capable |

Step 4 (DeepSeek generation) has not been started.
