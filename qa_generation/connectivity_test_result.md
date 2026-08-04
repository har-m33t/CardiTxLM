# DeepSeek connectivity test — results

**Run:** 2026-08-03 · **Re-run against canonical prompts:** 2026-08-03 ·
**Model:** `deepseek-v4-flash` · **Key:** read from `DEEPSEEK_API_KEY`
(real key, present in `.env`, never printed or logged)

**Verdict: pass.** 37 calls total, 0 errors, 0 retries. Response format matches
what the client parses, token usage is measured and consistent, and latency
supports the batching strategy. Two configuration problems were found and fixed
along the way — see §5.

> **Revision note.** The first test run used a prompt written from the
> description in `qa_generation_pipeline_todo.md`, since the real text was not
> yet in the repo. The canonical prompts have since been added at
> `prompts/stage{1,2}_generation_prompt.txt`, the client now loads them verbatim,
> and **all ten categories plus both skip paths were re-tested against them**
> (17 further calls). Sections 2–4 report the canonical-prompt numbers; §5's
> findings were made under the earlier prompt and are called out where that
> matters.

---

## 1. What was run

Hand-filled examples covering **all ten template categories** (both stages),
plus deliberate `insufficient_data` cases to exercise the SKIP path. Run through
`deepseek_client.py` itself, not a bespoke script, so the checkpointing, parsing,
retry, and costing paths were all exercised.

| Phase | Calls | Purpose |
|---|---|---|
| Initial Stage 1 run | 5 | format / SKIP / usage baseline |
| Thinking-mode comparison | 6 | measure reasoning-token overhead |
| Final Stage 1 (hardened client) | 5 | confirm fixes |
| Final Stage 2 + interaction | 5 | remaining categories |
| Resume check | 0 | verify no re-billing on restart |
| **Canonical prompts — cold pass** | 7 | both stages + both skip paths |
| **Canonical prompts — warm pass** | 7 | measure warm-cache hit rate |
| **Canonical prompts — remaining categories** | 3 | complete the 10-category table |

Total spend across all phases: **under $0.003.**

---

## 2. Response format — matches expectations

All calls returned the canonical prompts' output contract, parsed cleanly by
`parse_completion()`:

```
Question: <paraphrased question>
Answer: <reworded answer>
```

Note the canonical labels are `Question:`/`Answer:`, not the ALL-CAPS form the
earlier draft prompt used. The parser matches case-insensitively, so it accepted
both without modification.

Fidelity was the thing to check hardest, and it held on every call. A
representative Stage 1 result:

> **GT question:** Find the genes whose expression values exceed 10.0.
> **GT answer:** The genes with expression above 10.0 are MYH7 (12.31), ACTC1 (11.87), TNNT2 (11.42), DES (10.95), TPM1 (10.61), MYL2 (10.44), ATP2A2 (10.29), and NPPB (10.18) (log2-CPM).
>
> **Generated question:** What are the genes whose expression levels exceed 10.0?
> **Generated answer:** The genes with expression values above 10.0 are MYH7 (12.31), ACTC1 (11.87), TNNT2 (11.42), DES (10.95), TPM1 (10.61), MYL2 (10.44), ATP2A2 (10.29), and NPPB (10.18) (log2-CPM).

Across every category: **no gene symbol altered, no value rounded or reordered,
no invented biological interpretation, no hedging added.** Multi-gene lists came
back complete and in order. Question paraphrases were genuinely varied rather
than near-copies — a good sign for the Step 6 dedup check, though 37 calls proves
nothing about diversity at 250K scale.

The Stage 2 restrictions were also observed to hold: `gene_driver_reasoning`
output stayed at the broad cardiovascular-disease level with no per-patient or
per-subtype attribution, as that prompt's restriction requires.

**SKIP handling works, and each stage produced its own required skip string.**
Both skip inputs returned a bare `SKIP:` line, were recorded with
`status: "skip"`, and were never retried:

```
C4__stage1_skip  skip  in=832  out=6   SKIP: no answer provided
C7__stage2_skip  skip  in=1091 out=5   SKIP: insufficient_data
```

Stage 1 was given `(none provided)` and Stage 2 `insufficient_data` — matching
the two prompts' differing conventions, both of which the client treats as
missing ground truth.

---

## 3. Token usage — measured against the canonical prompts

| Stage | Category | Input | Output |
|---|---|---|---|
| 1 | `direct_abundance_query` | 835 | 35 |
| 1 | `comparative_query` | 860 | 63 |
| 1 | `ranking_ordering_query` | 875 | 75 |
| 1 | `interaction_network_query` | 891 | 77 |
| 1 | `threshold_query` | 896 | 101 |
| 2 | `disease_subtype_classification` | 1105 | 28 |
| 2 | `gene_driver_reasoning` | 1143 | 70 |
| 2 | `comparative_differential_reasoning` | 1154 | 75 |

The pre-flight estimate (DeepSeek's 0.3 tokens/character heuristic) predicted
Stage 1 input at 578/call against **519.8 measured** — 10% conservative. Output
Averages: Stage 1 **871.4 in / 70.2 out**, Stage 2 **1134.0 in / 57.7 out**. The
canonical prompts are ~1.7x longer than the earlier draft (they carry worked
examples), so input roughly doubled — but cost *fell*, because the extra length
is all cacheable prefix. `deepseek_cost_estimate.md` is built on these numbers.

**Caching confirmed working, and it survives the single-user-message layout.**
This was the main risk in adopting the canonical prompts: they are self-contained
documents with the per-call `Given:` block at the end, so the client sends one
user message rather than a system/user split. Prefix caching still hits, because
everything above `Given:` is byte-identical:

| | Warm `prompt_cache_hit_tokens` | % of prompt |
|---|---|---|
| Stage 1 | 768 | 88.1% |
| Stage 2 | 1024 | 90.3% |

All observed hit values were multiples of 64. **Warming is not instant:** a cold
pass of 7 calls hit 13.2% cumulative; an immediate second pass of the same 7 hit
**90.7%**. A 250K-call run amortizes that away entirely, but a short pilot will
understate the cache benefit — don't extrapolate cost from the first 50 calls.

Raw usage object from one call, for reference:

```json
{"prompt_tokens": 496, "completion_tokens": 104, "total_tokens": 600,
 "prompt_tokens_details": {"cached_tokens": 384},
 "completion_tokens_details": {"reasoning_tokens": 69},
 "prompt_cache_hit_tokens": 384, "prompt_cache_miss_tokens": 112}
```

(That sample is from the earlier draft-prompt run — kept because it is the one
call where `reasoning_tokens` was inspected directly. Prefix sizes there are the
old prompt's, not the canonical one's.)

---

## 4. Latency — supports the batching strategy

| Config | Per-call latency | Notes |
|---|---|---|
| Thinking disabled (shipped default) | **1.06 – 3.84 s**, ~2.5 s mean | scales with output length |
| Thinking enabled (API default) | 2.1 – 4.2 s | ~2x slower |

The longer canonical prompts did not measurably change latency — it tracks output
length, and output is unchanged.

At 2.5 s mean and the client's default concurrency of 32, that projects to ~13
calls/s and **~5.4 hours for the full 250K-call job** — comfortable for an
overnight run, with room to raise concurrency (the published ceiling is 2500;
32 is 1.3% of it). No 429s, no timeouts, no retries at any tested concurrency.

The keep-alive behaviour documented for queued requests never triggered at this
scale, so the 180 s read timeout is untested under real load. It only matters if
the account hits queueing during the full run.

---

## 5. Two problems found and fixed

**(a) SKIP could have leaked a fabricated pair into the corpus.** On one Stage 2
`insufficient_data` case the model wrapped the skip inside the normal format:

```
QUESTION: Which cardiovascular subtype does this molecular expression profile correspond to?
ANSWER: SKIP: The provided answer is insufficient_data.
```

The original parser would have accepted that as a valid QA pair with the literal
answer `"SKIP: The provided answer is insufficient_data."` — exactly the failure
mode the SKIP rule exists to prevent. It was fixed on both sides: a prompt
instruction that `SKIP:` must be the entire response, plus a parser guard.

**Only the parser guard survives, and that makes it load-bearing.** Adopting the
canonical prompts verbatim removed the prompt-side half — the canonical files do
not contain that instruction, and they are authoritative, so it was not
reinstated. `parse_completion()` therefore remains the only thing standing
between a wrapped skip and the corpus. It takes the ground truth as a guard:
when GT is missing (`insufficient_data`, `(none provided)`, or empty), *any*
non-skip response is recorded as a skip with a `parse_anomaly` note rather than
accepted, and is never retried. This is client-side defensive parsing, not prompt
text, so keeping it does not conflict with the prompts being authoritative.

Re-tested under the canonical prompts: both skip cases returned a bare `SKIP:`
line with no anomaly. Edge cases (skip inside an `Answer:` line, paraphrase
returned despite missing GT, malformed output, empty completion) are covered by
direct unit checks of `parse_completion()`. **Worth watching in the pilot batch:**
if `parse_anomaly` shows up at any material rate, the canonical Stage 2 prompt
may need a hardening line — but that is the prompt owner's call, not the client's.

**(b) Thinking mode was doubling the bill.** `deepseek-v4-flash` enables thinking
by default at `high` effort, and reasoning tokens bill as output. On identical
inputs: **837 output tokens with it on vs 318 with it off — 2.63x** — for a
verbatim-preserving rewrite where the reasoning traces were just restatements of
the instructions. Fidelity was identical either way. Projected to full scale that
is **$18.63 vs $8.98**. The client now sends `"thinking": {"type": "disabled"}`
by default (`--thinking` re-enables it).

---

## 6. Also verified

* **Incremental checkpointing** — each result lands in the output JSONL as it
  arrives, flushed per record, `fsync` every 25.
* **Resume** — re-running the same command against an existing checkpoint
  reported `resume: 5 already complete, 0 pending` and made zero API calls. Only
  `ok` and `skip` count as complete, so errored items are retried on resume.
* **Cost accounting** — per-call and cumulative usage/cost logged and written to
  a summary JSON; the reported $0.00029 for 5 calls matches hand-computed cost
  from the published rates.
* **Key hygiene** — the key is read only from `DEEPSEEK_API_KEY`, never written
  to the checkpoint, the logs, or the usage summary; `redact()` scrubs it from
  any error text before logging. Confirmed by inspecting all test output files.

---

## 7. Not covered by this test

* Behaviour at production concurrency (32+) — untested; 429 backoff and the
  keep-alive path never triggered at concurrency 4.
* Paraphrase **diversity** at scale — 20 calls says nothing about whether
  DeepSeek collapses to repetitive phrasing across 250K generations. That is
  Step 6's dedup check.
* Sustained cache hit rate over hours — measured warm, not sustained.
* The recommended **100–500 call pilot** from `qa_generation_pipeline_todo.md`
  Step 4 still stands. This test proves the plumbing; the pilot proves the
  economics and output quality at scale, and should run once Step 3 has produced
  real `filled_pairs_*.jsonl`.
