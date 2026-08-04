# Step 4 — Full Generation Run

219,747 filled pairs paraphrased through DeepSeek V4-Flash in three checkpoints,
validated at full scale, then converted to the training schema.

| | Result |
|---|---|
| Calls | **219,747** — 199,954 Stage 1 + 19,793 Stage 2 |
| Final status | **219,747 ok / 0 skip / 0 error** |
| **Total cost** | **$19.43** (projection was $21.07 — came in **7.8% under**) |
| Truncated responses | **0** |
| Gene-list completeness | **0 incomplete** across all 128,507 list records |
| Numeric drift (1,000 stratified) | **0** |
| Training output | `stage1_train.json` (199,954), `stage2_train.json` (19,793) |

---

## Checkpoint tracking

| Checkpoint | Calls (cum.) | Cost (cum.) | Extrapolated total | vs $21.07 | Decision |
|---|---|---|---|---|---|
| **1 — 10%** | 21,974 | $1.95 | $19.48 | **−7.5%** | within ±15% → continue |
| **2 — 50%** | 109,873 | $9.73 | $19.42 | **−7.8%** | within ±15% → continue |
| **3 — 100%** | 219,747 | **$19.43** | — | **−7.8%** | complete |

The run tracked the pilot projection closely and never approached the ±15% stop
condition. It came in under projection because the cache-hit rate rose from the
pilot's 69.0% to **78.4%** at full scale — a longer run keeps the 768-token prefix
warm, and cached tokens bill at 1/50th the miss rate.

### Token totals

| | Tokens | Rate | Cost |
|---|---|---|---|
| Input, cache hit | 177,862,144 | $0.0028/M | $0.50 |
| Input, cache miss | 48,699,942 | $0.14/M | $6.82 |
| Output | 43,277,438 | $0.28/M | $12.12 |
| **Total** | **269,839,524** | | **$19.43** |

### Per category

| Category | Calls | Output/call | $/call | Total |
|---|---|---|---|---|
| `ranking_ordering_query` | 39,954 | 481 | 0.0002076 | $8.29 |
| `threshold_query` | 40,000 | 205 | 0.0000924 | $3.70 |
| `interaction_network_query` | 40,000 | 182 | 0.0000844 | $3.38 |
| `comparative_query` | 40,000 | 118 | 0.0000556 | $2.22 |
| `direct_abundance_query` | 40,000 | 44 | 0.0000258 | $1.03 |
| `gene_driver_reasoning` | 8,553 | 109 | 0.0000458 | $0.39 |
| `comparative_differential_reasoning` | 8,553 | 131 | 0.0000437 | $0.37 |
| `disease_subtype_classification` | 2,687 | 29 | 0.0000183 | $0.05 |

`ranking_ordering_query` is 43% of spend on 18% of the calls, entirely from output
volume — the wide bands Step 2 introduced to break mitochondrial dominance.

### Retry cost

| | |
|---|---|
| Calls needing >1 attempt | **2** (0.0009%) |
| Retry-consumed billed cost | **$0.00** |

Failed attempts raise before the API returns a usage object, so only the
successful attempt bills. Retry cost is genuinely nil, not merely small. The
explicitly re-called records (below) are inside the total above.

### SKIP handling

**0 SKIP responses**, and correctly so: Step 3 excluded every rejection at fill
time, so no call was sent with missing ground truth. No SKIP was ever retried —
the client breaks the retry loop on skip by construction.

---

## Issues found and resolved during the run

Three defects surfaced. All were repaired and re-verified; none required
abandoning the run.

### 1. Parser rejected Markdown-decorated labels (18 records at Checkpoint 1)

16 responses used `**Question:**` / `**Answer:**` instead of the plain labels, and
the parser discarded them as malformed. The content was complete and correct.

Fixed `parse_completion` to strip Markdown emphasis and heading decoration from
the label before matching, then **re-parsed the 16 from their stored
`raw_response` — no new API calls**. All skip guards were re-verified intact
(SKIP-first, skip-inside-Answer, missing-GT). Zero parse failures across the
remaining 197,773 calls.

### 2. Truncation at the 3,000-token cap (17 records)

The pilot's observed maximum was 2,705 output tokens, so 3,000 looked safe. At
full corpus scale the tail ran higher: 17 ranking responses (0.04%) hit the cap
and were cut off mid-list.

Raised `MAX_TOKENS_BY_CATEGORY["ranking_ordering_query"]` to **4,096** and re-ran
those 17. Final maximum across all 219,747: **2,999 tokens, 0 at cap.**

This is the pilot's main limitation as a predictor — a 300-call sample does not
reach the extreme tail of a 40,000-call category.

### 3. Incomplete gene lists (60 records)

Two sub-modes, both caught by a corpus-wide completeness check rather than
sampling:

- **57 abbreviations** — the model wrote "continuing through the full list to X"
  instead of enumerating, at 118–181 output tokens (nowhere near the cap). Almost
  all were 290-gene percentile answers.
- **3 single-gene faults**, including one genuine corruption: `threshold_query:41565`
  rendered GT `COL3A1 (9.0752)` as `COL1A3 (9.0752)` — an altered symbol.

Re-ran all 60 at temperature 0.2 (57 fixed), then the remaining 3 at temperature
0.0 (all fixed). **Final: 0 incomplete across all 128,507 list-bearing records.**

Worth carrying forward: the default temperature 0.7 produced these at ~0.03%. If
this corpus is ever regenerated, a lower temperature for the long-list categories
would avoid the repair pass.

---

## Post-completion validation

Full-corpus checks plus a 1,000-item stratified factual sample (the pilot used 300).

| Check | Scope | Result |
|---|---|---|
| Status | all 219,747 | **219,747 ok**, 0 skip, 0 error |
| Unique ids | all | ✓ no duplicates |
| Empty parsed question or answer | all | **0** |
| Responses at their token cap | all | **0** |
| `<image>` handling, schema shape | all | **0 malformed** |
| **Gene-list completeness** | **all 128,507 list records** | **0 incomplete** |
| Numeric fidelity (decimal values) | 1,000 stratified | **0 drift** |
| Gene symbols dropped/invented | 1,000 stratified | **0 / 0** |

One residual, reported rather than repaired: in a small fraction of list answers
the model omits the *count* from the prose ("The genes with the highest expression
are…" rather than "The top 100 genes…") while enumerating the full list correctly.
No fact is altered and the list is complete, so this is a phrasing artifact, not
drift.

---

## Task 6 — Training schema

Produced only after validation came back clean.

| File | Entries | Size | Distinct images | Max per image |
|---|---|---|---|---|
| `stage1_train.json` | **199,954** | 114.7 MB | 8,553 | 25 |
| `stage2_train.json` | **19,793** | 11.2 MB | 8,553 | 3 |

Format verified against the confirmed byte spec:

- Single top-level JSON array (not JSONL) — `dataset.py:29` does `json.load()`
- UTF-8, no trailing commas
- `image` is `<GSM accession>.npy`, all entries
- `conversations` alternates human → gpt
- `<image>
` prefixes every human turn; `<image>` never appears in a gpt turn
- **0 malformed entries** in either file

```json
{"image": "GSM8187400.npy",
 "conversations": [
   {"from": "human", "value": "<image>\nWhat is the normalized expression value for LIMS2 in this sample?"},
   {"from": "gpt", "value": "LIMS2 has an expression value of ..."}
 ]}
```

Multiple pairs per patient share an `image` value, as intended — 8,553 distinct
samples across both files.

---

## Files

| File | Contents |
|---|---|
| `qa_generation/generated_pairs_stage1.jsonl` | 199,954 raw responses with usage/cost |
| `qa_generation/generated_pairs_stage2.jsonl` | 19,793 raw responses |
| `stage1_train.json`, `stage2_train.json` | Training-format output |
| `qa_generation/deepseek_client.py` | Markdown-tolerant parser; ranking cap 3,000 → 4,096 |
| `qa_generation/step4_input_stage{1,2}.jsonl` | Shuffled inputs, for reproducibility |

Note the `.npy` image files themselves are not yet materialised for these 8,553
samples — `integration/build_dataset_json.py` writes them, and that must run
before training reads these JSONs.
