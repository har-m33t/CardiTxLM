# GT Computation Functions — Step 1 Report

Deterministic ground-truth layer for QA generation: `qa_generation/gt_functions.py`,
one function per verified template category. No LLM is involved. These produce the
facts Step 4's paraphrasing model rewords; anything not returned here is not
available to be said.

- **Module:** `qa_generation/gt_functions.py`
- **Tests:** `qa_generation/tests/test_gt_functions.py` — 66 tests, all passing
- **Support script:** `qa_generation/build_coexpression_edges.py` (see §1)

Every function returns a `GTResult` with `status` of `ok` or `insufficient_data`.
A missing *input file* raises `MissingArtifactError` instead — a broken install
must never be reported as a data-coverage gap, since Step 2 filters on
`insufficient_data` and would silently drop the entire corpus.

---

## 1. One prerequisite had to be built: the co-expression edge list

**The premise that Track 4's edge list was saved is false.** Track 4
(`gene_pool_prep/compute_centrality_genes.py`) computed the correlations but
persisted only the in-degree *counts*. At `compute_centrality_genes.py:182-183`:

```python
top = np.argpartition(-C, TOP_K, axis=1)[:, :TOP_K]
in_degree += np.bincount(top.ravel(), minlength=n_genes)
```

`top` is a local, folded into a histogram and dropped. The only persisted output
is `high_centrality_genes.csv` (`gene, in_degree, centrality_rank, selected`) —
how many genes name each gene as a partner, with no record of *which*. No
correlation matrix or edge list exists anywhere in the repo. Interaction Network
Query had nothing to read.

`build_coexpression_edges.py` recovers it, inheriting Track 4's methodology
rather than introducing a narrower one:

| | Track 4 | This script |
|---|---|---|
| Source matrix | `cvd_only_expression.npy` (8,553 × 49,231) | same |
| Standardization | z-scored columns ÷ √(n−1), zero-variance zeroed | same |
| **Partner universe** | **all 49,231 QC columns** | **all 49,231 QC columns** |
| Top-k | 100 | 100 |
| Rows computed | all 49,231 columns (in-degree needs every edge) | 7,166 pool genes only (the only genes QA queries) |

Two column-level details Track 4 never had to resolve, because in-degree is a
column-level statistic while QA is symbol-level:

- **Duplicate symbol columns.** A pool symbol carried by k > 1 columns contributes
  all k as source rows; their partner sets merge, keeping the highest r per
  partner symbol. 7,166 pool genes → 8,113 source columns. This is the union of
  Track 4's edges leaving that symbol's nodes.
- **Self-symbol partners are dropped.** Track 4 kept them but flagged them as an
  artifact (`centrality_manifest.json` → `n_same_symbol_edges`: 6,470 edges among
  byte-identical duplicate columns, guaranteed to be each other's top partner).
  "TTC34 is co-expressed with TTC34" is not a usable QA fact.

**Output:** `qa_generation/coexpression/coexpression_edges.parquet` — 716,600 edges
(7,166 genes × 100 partners), 53 s, plus `coexpression_manifest.json`. Verification:
all genes have exactly 100 ranked partners, r monotone non-increasing with rank,
no self-symbol edges, |r| ≤ 1.001, and 25 randomly sampled edges recomputed with
`np.corrcoef` straight off the raw matrix — max absolute error **7.7 × 10⁻⁷**.

Face validity: MYH7's top partners are CSRP3, CKM, LMOD2, TRIM63, SMYD1 (r ≈ 0.95),
all cardiac sarcomere/muscle genes. TTN's are MYLK3, TECRL, CMYA5.

---

## 2. Conventions shared by all functions

**Sample id** is the GEO accession (`GSM1126620`), matching the `.npy` filenames
`integration/build_dataset_json.py:87-95` writes and therefore the `image` field
of the training JSON. A global `sample_index` int is also accepted and resolved
to its accession.

**Expression values** are `log2(count + 1)` on ARCHS4 gene-level counts, read from
`cvd_only_expression.npy`, rounded to 4 dp.

> **Flag for Step 3/4.** The model's input `.npy` is a **20,010-gene BulkFormer-vocab
> vector under TPM → log1p** (`integration/build_dataset_json.py:6-8`), a different
> normalization and a different gene universe from the GT values here. Numeric
> answers ("MYH7 = 19.2641") are true of the project's expression matrix, not of
> the tensor the model sees. This does not affect ordering-based answers (ranking,
> comparison, threshold direction), but whoever owns Step 3 should decide
> deliberately whether absolute values belong in the answer text.

**Duplicate symbol columns.** 2,691 of the 49,231 QC columns share a symbol with
another; 468 curated-pool genes are affected, up to 10 columns each. A symbol's
value is the **mean across its columns** — standard handling for redundant probes,
and applied identically in every function so lookup, threshold, ranking and
comparison never contradict each other. `n_matrix_columns` reports the count.

**Gene universe for threshold and ranking** is the 7,166-gene curated pool, not all
49,231 QC columns. The pool is the documented gene universe QA operates over, and
an answer enumerating tens of thousands of genes is not a usable training target.

**Eligible populations differ by category and do not nest.** Step 2 must intersect
them explicitly:

| Category | Gate | Eligible | ∩ has expression row |
|---|---|---|---|
| All Stage 1 | has row in `cvd_only_expression.npy` | 8,553 | 8,553 |
| `disease_subtype_classification` | `is_cvd_disease` ∧ subtype resolved | 2,942 | **2,689** |
| `comparative_differential_reasoning` | probe positive | 8,725 | 8,553 |
| `gene_driver_reasoning` | `is_cvd_disease` | 10,557 | 8,553 |

Resolved-subtype samples with an expression row, by class: coronary_artery_disease
943, heart_failure 804, hypertension 679, cardiomyopathy_other 141, arrhythmia_afib
122. The 5,864 `disease_matched_subtype_unresolved` samples in the matrix are
Stage-1-eligible but yield `insufficient_data` for subtype classification — as
intended.

---

## 3. Function reference

Worked examples below are **real output**, sample `GSM1126620` (heart failure,
GSE46224) unless noted. Long lists are truncated with `…`.

### 1. `direct_abundance_query(sample_id, gene)`

**Sources:** `cvd_only_expression.npy`, `cvd_only_sample_index.npy`,
`gene_symbols.npy`, `sample_labels.parquet`, `curated_gene_pool.csv`.

**insufficient_data when:** `unknown_sample_id` · `sample_not_in_expression_matrix`
(the sample has no row in the 8,553-sample CVD matrix) · `gene_not_in_expression_matrix`.

```python
>>> direct_abundance_query("GSM1126620", "MYH7")
{"gene": "MYH7", "expression": 19.2641, "units": "log2(count + 1)",
 "n_matrix_columns": 1, "in_curated_pool": true}
```

Any gene present in the matrix is accepted, not only pool genes; `in_curated_pool`
records which.

---

### 2. `threshold_query(sample_id, threshold, direction="above")`

**Sources:** as above. **Universe:** 7,166 pool genes.

**insufficient_data when:** `unknown_sample_id` · `sample_not_in_expression_matrix` ·
`unrecognized_direction:<value>` · `threshold_not_numeric` · `threshold_not_finite`.

Comparison is **strict** (`>` / `<`). Results are sorted most-extreme-first.
`degenerate: true` marks answers matching 0 or all 7,166 genes — true, but useless
as training items; Step 2 should drop rather than reword them.

```python
>>> threshold_query("GSM1126620", 12.0, "above")
{"threshold": 12.0, "direction": "above", "comparison": "strictly_greater",
 "n_universe": 7166, "n_matching": 314,
 "genes": ["MT-CO1", "MYH7", "MT-ND4", "MT-ND5", "MT-RNR2", …],
 "expression": [20.8896, 19.2641, 18.9227, 18.8816, 18.8298, …],
 "degenerate": false}
```

---

### 3. `ranking_query(sample_id, n_or_percentile, direction="top")`

Registered as `ranking_ordering_query`. **Sources:** as above.

`n_or_percentile` accepts an int count, a `"10%"` string, or a fraction in (0, 1);
percentiles resolve as `ceil(pct/100 × 7166)`.

**insufficient_data when:** `unknown_sample_id` · `sample_not_in_expression_matrix` ·
`unrecognized_direction:<value>` · `unusable_size_spec:<value>` (non-numeric, ≤ 0,
or larger than the universe).

Ties break on gene symbol ascending, so the answer is reproducible. **`boundary_tie`
flags the cut falling inside a run of equal values** — "the top 10" is then genuinely
ambiguous and Step 2 should drop those items. It fires often at the bottom end,
where hundreds of pool genes read exactly 0.0.

```python
>>> ranking_query("GSM1126620", 5, "top")
{"direction": "top", "mode": "count", "n": 5, "percentile": null, "n_universe": 7166,
 "genes": ["MT-CO1", "MYH7", "MT-ND4", "MT-ND5", "MT-RNR2"],
 "expression": [20.8896, 19.2641, 18.9227, 18.8816, 18.8298],
 "tie_break": "gene_symbol_ascending", "boundary_tie": false}

>>> ranking_query("GSM1126620", "1%", "top")["n"]      # ceil(0.01 × 7166)
72
```

---

### 4. `comparative_query(sample_id, gene_a, gene_b)`

**Sources:** as above.

**insufficient_data when:** `unknown_sample_id` · `sample_not_in_expression_matrix` ·
`gene_not_in_expression_matrix:<names>` (names the offending gene(s)).

Since values are already log2, their difference *is* the log2 fold change.
`equal: true` with `higher: null` when the two are identical.

```python
>>> comparative_query("GSM1126620", "MYH7", "TTN")
{"gene_a": "MYH7", "gene_b": "TTN", "expression_a": 19.2641, "expression_b": 16.9385,
 "higher": "MYH7", "lower": "TTN", "equal": false,
 "difference": 2.3256, "log2_fold_change_a_vs_b": 2.3256}
```

---

### 5. `interaction_network_query(sample_id, gene, n=None)`

**Sources:** `coexpression_edges.parquet` (§1) for partner selection;
`cvd_only_expression.npy` for the values. Correlations are **not** recomputed at
query time.

**insufficient_data when:** `unknown_sample_id` · `sample_not_in_expression_matrix` ·
`gene_not_in_coexpression_edges` (the gene is outside the curated pool — no partner
set is invented for it) · `unusable_partner_count:<value>`.

Defaults to all 100 partners. The reported fact is each partner's **expression in
this sample**; `pearson_r` is carried as provenance, and per `stage1.yaml` the
correlation strength itself is not the answer.

```python
>>> interaction_network_query("GSM1126620", "MYH7", n=3)
{"gene": "MYH7", "gene_expression": 19.2641, "n_partners": 3,
 "partners": [{"gene": "CSRP3", "rank": 1, "expression": 14.4501, "pearson_r": 0.9591},
              {"gene": "CKM",   "rank": 2, "expression": 15.3529, "pearson_r": 0.9573},
              {"gene": "LMOD2", "rank": 3, "expression": 14.3239, "pearson_r": 0.957}],
 "partner_source": "coexpression_edges.parquet (Track 4 methodology)",
 "correlation_population": "8,553 disease-confirmed CVD samples"}
```

Partners are drawn from all 49,231 QC columns, so a partner may sit outside the
curated pool; its value still comes from the same matrix.

---

### 6. `disease_subtype_classification(sample_id)`

**Sources:** `extended_eda_out/labels/sample_labels.parquet`.

**insufficient_data when:** `unknown_sample_id` · `not_disease_confirmed`
(`is_cvd_disease` is false — covers both the tissue-only-unconfirmed pool and
non-CVD samples, per the standing rule that tissue-only must never be treated as
disease-positive) · `subtype_unresolved:<label>` (the
`disease_matched_subtype_unresolved` bucket).

```python
>>> disease_subtype_classification("GSM1126620")
{"subtype": "heart_failure", "disease_category": "cardiovascular",
 "series_id": "GSE46224", "n_disease_categories_matched": 1,
 "label_source": "extended_eda_out/labels/sample_labels.parquet",
 "in_expression_matrix": true}

>>> disease_subtype_classification("GSM1201748")
status=insufficient_data  reason="subtype_unresolved:disease_matched_subtype_unresolved"
```

---

### 7. `comparative_differential_reasoning(sample_id, comparison_group)`

**Sources:** `linear_probe/probe_sample_labels.parquet`,
`linear_probe/results/*/*/probe_results.json`, `sample_labels.parquet`.

**Only `neg_hard` is answerable.** `"tissue_only_disease_unconfirmed"` is accepted
as its alias, since `stage2.yaml` names the pool that way and the probe names it
`neg_hard`. Every other binding — including `neg_whole_corpus` — returns
`comparison_group_not_permitted:<value>` with an empty payload. Nothing is
substituted.

**insufficient_data when:** `unknown_sample_id` · `comparison_group_not_permitted:<value>` ·
`sample_not_in_probe_labels` · `sample_not_a_probe_positive` ·
`neg_hard_probe_results_missing`.

```python
>>> comparative_differential_reasoning("GSM1126620", "neg_hard")
{"sample_subtype": "heart_failure", "comparison_group": "neg_hard",
 "comparison_group_definition": "tissue_only_disease_unconfirmed — CVD-relevant tissue,
   bulk, with no disease confirmation in the metadata",
 "n_positive": 8725, "n_comparison": 22307, "n_series": 1654,
 "separability": {"metric": "linear probe ROC-AUC, grouped 5-fold by series",
                  "primary_variant": "BulkFormer-37M",
                  "roc_auc_mean": 0.7806, "roc_auc_std": 0.1055,
                  "by_variant": {"BulkFormer-37M": 0.7806, "BulkFormer-50M": 0.7236}},
 "confound_context": {"random_tissue_roc_auc": 0.9247, "note": "…not valid ground truth…"},
 "per_gene_differential": null,
 "per_gene_differential_reason": "no expression matrix exists for the neg_hard pool,
   so no per-gene contrast has been computed at this comparison"}

>>> comparative_differential_reasoning("GSM1126620", "neg_whole_corpus")
status=insufficient_data  reason="comparison_group_not_permitted:neg_whole_corpus"
```

The 0.9247 vs 0.7806 pair is read live from the probe results and reproduces the
documented 0.925 → 0.781 confound.

> **Scope limitation — needs a Step 3 decision.** This payload carries *corpus-level
> separability*, not per-gene differential expression. `cvd_only_expression.npy`
> covers disease-confirmed samples only; no expression matrix exists for the 22,307
> `neg_hard` samples, so no per-gene contrast has ever been computed at this
> comparison, and the elastic-net ranking cannot stand in — it was trained against
> the *confounded* random-tissue negatives. `per_gene_differential` is therefore
> present and explicitly `null` rather than absent, so Step 4 cannot read silence as
> licence to invent one. The honest answer to "which genes distinguish X from Y" at
> this comparison is currently *"a measurable signal exists (ROC-AUC 0.78 ± 0.11),
> per-gene attribution not established."* If that is too thin to train on, the fix is
> a differential-expression run against the `neg_hard` pool — new work, out of scope
> here — not a looser GT function.

---

### 8. `gene_driver_reasoning(sample_id, top_n=None)`

**Sources:** `elasticnet_out/gene_signal/gene_signal_ranking.csv`,
`sample_labels.parquet`.

**Broad CVD only. There is no subtype-conditioned variant and the function takes no
`condition` argument** — a test asserts its signature is exactly
`{sample_id, top_n}`. The ranking comes from one global CVD-vs-random-bulk-tissue
elastic-net model; no per-subtype ranking exists anywhere in the project
(`gene_pool_prep/elastic_net_ranking_audit.md`).

**insufficient_data when:** `unknown_sample_id` · `not_disease_confirmed` ·
`unusable_top_n:<value>`.

Returns only the cross-fold-stable subset: `nonzero_frac == 1.0`, a nonzero
coefficient in all five outer folds. That is **1,291 ranking rows → 1,234 unique
genes** after collapsing symbols spanning multiple columns (keeping each symbol's
strongest coefficient). The other ~96% of the 49,231-row ranking is zero-coefficient
noise and is never sampled.

Every disease-confirmed CVD sample gets the **same** list — this is a corpus-level
fact, not a per-sample computation, and `sample_role: "eligibility_gate_only"` says
so in the payload. A test asserts four samples of four different subtypes return
byte-identical gene lists.

```python
>>> gene_driver_reasoning("GSM1126620", top_n=3)
{"scope": "broad_cardiovascular_disease", "not_subtype_specific": true,
 "n_returned": 3, "n_stable_genes": 1234, "n_stable_rows_before_symbol_dedup": 1291,
 "stability_criterion": "nonzero_frac == 1.0 (all 5 outer folds)",
 "genes": [{"gene": "FCGR2A", "rank": 1, "mean_coef": 0.71075, "direction": "up_in_cvd",
            "in_clingen_hcvd": false},
           {"gene": "GFRA1",  "rank": 2, "mean_coef": -0.44528, "direction": "down_in_cvd",
            "in_clingen_hcvd": false},
           {"gene": "BGN",    "rank": 3, "mean_coef": 0.41336, "direction": "up_in_cvd",
            "in_clingen_hcvd": false}],
 "ranked_by": "abs_mean_coef, descending", "sample_role": "eligibility_gate_only"}
```

The gate is `is_cvd_disease` alone, so this is the one category answerable for
samples with no expression row (10,557 vs 8,553). `in_expression_matrix` is reported
so Step 2 can intersect if it wants a single eligible set.

---

## 4. Tests

`qa_generation/tests/test_gt_functions.py` — 66 tests, all passing (3.1 s).

Expected values are hand-derived **through a different code path than the module**:
raw `np.load` on the matrix indexed by symbol, `pd.read_csv` on the ranking, direct
parquet reads. A bug in `gt_functions` cannot make its own tests pass.
`test_fixtures_still_match_source_data` re-derives the pinned literals at run time,
so regenerating source data fails loudly instead of drifting silently.

Beyond per-function value and `insufficient_data` checks, the suite covers:

- **Cross-function consistency** — `ranking_query` top-1 equals `direct_abundance_query`
  on that gene; every `interaction_network_query` partner value equals its direct lookup.
- **The duplicate-column rule** — CNOT3 (10 columns) returns the mean, and the test
  asserts it differs from column 0.
- **Guardrails** — `gene_driver_reasoning`'s signature exposes no subtype/condition
  parameter; `VALID_COMPARISON_GROUPS` is exactly the two `neg_hard` spellings; five
  other comparison groups are each rejected with an empty payload; all eight functions
  return `insufficient_data` for an unknown sample.
- **Missing artifact ≠ insufficient data** — pointing the edge-list path at a
  nonexistent file raises `MissingArtifactError`.

**Mutation-checked.** Three deliberate bugs were injected to confirm the suite bites,
then reverted (module verified byte-identical afterwards):

| Injected bug | Result |
|---|---|
| Duplicate-column rule mean → first column | 1 failure |
| `nonzero_frac == 1.0` stability filter removed | 1 failure |
| Comparison-group guardrail bypassed | 5 failures |

---

## 5. Notes for Step 2

1. **Intersect the eligibility sets** in §2 — they do not nest. Only 2,689 samples
   satisfy all four gates.
2. **Filter on `degenerate` and `boundary_tie`** before assigning threshold and
   ranking templates. Both mark answers that are true but ambiguous or useless.
3. **`gene_driver_reasoning` is identical for all 10,557 eligible samples.** Sampling
   it per patient produces near-duplicate training items at whatever rate it is
   assigned; weight it accordingly.
4. **Decide the `comparative_differential_reasoning` question in §3.7** before
   budgeting that category — its GT is currently corpus-level only.
5. **Decide whether absolute expression values belong in answer text**, given the
   normalization mismatch flagged in §2.
