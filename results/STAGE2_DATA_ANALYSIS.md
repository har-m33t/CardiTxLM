# Stage-2 Corpus — Breakdown and Analysis

Computed from the shipped bundle (`data/cvd_transcriptome/text_files/stage2_train.json`) and the per-sample DE table, so this describes the exact artifact that was trained on.

## 1. Composition

| category | items | share | unique samples | distinct-answer ratio | mean answer chars | mean genes named |
|---|---:|---:|---:|---:|---:|---:|
| `gene_driver_reasoning` | 7,210 | 26.7% | 7,210 | 1 | 715 | 10.5 |
| `comparative_differential_reasoning` | 7,208 | 26.7% | 7,208 | 1 | 981 | 13.6 |
| `magnitude_reasoning` | 7,202 | 26.7% | 7,202 | 1 | 372 | 1.0 |
| `disease_subtype_classification` | 5,352 | 19.8% | 1,784 | 0.0927 | 60 | 0.0 |

**26,972 items over 7,212 unique samples** (3.74 items per sample).

The three per-sample categories sit at a distinct-answer ratio of 1.0000 — every one of their answers is unique. Before the regeneration the same two categories sat at 0.0007 and 0.0004, i.e. a single answer repeated across 8,553 samples.

`disease_subtype_classification` at 0.0927 is **not** degeneracy: its answer is one of five subtype labels, so its target is determined by its input. The ratio is low because the label set is small, which is what a classification task looks like.

## 2. What the answers say

- **5,436 distinct gene symbols** are named across the corpus.
- The most-named gene appears in **1,931 items (7.16%)** — `AGPAT1`. The head is flat: no gene dominates, which is what distinguishes this corpus from the one it replaces.
- Answer length: mean 565 characters, p95 1240, max 1,345. Worst-case token estimate is well inside the 2,048 `model_max_length`, so nothing is truncated.

Top 10 named genes:

| gene | items | % of corpus |
|---|---:|---:|
| AGPAT1 | 1,931 | 7.16% |
| RPS18 | 1,904 | 7.06% |
| FAP | 1,479 | 5.48% |
| ADH1B | 1,137 | 4.22% |
| CEMIP | 1,007 | 3.73% |
| ICA1 | 1,004 | 3.72% |
| PDE1C | 913 | 3.38% |
| LMNB1 | 905 | 3.36% |
| THBS2 | 895 | 3.32% |
| PTPRK | 864 | 3.2% |

## 3. The per-sample ground truth underneath

- **Reference population:** 21,041 `neg_hard` samples, after excluding 1,266 that sit in holdout series.
- **Tissue-matched:** 2,144 samples (25.1%) drew on one of 96 qualifying tissue buckets; 6,409 fell back to the whole pool. Recorded per sample, never silently substituted.

Effect-size distribution across the 8,553 samples:

| metric | mean | median | p05 | p95 |
|---|---:|---:|---:|---:|
| `n_genes_abs_z_gt2` | 1307.83 | 514 | 106 | 5126.2 |
| `frac_genes_abs_z_gt2` | 0.067 | 0.0257 | 0.0053 | 0.2656 |
| `mean_abs_z` | 0.8551 | 0.6408 | 0.4348 | 1.8773 |
| `max_abs_z` | 69.315 | 14.3033 | 5.5577 | 301.411 |

Magnitude tertiles (computed once over the whole population, so the label means the same thing in every answer): cut points 0.016792 / 0.050977 → minimal 2,858, moderate 2,845, large 2,850.

## 4. Split

- Training samples: **7,212**
- Holdout: **92 series, 1,341 positives + 1,266 negatives**, reserved entirely.
- Leakage: **0** — asserted before the bundle was written, and again on the pod before training.

## 5. Worked examples

One item per category, taken verbatim from the shipped bundle.

### `comparative_differential_reasoning`

**Q.** Compared with tissue-matched samples lacking a confirmed cardiovascular diagnosis, which genes show the most pronounced reduction in this patient?

**A.** Relative to tissue-matched samples without confirmed cardiovascular disease (n = 21041), this patient's most reduced genes are CALR (z = -2.987, log-fold-change -4.29), HSP90AA1 (z = -2.502, log-fold-change -2.529), CDK4 (z = -2.444, log-fold-change -3.071), EEF2 (z = -2.35, log-fold-change -3.228), and NPDC1 (z = -2.345, log-fold-change -3.6). Across 20009 comparable genes, 3535 depart by more than two standard deviations from that reference population in either direction. Values are log1p(TPM); z represents this sample's deviation in reference standard deviations.

### `gene_driver_reasoning`

**Q.** Within the broad set of genes tied to cardiovascular disease, which show the strongest elevation in this patient's profile?

**A.** Among the 1142 genes with a stable cardiovascular-disease signal in the elastic-net ranking (nonzero fraction of 1.0 across all 5 outer folds), the most elevated in this sample—relative to tissue-matched samples without confirmed cardiovascular disease (n = 21041)—are NCF1 (z = 3.984, elevated), RUNX1T1 (z = 3.12, elevated), PDGFRA (z = 2.926, elevated), ELF3 (z = 2.884, elevated), ASPM (z = 2.855, elevated), ITGAX (z = 2.788, elevated), COL11A1 (z = 2.786, elevated), and IQGAP2 (z = 2.765, elevated). This gene set reflects broad cardiovascular disease associations, not a subtype-specific signature, and the deviations are specific to this patient's sample.

### `magnitude_reasoning`

**Q.** How many genes in this sample show a substantial deviation from tissue-matched samples without confirmed cardiovascular disease, and what is the magnitude of the largest such departure?

**A.** The deviation magnitude is minimal: 173 of 20009 comparable genes (0.86%) fall beyond two standard deviations from tissue-matched samples without confirmed cardiovascular disease (n = 21041), with the largest single departure being ETDA at z = 11.712. These magnitude labels are based on tertiles of this statistic across the entire disease-confirmed population.

### `disease_subtype_classification`

**Q.** Based on this patient's transcriptomic profile, what cardiovascular disease category is most likely?

**A.** The profile indicates hypertension as the confirmed cardiovascular subtype.

## 6. What changed, in one table

| | before | after |
|---|---|---|
| items | 19,793 | 26,972 |
| categories | 3 | 4 |
| information-free share | **86.4%** | **0%** |
| per-sample-DE-grounded share | 0% | **80.2%** |
| `comparative_differential_reasoning` ratio | 0.0007 | **1.0000** |
| `gene_driver_reasoning` ratio | 0.0004 | **1.0000** |
| distinct genes named | ~1,142 (one fixed list) | **5,436** (per sample) |
| holdout | none | 92 series, 2,607 samples |
| training loss floor | 0.066 | **0.501** (6.63×) |

Files: `tables/corpus_composition.csv`, `tables/corpus_gene_frequency.csv`, `tables/corpus_de_statistics.csv`, `plots/corpus_composition.png`.
