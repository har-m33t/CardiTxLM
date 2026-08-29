# Stage-2 Regeneration — Results

Generated 2026-08-29T20:54:37+00:00 from the artifacts in this directory. Every number below is read from a file rather than transcribed, so the write-up cannot drift from the run.

## 1. What was fixed

86.4% of the original Stage-2 corpus carried no sample-specific information. `comparative_differential_reasoning` returned a corpus-level probe ROC-AUC with `per_gene_differential: None`, and `gene_driver_reasoning` returned the same global elastic-net ranking for every sample — self-documented as `sample_role: "eligibility_gate_only"`. For that share of the supervision the loss-minimising behaviour is to emit a fixed string and ignore the expression profile entirely.

| category | items | distinct-answer ratio |
|---|---:|---:|
| `gene_driver_reasoning` | 7,210 | 1.0000 |
| `comparative_differential_reasoning` | 7,208 | 1.0000 |
| `magnitude_reasoning` | 7,202 | 1.0000 |
| `disease_subtype_classification` | 5,352 | 0.0927 |

Before the fix those ratios were 0.0007 (`comparative_differential_reasoning`) and 0.0004 (`gene_driver_reasoning`) — one distinct answer across 8,553 samples.

`disease_subtype_classification`'s low ratio is CORRECT and is not degeneracy: its answer is one of five subtype labels, so the target is determined by the input. That is what a classification task looks like. The three per-sample categories are the ones that were broken.

Generation: 26,988 DeepSeek calls, 0 errors, 89.2% prefix-cache hit, $2.45, 12.9 min. 16 records were rejected (not repaired) for paraphraser factual drift.

## 2. How the per-sample ground truth is built

- Reference: 21,041 `neg_hard` samples (1,266 excluded because they sit in holdout series — otherwise holdout information would leak into *training answers*).
- Tissue matching: 2,144 samples got a tissue-matched reference, 6,409 fell back to the whole pool. Which one was used is recorded per sample, never silently substituted.
- Effect sizes: z against the reference mean/sd, plus a log-fold change (both sides are log1p(TPM), so their difference already is one).

Four gates apply to the NAMED gene lists only — population statistics and the z/lfc matrices are untouched:

- **`lfc_gate`** — |lfc| >= 0.5 (log1p(TPM) scale, ~1.65x). Removes near-zero-denominator artifacts. A gene whose reference sd sits just above the 1e-6 floor can post an enormous z off a trivial fold change; 'GENE elevated (z=10548)' is supervision that teaches the model to describe noise.
- **`nameable`** — gene_symbol is a real symbol, not an ENSG accession. symbol_vocab_map falls back to the ENSG accession when a gene has no symbol (625 vocab entries). An accession is not a usable clinical claim — a reader cannot check it — so a gene we cannot name is a gene we should not claim.
- **`not_sex_linked`** — symbol not in ['RPS4Y1', 'DDX3Y', 'UTY', 'USP9Y', 'KDM5D', 'EIF1AY', 'NLGN4Y', 'ZFY', 'XIST', 'TSIX']. Sex is a covariate here, not the phenotype under study. Measured before acting: 8.9% of samples had a sex-linked gene as their single most notable elevated gene. Excluding sex-linked genes from a DE ranking is routine when sex is not the variable of interest.
- **`not_cnv_polymorphism`** — symbol not in ['GSTM1', 'GSTT1', 'UGT2B17']. Inherited copy number, not phenotype. These are common whole-gene germline deletion polymorphisms, and all three have a zero elastic-net coefficient in every fold, so the exclusion is evidence-based rather than a taste call.

## 3. The holdout

92 GEO series contain BOTH a probe positive and a `neg_hard` negative: 1,341 positives and 1,266 negatives, reserved entirely from training.

Mixed series specifically, because a series carrying both classes cannot be separated on its batch signature — the signature is shared by the positives and negatives inside it. The previous session's best-looking number (0.9343) was an artifact of exactly this: its 15 held-out series contained no negatives at all, so the classifier was separating studies, not disease.

## 4. Representation quality

On the clean holdout (grouped 5-fold by series, `random_state=20260707` — identical to every prior probe):

| feature set | estimator | dim | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|
| LLM-latent-imgtok | mlp | 4096 | 0.6823 ± 0.0770 | 0.6721 ± 0.0954 |
| BulkFormer-93M | linear | 515 | 0.6680 ± 0.0786 | 0.6316 ± 0.0952 |
| LLM-latent-imgtok | linear | 4096 | 0.6664 ± 0.0795 | 0.6424 ± 0.0932 |
| LLM-latent-imgtok | linear_pca_matched | 4096 | 0.6625 ± 0.0760 | 0.6379 ± 0.0854 |

### Does this close the "ties but doesn't beat the encoder" finding?

The LLM's linear probe **ties** the frozen encoder it sits on: 0.6664 vs 0.6680 (delta -0.0016).

The MLP probe reaches 0.6823, a gap of +0.0159 over the linear probe on the same embeddings. A large gap would mean the LLM's space holds non-linearly separable structure a linear probe cannot reach; a small one means it does not. Both are reported because the comparison between them is itself the result.

The full-population numbers are also computed, for continuity with the prior session's table. They are CONTAMINATED — most positives were in Stage-2 training — and must not be read as a held-out result:

| feature set | estimator | ROC-AUC |
|---|---|---:|
| BulkFormer-93M | linear | 0.8006 |
| LLM-latent-imgtok | linear_pca_matched | 0.7705 |
| LLM-latent-imgtok | linear | 0.7478 |
| LLM-latent-imgtok | mlp | 0.7151 |

## 5. Broad multi-label probing, before vs after

| label | kind | before | after | delta | folds scored |
|---|---|---:|---:|---:|---:|
| tissue | scientific | 0.8536 | 0.8577 | +0.0040 | 2/2 of 5 |
| disease_category | scientific | 0.7885 | 0.7758 | -0.0127 | 5/5 of 5 |
| cvd_subtype | scientific | 0.8494 | 0.8501 | +0.0006 | 5/5 of 5 |
| platform | technical_control | 0.7071 | 0.7054 | -0.0017 | 5/5 of 5 |
| instrument | technical_control | 0.7054 | 0.7016 | -0.0037 | 5/5 of 5 |

**Read `folds scored` before reading any delta.** Under series-grouped CV a many-class label loses folds whose validation split contains only one class, and a mean over 2 folds is not comparable to a mean over 5. `tissue` (28 classes) is the case to watch.

`platform` and `instrument` are TECHNICAL CONTROLS. A high score there measures sequencing-batch signal, not representation quality, and is not a result.

## 6. Encoder scale (for the record, no action taken)

BulkFormer-93M is at the plateau of the completed 5-variant sweep, which is why no 147M retrain was pursued in this plan. Stated explicitly rather than left as an unstated assumption.

See `tables/encoder_scale_sweep.csv` and its note.

## 7. Loss curves

`loss_curves/stage1_loss.png` — Stage 1 replotted from existing logs; it was not retrained, since its data was never the problem.

`loss_curves/stage2_loss_before_after.png` — the original run collapsed 2.883 → 0.066 within roughly 20 of its 155 steps, which is what predicting a fixed string looks like. **A higher loss floor on the corrected run is evidence the fix worked**, not evidence of a worse model: the task is now genuinely harder. A corrected run that reproduced the old curve would be the alarming outcome.

## 8. Known limitations

- `LEP` is retained as a confound (adipose content) and appears as the most deviant gene in ~2.3% of samples. It is real biology, so excluding it would mean deciding which biology counts; sex-linked genes and common deletion polymorphisms were excluded on the narrower ground that they are covariates and genotyping artifacts rather than phenotype.
- Tissue matching covers only part of the corpus; the rest uses the whole `neg_hard` pool, recorded per sample.
- Stage 1 was not retrained, so any limitation in the connector alignment carries forward unchanged.
- **The clean holdout is small and the folds are noisy.** 2,607 samples over 92 series, with fold standard deviations around ±0.08. Differences below roughly 0.02 between feature sets are inside that noise and should not be read as a ranking.
- **The two feature sets are not matched on dimensionality.** The LLM latents are 4096-d and the encoder 515-d, probed on 2,607 holdout samples. A 4096-d linear probe on that few samples is in a high-dimensional regime where the regulariser, not the representation, can be the binding constraint — so a tie here is weaker evidence *against* the LLM representation than it first appears. Testing that would need a dimensionality-matched comparison (e.g. PCA to 515), which this run did not do.
- The encoder's full-population number reproduces the prior session's to within 0.0005 (0.8006 vs 0.8011) on the same data and method, which is the check that says this pipeline is measuring what the previous one measured.

