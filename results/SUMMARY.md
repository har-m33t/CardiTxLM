# Stage-2 Regeneration & Retrain — Full Summary

**Date:** 2026-08-29 · **Encoder:** BulkFormer-93M (locked) · **Backbone:** Vicuna-7B v1.5 + LoRA
**Hardware:** 4× NVIDIA L40S 46 GB (Runpod, US-MO-1) · **Total spend: $11.62**

---

## 0. The one-paragraph version

Stage-2 training data was broken: 86.4% of its answers were a single fixed
string, so the loss-minimising behaviour was to ignore the expression profile
entirely. That is now fixed — the two degenerate categories carry real
per-sample differential expression, the distinct-answer ratio went from 0.0004
to 1.0000, and the retrained model's loss floor is **6.6× higher** because the
task is genuinely harder. **But fixing the data did not improve the
representation.** On the first genuinely uncontaminated holdout, the LLM's
latents still only tie the frozen encoder they sit on (0.6664 vs 0.6680), and a
dimensionality-matched control confirms that tie is real rather than an
artifact. The data defect was real and is fixed; it was not what was holding the
representation back.

---

## 1. The problem

From `stage2_session_report.md`: the trained multimodal LLM reproduced the
slide's linear-probe result (macro ROC-AUC 0.8709 vs a 0.81 target) but only
*tied* the frozen encoder beneath it (0.8685). Investigating that margin found
the cause:

> **86.4% of Stage-2 training answers carried no sample-specific information.**

Concretely, in `qa_generation/gt_functions.py`:

| category | items | what it returned |
|---|---:|---|
| `comparative_differential_reasoning` | 8,553 | a corpus-level linear-probe ROC-AUC, with `per_gene_differential: None` |
| `gene_driver_reasoning` | 8,553 | the same global elastic-net ranking for every sample, self-documented as `sample_role: "eligibility_gate_only"` |
| `disease_subtype_classification` | 2,687 | a real per-sample subtype label (never the problem) |

17,106 of 19,793 items — 86.4% — had an answer that was identical across every
sample. For that share of the supervision, emitting a fixed string is optimal
and the expression profile is irrelevant.

`comparative_differential_reasoning` was honest about why: no expression matrix
existed for the `neg_hard` comparison pool, so no per-gene contrast had ever
been computed. Fixing the data therefore required computing it.

---

## 2. What was done

### 2.1 Real per-sample differential expression

`qa_generation/build_per_sample_de.py` materialises the `neg_hard` pool's
expression from the 62 GB ARCHS4 H5 and computes, for each of the 8,553
disease-confirmed samples, how **its own** transcriptome deviates from a
tissue-matched comparison population.

- Reference pool: 22,307 `neg_hard` samples → **1,266 excluded** because they
  sit in holdout series → **21,041** used.
  This exclusion matters: without it, holdout information leaks into *training
  answers*, re-contaminating the split the holdout exists to protect.
- Tissue matching: 96 `source_name_ch1` buckets qualified (≥30 samples).
  **2,144** samples got a tissue-matched reference; **6,409** fell back to the
  whole pool. Which was used is recorded per sample and never silently substituted.
- Effect sizes: `z = (x − μ_ref) / σ_ref`, and a log-fold-change (both sides are
  already log1p(TPM), so their difference *is* one).
- Transform reused verbatim from `linear_probe/extract.py::normalize_and_align`
  — not reimplemented, so the manifest's `"reimplemented": false` stays true.

**Four gates** apply to the *named gene lists* only. Population statistics, the
magnitude tertiles and the z/lfc matrices are untouched — verified: the tertile
cut points came out bit-identical (0.016792 / 0.050977) across all four gate
revisions.

| gate | rule | why |
|---|---|---|
| `lfc_gate` | \|lfc\| ≥ 0.5 | A reference σ just above the 1e-6 floor can post \|z\| in the thousands off a trivial fold change. "GENE elevated (z = 10548)" teaches the model to describe noise. |
| `nameable` | real symbol, not an ENSG accession | 625 of 20,010 vocab entries have no symbol (the map fills in the accession). A reader cannot check "ENSG00000269179", so it is not a usable claim. |
| `not_sex_linked` | RPS4Y1, DDX3Y, UTY, USP9Y, KDM5D, EIF1AY, NLGN4Y, ZFY (+XIST/TSIX, not in vocab) | **Measured before acting: a sex-linked gene was the single most deviant gene in 8.9% of samples.** Sex is a covariate here, not the phenotype. |
| `not_cnv_polymorphism` | GSTM1, GSTT1, UGT2B17 | Common whole-gene germline deletions. All three carry a **zero elastic-net coefficient in every fold**, so excluding them removes no CVD signal — evidence-based, not a taste call. |

**LEP was deliberately retained** and is documented as a known confound (most
deviant gene in 2.3% of samples). It tracks adipose content — real biology, not
a genotyping artifact — and excluding it would mean deciding which biology
counts as interesting. Zero-coefficient genes were *not* gated generally; that
would collapse the lists toward the 1,142 CVD genes and destroy the point of an
unbiased comparison.

### 2.2 The clean holdout

92 GEO series contain **both** a probe positive and a `neg_hard` negative:
**1,341 positives + 1,266 negatives**, reserved entirely. Training positives
drop 8,553 → **7,212**.

Mixed series specifically, because a series carrying both classes cannot be
separated on its batch signature — the signature is shared by the positives and
negatives inside it. The previous session's best-looking number (0.9343) was an
artifact of exactly this failure: its 15 held-out series contained **no
negatives at all**, so the classifier was separating studies, not disease. The
both-classes property is asserted at eval time, not assumed.

> The source plan estimated "~20-25 mixed series". The real count is 92. All
> were reserved, which is the plan's clear intent.

### 2.3 Regenerated corpus

New/reworded templates, a new `magnitude_reasoning` category, direction-aware
ranking, and a rewritten DeepSeek prompt (its ban on per-patient phrasing was
**inverted** — correct when answers were corpus-level, wrong once they are not;
the ban on per-*subtype* phrasing stays, since no per-subtype ranking exists).

Generation: **26,988 calls, 0 errors, 89.2% prefix-cache hit, $2.45, 12.9 min.**

**16 records (0.06%) were rejected, not repaired**, for factual drift the
paraphraser introduced — gene-symbol transpositions (NCAPG→NCGAP, RBMY1B→RBM1Y8,
OR4F4→ORF4F) and an altered count (19191→19184). Repairing would mean guessing
which side is right; this is fabricated biology dressed as fact, the exact class
of defect the corpus exists to remove.

### 2.4 Retrain

Stage 1 was **not** retrained (its data was never the problem); its connector
checkpoint was reused unchanged.

---

## 3. Results

### 3.1 The data fix — worked

| category | items | distinct-answer ratio, before → after |
|---|---:|---|
| `comparative_differential_reasoning` | 7,208 | 0.0007 → **1.0000** |
| `gene_driver_reasoning` | 7,210 | 0.0004 → **1.0000** |
| `magnitude_reasoning` (new) | 7,202 | — → **1.0000** |
| `disease_subtype_classification` | 5,352 | 0.0019 → 0.0927 |

**80.2% of the corpus is now grounded in real per-sample differential
expression; it was 0%.**

**5,436 distinct gene symbols** are named across the corpus, with a flat head —
the most-frequent appears in 7.2% of items. The corpus it replaces named one
fixed 1,142-gene list, identically, in every one of its 8,553
`gene_driver_reasoning` answers. Full breakdown in `STAGE2_DATA_ANALYSIS.md`.

`disease_subtype_classification`'s low ratio is **correct and is not
degeneracy**: its answer is one of five subtype labels, so the target is
determined by the input. That is what a classification task looks like, and the
bundle builder exempts it by name for this reason.

Mixture reweighting (plan Phase 1a) was applied by *raising* the subtype share
rather than discarding grounded items: 43.2/43.2/13.6 → **26.7/26.7/26.7/19.8**.

### 3.2 Training — the loss curve is the evidence

`loss_curves/stage2_loss_before_after.png`

| | original | corrected |
|---|---|---|
| steps | 155 | 211 |
| wall clock | 5 m 44 s | 38 m 05 s |
| trainable params | 321,929,216 | **321,929,216** (identical) |
| LoRA config | r=128, α=256 | **r=128, α=256** (identical) |
| loss at step 20 | 0.1016 | **0.7020** |
| final train loss | 0.1664 | **0.5675** |
| loss floor, last 20% | — | **6.63× higher** |

Same architecture, same LoRA config (verified in the saved
`adapter_config.json`), same global batch of 128, same Stage-1 connector. **The
only thing that changed is the data.**

The original collapses from 2.883 to ~0.07 within roughly 20 of its 155 steps
and flatlines — that is what predicting a fixed string looks like. The corrected
run settles near 0.50 and stays there.

**A higher floor is the evidence the fix worked.** A corrected run that
reproduced the old curve would have meant the new data was still degenerate.

### 3.3 Representation quality — did NOT improve

`plots/probe_three_way.png` · `tables/probe_comparison.csv`

**Clean holdout** (2,607 samples, 92 series, grouped 5-fold, `random_state=20260707`):

| feature set | estimator | dim | ROC-AUC |
|---|---|---:|---|
| LLM latent | MLP | 4096 | 0.6823 ± 0.0770 |
| **BulkFormer-93M** | linear | 515 | **0.6680 ± 0.0786** |
| LLM latent | linear | 4096 | 0.6664 ± 0.0795 |
| LLM latent | linear, PCA-matched | 515 | 0.6625 ± 0.0760 |

**This does not close the "ties but doesn't beat the encoder" finding.** The
LLM's linear probe ties the frozen encoder (−0.0016). The MLP's +0.0143 is a
fifth of the fold standard deviation.

The **PCA-matched control** was added specifically because the headline compares
4096-d latents against a 515-d encoder on only 2,607 samples, where the
regulariser rather than the representation could be binding. At matched width
the LLM scores 0.6625 vs the encoder's 0.6680 — **so the tie is real, not a
dimensionality artifact.** This control could have overturned the conclusion; it
did not.

**Pipeline sanity check:** on the contaminated full population the encoder
scores **0.8006** here vs **0.8011** in the previous session — same data, same
method, near-identical. That is what says this pipeline measures what the
previous one measured, so the LLM numbers are trustworthy too.

### 3.4 Broad multi-label probing — before vs after

`plots/multilabel_before_after.png` · `tables/multilabel_before_after.csv`

| label | kind | before | after | delta | folds |
|---|---|---:|---:|---:|---:|
| tissue (28 classes) | scientific | 0.8536 | 0.8577 | +0.0040 | 2/2 |
| disease_category | scientific | 0.7885 | 0.7758 | −0.0127 | 5/5 |
| cvd_subtype | scientific | 0.8494 | 0.8501 | +0.0006 | 5/5 |
| platform | **technical control** | 0.7071 | 0.7054 | −0.0017 | 5/5 |
| instrument | **technical control** | 0.7054 | 0.7016 | −0.0037 | 5/5 |

**No meaningful change in general representation quality.** Every delta is
within noise.

Two reading notes: `platform`/`instrument` are technical controls — a high score
there measures sequencing-batch signal, not representation quality, and is not a
result. And `tissue` scored only **2 of 5 folds** (with 28 classes under
series-grouped CV most validation folds contain a single class), so its delta
rests on far less evidence than the others.

### 3.5 Encoder scale — documented, not acted on (plan Phase 1b)

93M is at the plateau and is the best variant on both negative pools.
neg_whole_corpus: 37M 0.925 → 50M 0.928 → **93M 0.944** → 127M 0.943 → 147M
0.941. The 93M→147M spread is 0.0034, about a third of one fold-mean standard
error. **No 147M retrain is justified.** See `tables/encoder_scale_sweep_note.md`.

---

## 4. Interpretation

The regeneration did exactly what it was designed to do at the data level, and
that is verifiable three independent ways: the distinct-answer ratio, the loss
floor, and the fact that the model can no longer converge in 20 steps.

It did **not** improve the representation. The honest reading is that **the
degenerate supervision was a real defect but was not the binding constraint on
representation quality.** Something else is — candidates, in rough order of how
much they would explain:

1. **The connector is a single linear layer producing ONE token.** The tower
   mean-pools 20,010 genes to one 515-d vector, which a 515→4096 linear map
   projects to a single LLM token. Whatever the LLM can represent about the
   sample is bounded by that one token, and Stage 1 — which trains that
   connector — was not retrained here.
2. **Stage 2 teaches generation, not discrimination.** Nothing in the corpus is
   a discriminative objective against negatives; the model never sees a
   `neg_hard` sample during training.
3. **The absolute ceiling on this task is low.** On the clean holdout even the
   encoder reaches only 0.668. The much higher historical numbers (0.87) came
   from contaminated or batch-confounded populations.

The clearest next experiment is (1): retrain Stage 1's connector — possibly with
more than one output token — rather than further Stage-2 data work.

---

## 5. Corrections and defects found during the run

Recorded because each was silent and each would have produced a plausible,
reportable, wrong number.

| what | impact | resolution |
|---|---|---|
| **`startswith("e0")` column filter** | Silently truncated 4096-d latents to **1000 dims**; every probe computed on a quarter of the representation, no error raised. Not wrong for the 515-d encoder, so it only breaks on the thing under test. This repo had hit it before — `llm_latent_probe_TRUNCATED1000.json` is still in the tree. | One shared reader with two guards; the second compares against the file's own columns, because the truncated set is internally self-consistent and passes any internal check. All affected numbers recomputed. |
| **Logistic regression not converging** at 4096 raw dims over 28 classes | A non-converged fit gives an arbitrary number, not a conservative one. | PCA(256) inside the CV pipeline, fitted on training folds only, applied identically to before and after. |
| **Stale VRAM sizing** in `train_stage2_lora.sh` | Measured on the pre-fix bundle (mean 141 tokens). The regenerated answers are ~1.4× longer, and at `PER_DEVICE_BS=32` peak hit **45.1 GB of 46.1 GB (98%)** — not survivable across 211 variable-length batches. | Re-ran at `PER_DEVICE_BS=16 GRAD_ACCUM=2`, global batch unchanged at 128. Peak 29 GB. Sizing comment corrected. |
| **Direction-blind ranking** | "Identify the genes most **reduced**" received a list ranked by *absolute* deviation — an answer contradicting its own question. | Direction bound per template index; the plan builder refuses to run if any template lacks one. |
| **Read-only numpy mask** | Crashed the probe after latent extraction. | `.copy()` before in-place `&=`. |
| **Four pod bootstrap gaps** | gdown 6 dropped `fuzzy`; vendored BulkFormer source is gitignored; `torch-sparse` *is* required (my earlier comment said otherwise); `<stage1>/language_model` needed reconstructing. | All fixed and committed; bootstrap now runs clean end to end. |

Two shortcuts were **validated rather than assumed**:
- The 515-d encoder cache was derived on CPU from an existing parquet instead of
  a GPU encoder pass. Verified against a live GPU tower forward: **max_abs_diff
  2.83e-06** against a 1e-4 tolerance, with a rolled-vector control at 1.339 —
  five orders of magnitude of separation. This removed an entire GPU stage.
- The fabrication verifier was **negative-controlled**: deliberately corrupting a
  record with an invented gene, a dropped gene, a rounded number and an answered
  skip is caught in all four cases, while the untouched control passes. A check
  that never fails proves nothing.

---

## 6. Cost

Actual billed amounts, not estimates.

| item | |
|---|---|
| DeepSeek generation (26,988 calls, 0 errors) | $2.45 |
| Runpod GPU (4× L40S, pod `qa4v56jkqcjaym`) | $9.12 |
| Runpod disk | $0.06 |
| Local compute (62 GB H5 reads, DE, cache build) | $0 |
| **Total** | **$11.62** |

The pod was **deleted**, not stopped — a stopped pod still bills for its volume.
`list-pods` returns empty.

Two decisions kept this down. The 515-d encoder cache was derived on CPU from an
existing parquet instead of a GPU encoder pass, removing a whole GPU stage. And
the pod only ever needed 19 MB of pre-encoded vectors rather than the 668 MB of
raw ones, because the tower's passthrough takes them by width.

---

## 6b. The retrained model

`checkpoints/stage2-lora-bulkformer-93M-regen/` (615 MB, gitignored):

    adapter_model.safetensors   640 MB   the trained LoRA adapters
    adapter_config.json                  r=128, alpha=256 — verified unchanged
    connector/pytorch_model.bin 4.2 MB   the retrained connector
    trainer_state.json                   full loss history
    config.json, tokenizer files

The frozen `language_model/` the pod also wrote (13.5 GB) was deliberately NOT
retrieved: Stage 2 froze the LLM, so those are the base Vicuna weights unchanged
and `integration/materialize_stage1_llm.py` reconstructs them exactly from
HuggingFace. Only the adapter and connector carry training.

## 7. What is in this folder

```
results/
  SUMMARY.md                     this document
  STAGE2_DATA_ANALYSIS.md        breakdown of the regenerated corpus: composition,
                                 gene coverage, effect-size distributions, and one
                                 worked example per category
  README.md                      generated from artifacts — numbers only, no prose
  MANIFEST.json                  machine-readable index
  loss_curves/
    stage1_loss.png              replotted (Stage 1 was not retrained)
    stage2_loss_before_after.png the headline figure
  plots/
    probe_three_way.png          encoder vs linear vs MLP vs PCA-matched
    multilabel_before_after.png  paired bars, controls greyed
    corpus_composition.png       four panels: composition, answer lengths,
                                 magnitude tertiles, most-named genes
  tables/
    probe_comparison.csv         all probe results, both populations
    probe_three_way.json         raw, including per-fold values
    multilabel_before_after.csv  with fold counts
    multilabel_probe_{before,after}.csv
    encoder_scale_sweep.csv + _note.md
    corpus_composition.csv       per-category item counts, lengths, ratios
    corpus_gene_frequency.csv    all 5,436 named genes and their frequencies
    corpus_de_statistics.csv     effect-size distribution over 8,553 samples
  data/
    de_manifest.json             the full audit record for the DE computation
    stage2_bundle_stats.json     corpus composition and distinct-answer ratios
    holdout_series.json          the 92 reserved series
    stage2_regen_rejected.json   the 16 rejected records and why
    regen_usage_summary.json     DeepSeek cost/usage
    encoded_cache_manifest.json  the CPU-derived cache and its verification
    multilabel_labels_manifest.json
    stage2_regen_plan_stats.json, step3_regen_stats.json
    stage2_train.zip             THE CORPUS ITSELF — all 26,972 items
    stage2_train_SAMPLE_200.json 200 items (50 per category), readable
  logs/
    stage2_regen_trainer_state.json   full loss history, corrected run
    stage2_trainer_state.json         full loss history, original run
    eval_regen.log, pod_bootstrap.log
```

Rebuild with `python3 scripts/build_results_folder.py`. This folder is copied
from canonical locations and is never the source of truth, so it can be deleted
and rebuilt at any time.

---

## 8. Limitations

- **The clean holdout is small and noisy.** 2,607 samples over 92 series, fold
  std ≈ ±0.08. Differences under ~0.02 are inside that noise.
- **Tissue matching covers only a quarter of the corpus** (2,144 of 8,553); the
  rest uses the whole `neg_hard` pool. Recorded per sample.
- **Multi-label absolute numbers describe the top-256 principal subspace**, not
  the full representation — required for convergence, applied identically to
  both sides so the comparison stays fair.
- **`tissue` rests on 2 scorable folds**, not 5.
- **LEP is a retained confound** (2.3% of samples), deliberately.
- **Stage 1 was not retrained**, so any limitation in the connector alignment
  carries forward unchanged — and per §4 that is now the leading suspect.
- **Generative answer quality was not evaluated.** The plan called for it as a
  secondary metric; the embedding-quality result took priority and this was not
  run.
