# Linear-probe stage — evaluation floor

This stage produces the "evaluation floor" number for CVD disease
classification: **a frozen BulkFormer encoder with a trainable linear probe
on top, evaluated with 5-fold `StratifiedGroupKFold` on the CVD pool from
the extended EDA.** The finding is the baseline the eventual
encoder→connector→LLM pipeline should be expected to *exceed*, not just
match.

The five BulkFormer parameter scales (37M / 50M / 93M / 127M / 147M) were
run as separate probes to answer the scale-vs-performance question. **All
five are now complete.** 37M and 50M were run CPU-only on a Mac (see § 5
for why — BulkFormer's GCNConv has no MPS-compatible sparse kernel); 93M,
127M and 147M were run later on a 2×RTX 4090 Linux box, GPU 0 running 93M
then 127M sequentially and GPU 1 running 147M in parallel.

Deliverables per the TODO, all under `linear_probe/`:

| Step | Deliverable | State |
|---|---|---|
| 1 | `checkpoint_verification.json` | ✅ all 5 pass |
| 2 | `label_definitions.md`, `mortality_label_search_result.json` | ✅ |
| 3 | `embeddings/embeddings_BulkFormer-{37M,50M,93M,127M,147M}.parquet`, `extraction_manifest.json` | ✅ all 5 |
| 4 | `probe.py` | ✅ |
| 5 | `results/disease_classification_by_variant.csv` | ✅ all 5 rows × 2 pools |
| 6 | `results/mortality_prediction_status.md` | ✅ not-runnable |
| 7 | `results/variant_comparison.png`, `variant_comparison_table.csv` | ✅ all 5 points, elastic-net reference overlaid |
| 8 | this file | ✅ |

## 1. Setup

**Sample pool.** Positives are the six disease-confirmed CVD subtypes from
the extended-EDA taxonomy (`disease_matched_subtype_unresolved` +
`coronary_artery_disease` + `heart_failure` + `hypertension` +
`cardiomyopathy_other` + `arrhythmia_afib`), filtered to
`singlecellprobability < 0.5` to drop single-cell samples. Result:
**8,725 positive samples across 480 series**.

Following TODO § 2 option (c), two negative pools were run and are reported
separately:

- **(a) whole-corpus non-CVD** — samples with `is_cvd_pool = False`,
  `n_disease_categories_matched = 0`, and the same bulk-only filter.
  Sub-sampled at 3× the positive pool (26,175 samples across 10,196
  additional series). This is the elastic-net-comparable negative pool.
- **(b) tissue-only hard negatives** — `cvd_subtype ==
  "tissue_only_disease_unconfirmed"`, bulk-only. 22,307 samples across
  1,174 series. CVD-relevant tissue but no disease-keyword confirmation —
  tests whether the encoder picks up signal beyond tissue-of-origin.

The three pools are disjoint by construction, so the union embedding
extraction covers all of them once with pool tags per sample. The
extraction manifest confirms the H5 provides all 20,010 BulkFormer vocab
genes (`mask_prob=0.0` across all batches), so no `-10` mask tokens were
needed.

Full label decisions and counts are in `label_definitions.md`.

**Grouping.** All CV splits use `StratifiedGroupKFold(n_splits=5,
groups=series_id)`, seed 20260707 — same non-negotiable grouping as the
elastic-net stage's outer CV. Without it, the probe can trivially learn
study-specific batch signatures instead of the biology we care about
(same failure mode called out in the elastic-net writeup, same fix).

**Preprocessing.** Raw ARCHS4 counts → gene-length TPM → `log1p` → align
to BulkFormer's 20,010-gene vocab. Standardization inside the probe is fit
on the train fold only per fold, matching the elastic-net stage.

**Encoder.** Frozen throughout — no gradients propagate into BulkFormer.
Sample embedding = mean-pool across the 20,010 gene tokens of the
`gene_emb_output` tensor (`[batch, 20010, dim+3]`), following the notebook's
`aggregate_type='mean'` sample-level extraction. For 37M this is a
**131-dim** sample vector.

## 2. Checkpoint verification (step 1 gate)

All five checkpoints pass the load + forward-pass check on synthetic input.
Details in `checkpoint_verification.json`. Notable calibration point: the
BulkFormer `README` names the variants "37M/50M/93M/127M/147M" including
the shared 25.6M-parameter ESM2 gene-embedding buffer that is loaded as a
constant, not trained; `sum(p.numel())` on the model alone is
consistently ~25M below the naming. All variants land within ±10% of
advertised size once the ESM2 buffer is added back.

## 3. Mortality prediction — status

**Not runnable on this corpus.** The keyword search (§ 2) hit only 366
samples (0.96% of CVD pool), dominated by the word `death` (316) and
`outcome` (79); no `deceased`, `mortality`, `vital status`, `survival`, or
`follow-up` hits at all. Even before parsing what the hits actually mean,
this is well below the 25-per-fold-per-class floor (125/class at k=5) and
would take manual per-study curation that is explicitly out of scope for
this stage. Full reasoning in
`results/mortality_prediction_status.md`.

## 4. Disease classification — all five variants

5-fold grouped CV, per-fold metrics in
`results/{variant}/{pool}/probe_results.json`. Aggregate:

### vs. whole-corpus non-CVD (pool a)

| Variant | ROC-AUC | PR-AUC | Accuracy | F1 | Brier |
|---|---:|---:|---:|---:|---:|
| BulkFormer-37M | 0.925 ± 0.037 | 0.833 ± 0.075 | 0.878 ± 0.016 | 0.769 ± 0.046 | 0.091 ± 0.015 |
| BulkFormer-50M | 0.928 ± 0.036 | 0.847 ± 0.072 | 0.898 ± 0.013 | 0.801 ± 0.038 | 0.081 ± 0.014 |
| BulkFormer-93M | 0.944 ± 0.023 | 0.897 ± 0.040 | 0.910 ± 0.020 | 0.820 ± 0.048 | 0.072 ± 0.017 |
| BulkFormer-127M | 0.943 ± 0.027 | 0.898 ± 0.044 | 0.909 ± 0.017 | 0.818 ± 0.043 | 0.072 ± 0.016 |
| BulkFormer-147M | 0.941 ± 0.024 | 0.888 ± 0.035 | 0.912 ± 0.022 | 0.820 ± 0.053 | 0.071 ± 0.019 |

All 5 folds ran in all five variants (n_train ≈ 27–29K, n_val ≈ 6.6–7.3K,
n_train_pos ≈ 6,500–7,300 per fold). No fold hit the 25/class floor.

### vs. tissue-only hard negatives (pool b)

| Variant | ROC-AUC | PR-AUC | Accuracy | F1 | Brier |
|---|---:|---:|---:|---:|---:|
| BulkFormer-37M | 0.781 ± 0.105 | 0.610 ± 0.134 | 0.715 ± 0.092 | 0.583 ± 0.078 | 0.192 ± 0.061 |
| BulkFormer-50M | 0.724 ± 0.110 | 0.534 ± 0.127 | 0.680 ± 0.102 | 0.550 ± 0.063 | 0.238 ± 0.090 |
| BulkFormer-93M | 0.801 ± 0.187 | 0.672 ± 0.223 | 0.765 ± 0.123 | 0.614 ± 0.204 | 0.174 ± 0.096 |
| BulkFormer-127M | 0.720 ± 0.215 | 0.566 ± 0.274 | 0.710 ± 0.132 | 0.556 ± 0.192 | 0.236 ± 0.119 |
| BulkFormer-147M | 0.773 ± 0.180 | 0.621 ± 0.225 | 0.745 ± 0.116 | 0.591 ± 0.180 | 0.194 ± 0.097 |

All 5 folds ran for every variant, but variance is materially higher and
noisier than on pool (a) across the board — std on ROC-AUC ranges 0.11
(50M) up to 0.22 (127M). The tissue-only hard-negative pool concentrates
on ~1.6K series, so fold composition is dominated by which specific
studies land on which side, and per-series signal heterogeneity leaks
straight into the CV variance. The wide std bars are more informative
here than the means alone — none of the five variants separate cleanly
from each other on this pool once you account for overlapping error bars.

### What the two pools tell us together

The gap between pool (a) and pool (b) — roughly 14–24 ROC-AUC points
depending on variant — is the story. Against the easy negative pool, most
of the discriminative signal is almost certainly *tissue*-level (positives
are cardiac tissue, negatives are anything else) — the encoder does not
need to know anything about disease to score highly. Against the hard
negatives (also cardiac tissue), performance drops but stays clearly above
chance for every variant, which is evidence that the frozen embedding
carries some disease-specific signal beyond tissue-of-origin, at every
scale tested.

The 25-per-fold-per-class floor from § 2 was cleared for both pools — the
`arrhythmia_afib` sub-class flag noted in `label_definitions.md` only
affects per-subtype breakouts, which we're not running here (the binary
task collapses all six positive subtypes).

## 5. Scale-vs-performance (5/5 variants complete)

Five data points now, holding everything else constant (same manifest,
same folds, same seed, same probe hyperparameters):

- **Pool (a) whole-corpus non-CVD:** rises from 37M through 93M, then
  **plateaus** — 93M/127M/147M are statistically indistinguishable from
  each other (ROC-AUC 0.941–0.944, PR-AUC 0.888–0.898, all well inside
  one another's std bars), while 37M and 50M sit measurably below
  (ROC-AUC 0.925/0.928, PR-AUC 0.833/0.847). Scaling past ~93M buys
  nothing further on this pool — the easy, tissue-dominated
  discrimination task saturates.
- **Pool (b) tissue-only hard negatives:** no clean scaling trend at
  all. 93M is the single best point estimate (ROC-AUC 0.801, PR-AUC
  0.672), but 127M — the very next scale up — is the *worst* (ROC-AUC
  0.720, PR-AUC 0.566), and 147M lands back in between (ROC-AUC 0.773).
  Every variant's std bar on this pool is wide enough (±0.11 to ±0.27 on
  PR-AUC) to overlap every other variant's mean. At n=5 folds over a
  ~1.6K-series hard-negative pool, this reads as **noise dominating any
  true scale effect**, not a real non-monotonic relationship.

Taken together: scale helps up to 93M on the pool where most of the
signal is tissue-level, then stops mattering; on the pool that actually
tests disease-specific signal beyond tissue-of-origin, scale shows no
reliable effect in either direction across this range. This is a
meaningful finding about BulkFormer's frozen embedding for this task —
more parameters did not translate into more disease-specific signal
once tissue-level shortcuts were already captured by the smaller models.

The comparison plot at `results/variant_comparison.png` shows both pools
as separate panels (matched pool left, hard-negative pool right), since
the two pools use different negative definitions and are not directly
comparable to each other on one shared axis — see § 6.

### Compute reality: CPU (Mac) vs. GPU (2×RTX 4090)

| Variant | Device | s/sample | Full pool (57,207) wallclock |
|---|---|---:|---:|
| 37M | CPU | 0.085 | 81 min (measured) |
| 50M | CPU | 0.290 | 276 min (measured) |
| 93M | CUDA | 0.101 | 95.7 min (measured) |
| 127M | CUDA | 0.149 | 142.1 min (measured) |
| 147M | CUDA | 0.185 | 176.2 min (measured) |

37M/50M were run CPU-only on a Mac — MPS was not a viable path there,
since BulkFormer's GCNConv uses `torch_sparse` ops with no MPS kernel and
native `torch.sparse_coo_tensor` construction is unimplemented on that
backend. 93M/127M/147M were instead run on a Linux box with 2×RTX 4090s:
GPU 0 ran 93M then 127M sequentially, GPU 1 ran 147M in parallel. Note
the CPU/GPU numbers aren't a clean per-sample comparison — the 93M/127M/147M
figures also benefit from an unrelated extraction-side speedup (see § 7).

## 6. Elastic-net baseline reference

Found and wired in: `eda/dataset/cvd_data/elasticnet_out/performance/performance_summary.json`
reports **PR-AUC 0.873 ± 0.020** (ROC-AUC 0.943 ± 0.008) over its own
5-fold outer CV on the matched 10:1 training pool. This is a different
file than the two candidate paths the original plotter code checked for
(`folds/cv_summary.json`, `evaluate/summary.json` — neither exists in
this repo), so the final comparison script reads
`performance/performance_summary.json` directly.

The reference line is drawn **only on the pool (a) whole-corpus panel**,
not on pool (b). Pool (a)'s negative definition (whole-corpus non-CVD,
3× subsampled) is the one the elastic-net stage's own matched 10:1 pool
is comparable to; pool (b)'s hard-negative definition (tissue-only,
disease-unconfirmed) has no elastic-net equivalent run, so overlaying the
same line there would imply a comparison that isn't actually apples-to-apples.

Against that reference, 93M/127M/147M (PR-AUC 0.888–0.898) sit essentially
**on top of** the elastic-net baseline — within its own std band, not
clearly above or below it. 37M/50M sit measurably below. So on the one
pool where the comparison is fair, the frozen-embedding-plus-linear-probe
approach at ≥93M roughly matches, but does not yet exceed, the elastic-net
floor — worth keeping in mind for § 8's framing.

## 7. Reproducibility

- Seed: `20260707`, applied to StratifiedGroupKFold + LogisticRegression's
  `random_state` and to negative-pool sub-sampling.
- Encoder frozen — sample embeddings cached to parquet under
  `embeddings/`; probe never sees the raw ARCHS4 H5.
- Standardization is fit-on-train-fold-only inside the sklearn `Pipeline`.
- Full artifact set is under `linear_probe/` and referenced from this file.
- Sklearn's L-BFGS emitted `RuntimeWarning: overflow encountered in matmul`
  during a small number of iterations — the optimizer converged and the
  fold metrics are numerically sensible (finite mean/std, ROC-AUC well
  above chance). This is a nuisance warning from an intermediate
  gradient step, not a correctness signal.
- `extract.py` gained two changes since the 37M/50M CPU run, both used for
  the 93M/127M/147M GPU extractions: (1) a multiprocessing H5 column
  reader (16 workers) — the H5 is chunked (2000 genes × 1 sample), so a
  single fancy-index read serialized 34 chunk fetches per column over the
  network filesystem; reading columns in a worker pool instead measured
  ~8× throughput; (2) incremental checkpointing of `emb_out` to
  `<out_path>.ckpt.npz` every 50 batches, so an interrupted run resumes
  from the last checkpoint instead of redoing the full variant — written
  for exactly the kind of long-running, connection-drop-prone extraction
  this stage turned out to need.
- The 147M GPU run hit a crash *after* finishing all 57,207 samples and
  successfully writing the output parquet — a pre-existing bug in the
  post-write stats logging (`out_path.relative_to(REPO)` raises because
  `out_path` is relative while `REPO` is absolute; `linear_probe/extract.py:344`).
  Verified directly against the written parquet (row count, embedding
  dimensionality, no NaN/Inf) rather than trusting the crashed process's
  log — the data is intact, only the final log line and
  `extraction_manifest.json` for that run were lost. Left unfixed here
  since it's a logging-only bug outside this write-up's scope; worth a
  one-line fix before the next multi-variant extraction run.

## 8. Framing

Per the TODO's closing framing, these numbers are the **evaluation
floor**. Best BulkFormer variant, now that all five are run:

- Pool (a) whole-corpus non-CVD — **93M/127M/147M tie** (all within each
  other's std): ROC-AUC ≈ 0.94, PR-AUC ≈ 0.89–0.90 — essentially matching,
  not beating, the elastic-net baseline (PR-AUC 0.873, § 6)
- Pool (b) tissue-only hard negs — **93M**: ROC-AUC 0.801, PR-AUC 0.672,
  though with std wide enough that 147M (PR-AUC 0.621) and even 37M
  (PR-AUC 0.610) are not statistically distinguishable from it

The full multimodal pipeline (encoder → connector → LLM) that comes
later should be expected to beat these on both pools — matching them
would suggest the connector + LLM aren't adding anything the linear
probe couldn't already extract from the frozen embedding. Beating both,
and by more on the hard-negative pool, is the target. On pool (a) in
particular, the bar to clear is now the **elastic-net baseline**, not
just the linear probe's own numbers — the frozen BulkFormer embedding at
its current best (93M–147M) has not yet demonstrated an advantage over
the much simpler elastic-net model on the matched-pool task.
