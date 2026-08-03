# gene_pool_prep

Prerequisite artifacts for gene-pool curation
(`.claude/gene_pool_prerequisites_todo.md`).

## Track 2 — external gene set acquisition (complete)

Track 2 is the KEGG fetch only. The ClinGen HCVD portal export was dropped
from this track; no `clingen_hcvd_genes.csv` is produced here.

### KEGG cardiomyopathy pathways

**Source:** KEGG REST API, `rest.kegg.jp`, fetched 2026-08-02. Enrichr's
`KEGG_2021_Human` GMT was the designated fallback and was not needed —
all three endpoints answered HTTP 200 directly.

**Cached, not live.** The three raw responses under `raw/` are the
acquisition:

| file | endpoint |
| --- | --- |
| `raw/kegg_hsa05410_link.tsv` | `/link/hsa/path:hsa05410` |
| `raw/kegg_hsa05414_link.tsv` | `/link/hsa/path:hsa05414` |
| `raw/kegg_hsa_gene_list.tsv` | `/list/hsa` (gene id → symbol) |

`build_kegg_cardiomyopathy.py` parses those cached files and does **not**
touch the network on a normal run. Re-fetching is an explicit manual act
(`--refetch`), never a pipeline-run side effect.

**Counts:**

| | genes |
| --- | --- |
| hsa05410 (dilated cardiomyopathy) | 99 |
| hsa05414 (hypertrophic cardiomyopathy) | 105 |
| union, deduplicated | 114 |
| **surviving QC-universe intersection** | **114 (100.0%)** |

In both pathways: 89 genes. No drop-off to flag.

**Gene universe.** `gene_symbols.npy` (49,231 symbols, already post-QC-mask)
is reused exactly as it exists; `kept_gene_mask.npy` is loaded only to
assert the two agree. The QC filter is never recomputed here.

**Symbol casing.** The universe uses ARCHS4-style uppercase (`C1ORF112`),
KEGG uses HGNC casing (`C1orf112`), so all matching is done on uppercased
symbols and the universe's own spelling is what gets written out. Without
this the intersection would appear to lose a large fraction of the pathway
genes, entirely as an artifact.

**Known gap.** One KEGG entry in hsa05414, `hsa:102723407`, declares no
gene symbol at all (its `/list/hsa` record carries a bare description,
as ~1,400 predicted-locus and ncRNA records do). It is reported as a
warning by the build script rather than silently dropped, and is not
counted in the 114.

**Output:** `kegg_cardiomyopathy_genes.csv` — columns `gene`,
`kegg_symbol`, `source_pathway` (`hsa05410` / `hsa05414` / `both`),
`matched_via`.

## Track 3 — variance ranking (complete)

Per-gene variance across the sample axis of Track 1's CVD-only matrix
(8,553 bulk disease-confirmed samples × 49,231 genes), on the log2 values
already in that matrix — no re-transform, no re-normalisation.

**Threshold: top 10%** (Adam et al.), confirmed 2026-08-02 →
**4,924 of 49,231 genes**, boundary variance 9.104.

**Method.** Unbiased sample variance (ddof=1), two-pass in float64:
streaming mean, then streaming sum of squared deviations. Two-pass rather
than the sum/sum-of-squares shortcut, which cancels catastrophically on
high-mean low-spread genes — precisely the housekeeping genes sitting near
the selection boundary. ddof cannot affect the ranking (constant factor);
it is recorded for citability.

**Population guardrail.** `compute_variance_genes.py` reads Track 1's
manifest and hard-fails unless it declares `pool_definition =
disease_confirmed` with a sample count below the elastic net's 34,900.
Pointing this at `elasticnet_out/expression/X.npy` would rank genes by
variance across a CVD-vs-random-negative pool — mostly tissue-of-origin
signal from the 26,175 random negatives — which is the original bug this
whole effort exists to fix. That path is closed by construction, not by
convention.

**Plausibility.** Every canonical cardiac gene lands in the top 50:

| gene | rank | gene | rank |
| --- | --- | --- | --- |
| MYH7 | **1** | ACTA1 | 30 |
| MYBPC3 | 5 | TTN | 38 |
| MYH6 | 6 | NPPB | 46 |
| TNNT2 | 7 | GAPDH | 1,555 |
| NPPA | 15 | ACTB | 2,337 |

Housekeeping controls (GAPDH, ACTB) are present but unremarkable, as they
should be. 76 of Track 2's 114 KEGG cardiomyopathy genes independently
clear the threshold.

**Output:** `high_variance_genes.csv` — the **full** universe, all 49,231
genes rather than only the selected ones, so Track 5 can cite a rank for
any gene: `gene`, `variance`, `variance_rank` (1 = highest), `selected`.
Plus `variance_manifest.json`.

## Track 4 — co-expression in-degree centrality (complete, with caveats)

Co-expression graph over Track 1's CVD-only matrix (8,553 bulk
disease-confirmed samples × 49,231 genes).

**Threshold: top 10% → 4,924 genes**, read from Track 3's
`variance_manifest.json` rather than chosen here, so both rankings
contribute equally sized sets to Track 5's union.

**In-degree, never out-degree.** Each gene draws a directed edge to each of
its top-100 most-correlated partners. The score is how many *other* genes
name this gene. Out-degree is exactly 100 for all 49,231 genes by
construction and carries no signal; the script asserts against it, failing
if more than half the universe lands in the [90, 110] band. Observed:
**4.9%**, with in-degree spanning 0–3,266 (median 40, mean 100).

**Approach: blocked, not dense.** A full 49,231² float32 correlation matrix
is ~9.7 GB against 25.7 GB of system RAM under existing pressure, and only
the top 100 entries per row are ever read. Each gene column is z-scored and
scaled by 1/√(n−1), which makes a row block's Pearson correlation exactly
`block.T @ Z` — one BLAS sgemm per block, ~403 MB peak for the correlation
slab plus ~71 MB for the transposed block. 25 blocks, ~40 s total.

Correctness was checked against a brute-force `np.corrcoef` reference on
synthetic data with a planted hub: **max in-degree difference 0**.

**Sanity checks — all pass.** 0 non-finite correlations, 0 zero-variance
genes (min std 0.290), edges exactly 49,231 × 100, ranks a complete
1..49,231 permutation, selected count 4,924 matching Track 3.

Correlation bounds are measured off-diagonal, *after* self-correlation is
masked. Measuring before was a bug in an earlier revision: a gene's
self-correlation is exactly 1.0, which float32 renders as up to 1.00018, so
the check was reporting float32 epsilon on a value that gets discarded. The
1e-3 tolerance is sized from a measured float32-vs-float64 deviation of
5.9e-5 (mean 2.6e-6) on off-diagonal entries.

### Caveats worth carrying into Track 5

**The top of this ranking is not biologically interpretable.** The highest
in-degree genes are lncRNAs, pseudogenes and unannotated ENSG identifiers
(`FRG1-DT` 3,266; `SNRPGP18`; `ENSG00000227948`; `LINC01226`), whose mean
log2 expression (2.73 across the top 500) sits *below* the universe-wide
mean of 3.03. This is the familiar co-expression hub artifact — sparsely
detected features co-vary through shared detection/library-depth patterns
rather than biology. It is a property of the measure, not a bug in the
implementation.

**Cardiac genes rank mid-pack, and two miss the cut.** MYH7 (8.1%), MYBPC3
(7.5%), ACTC1 (9.9%) and ACTN2 (5.4%) are selected; **TTN (12.8%) and NPPA
(18.5%) are not**, nor are MYH6, TNNT2 or NPPB. Only 21 of Track 2's 114
KEGG genes clear this threshold, against 76 for variance.

This is defensible rather than alarming: TTN and MYH7 are tissue-*specific*,
correlating tightly with a small cardiac module, so few of the other ~49,000
genes name them among their top 100. Centrality is measuring a genuinely
different axis from variance — which is the reason Track 5 unions three
sources instead of trusting one. **Every canonical cardiac gene above is
still captured by the union** through variance and/or KEGG membership, so
nothing is lost; verified explicitly.

A post-hoc expression floor does *not* fix the artifact (at detection ≥ 0.8
the cardiac genes rank *worse*, not better). Genuinely suppressing it would
mean rebuilding the graph on a restricted universe — a methodology change,
deliberately not made unilaterally here.

**Duplicate gene symbols.** 1,534 symbols map to more than one column
(4,225 columns, up to 15 per symbol; 46 of 155 sampled same-symbol pairs are
byte-identical). Identical columns are guaranteed to be each other's top
partner. Measured impact is small — **6,443 of 4,923,100 edges (0.13%)** —
but it is why `FRG1-DT` occupies ranks 1 and 2, and why the 4,924 selected
rows cover 4,823 distinct symbols. Track 5 should deduplicate on symbol.

**Output:** `high_centrality_genes.csv` — the full universe, all 49,231
genes: `gene`, `in_degree`, `centrality_rank` (1 = highest), `selected`.
Plus `centrality_manifest.json`.
