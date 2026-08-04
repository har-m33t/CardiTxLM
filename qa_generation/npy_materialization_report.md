# .npy Materialization and Real Loader Verification

The training JSONs referenced 8,553 per-sample `.npy` files that did not exist.
This materializes them and re-verifies the real training data through TinyLLaVA's
actual data pipeline.

| | Result |
|---|---|
| Source | **Row slice of the existing BulkFormer matrix — no transform re-executed** |
| Files written | **8,553** |
| Path resolution | **8,553 / 8,553**, 0 unresolved, 0 orphans |
| Shape/dtype/finiteness | 8,553 checked — all `[20010]` float32, **0 NaN/Inf** |
| Real loader test | **PASS** — both stages |
| On disk | **0.81 GB** |

---

## Task 1 — Source confirmed, transform not re-derived

Each `.npy` is a **byte-exact copy of one row** of
`qa_generation/bulkformer_input/bulkformer_expression.npy` — the matrix built
during the correction pass by calling `linear_probe/extract.py`'s
`normalize_and_align` directly (TPM → log1p over the 20,010-gene ENSG
vocabulary).

Coverage was checked before writing anything:

| | |
|---|---|
| Distinct images referenced by the two training JSONs | **8,553** |
| Rows in `bulkformer_expression.npy` | **8,553** |
| Referenced samples missing from the matrix | **0** |
| Matrix rows not referenced by any QA pair | **0** |

An exact 1:1 correspondence, so every file could be produced by slicing. **No TPM
→ log1p computation runs in this step at all** — not recomputed, not re-read from
the H5. Byte-exactness was verified by reloading 500 written files and comparing
to their source rows: **0 mismatches**.

This is the guardrail's point. A second, independently executed transform is the
divergence the correction pass existed to remove, and copying rows makes that
divergence structurally impossible rather than merely unlikely.

## Task 2 — Why `build_dataset_json.py` was not used directly

`integration/build_dataset_json.py` exists and does write per-sample `.npy`
files, but it is the wrong tool here on two counts:

1. **It selects its own samples.** `select_samples(labels, n_per_class, neg_pool,
   seed)` draws a balanced positive/negative set — not the specific 8,553 the
   training JSONs reference.
2. **It re-derives the vectors**, reading raw ARCHS4 counts from the 62 GB H5 and
   re-running the alignment. Correct when building a set from scratch; here it
   would re-execute a transform whose authoritative output already exists.

So per Task 2's fallback, the per-sample extraction is implemented in
`qa_generation/materialize_npy.py`, reusing the confirmed materialization by
slicing it. `build_dataset_json.py` is left untouched.

## Task 3 — Every referenced path resolves

Not sampled — every distinct path was resolved and loaded:

| Check | Result |
|---|---|
| `stage1_train.json` | 199,954 entries → 8,553 distinct images |
| `stage2_train.json` | 19,793 entries → 8,553 distinct images |
| Union of referenced images | **8,553** |
| Paths that do not resolve under `--image_folder` | **0** |
| Files loaded and inspected | **8,553** |
| Wrong shape (expected `[20010]`) | **0** |
| Wrong dtype (expected float32) | **0** |
| Containing NaN or Inf | **0** |
| Files on disk with no reference (orphans) | **0** |

**8,553 / 8,553 — matches exactly, no gap to explain.** Both JSONs reference the
same 8,553 samples, which is expected: every sample carries both Stage 1 and
Stage 2 QA pairs.

## Task 4 — Real end-to-end loader verification

Real generated QA entries with their real materialized vectors, through
`tinyllava.data.dataset.LazySupervisedDataset` and
`DataCollatorForSupervisedDataset` — the same classes the trainer uses. This
supersedes the original Step 0 check, which used hand-written records and a
synthetic `.npy`.

| Check | Stage 1 | Stage 2 |
|---|---|---|
| Entries loaded | 8 | 8 |
| Collated `images` shape | **(8, 20010)** | **(8, 20010)** |
| `input_ids` shape | (8, 96) | (8, 148) |
| Parses without error | ✓ | ✓ |
| `IMAGE_TOKEN_INDEX` sentinels | **8 / 8** | **8 / 8** |
| Literal `<image>` surviving into decoded text | **0** | **0** |
| Rows sharing one image | 3 | 3 |
| Distinct token sequences among them | **3** | **3** |
| Same vector reused across those rows | ✓ | ✓ |
| Loader tensor == file on disk == source matrix row | **0 mismatches** | **0 mismatches** |

Reading the results:

- **Batch shape** is `[B, 20010]` float32, which is what the BulkFormer tower
  consumes — the `.npy` branch bypasses image preprocessing entirely.
- **`<image>` handling is correct at the token level, not just textually.** The
  literal string is gone from every decoded sequence and replaced by exactly one
  `IMAGE_TOKEN_INDEX` sentinel per example, which is the template logic doing the
  strip-and-reposition rather than the text merely looking right.
- **Multi-turn works as specified.** Three rows share `GSM7073897.npy`
  (Stage 1) / `GSM1126665.npy` (Stage 2) and produce three *distinct* token
  sequences while reusing one identical vector — independent training examples
  over a shared input, exactly the intent.
- **Provenance holds end to end**: the tensor the loader hands the model is
  bit-identical to the file on disk, which is bit-identical to the source matrix
  row.

Labels were also confirmed not to be entirely `IGNORE_INDEX`, so there is
something to train on.

## Task 5 — Final layout

```
data/cvd_transcriptome/
├── embeddings/                  8,553 .npy files   685.7 MB   <- --image_folder
│   └── GSM*.npy                 [20010] float32, 80,168 B each
├── text_files/                                     125.9 MB
│   ├── stage1_train.json        199,954 entries
│   └── stage2_train.json         19,793 entries
└── materialization_manifest.json
```

**Total on disk: 0.81 GB** (811,617,152 bytes).

Training invocation:

| Flag | Value |
|---|---|
| `--image_folder` | `data/cvd_transcriptome/embeddings` |
| `--data_path` (stage 1) | `data/cvd_transcriptome/text_files/stage1_train.json` |
| `--data_path` (stage 2) | `data/cvd_transcriptome/text_files/stage2_train.json` |

The JSONs in `text_files/` are byte-identical copies (SHA-256 verified) of the
canonical files at the repo root, which were **not modified**. That duplicates
126 MB; if you prefer a single copy, delete the repo-root originals rather than
the bundled ones, so the layout stays self-contained.

---

## Handoff status

Everything from the QA-generation pipeline is now materialized and verified:

| Artifact | State |
|---|---|
| `stage1_train.json` | 199,954 entries, unmodified |
| `stage2_train.json` | 19,793 entries, unmodified |
| Per-sample vectors | 8,553 / 8,553 present, verified against source |
| Real loader test | Passing on both stages |

**One thing this does not cover:** the loader test used a tiny stub LLM
tokenizer, so it verifies the *data path* — parsing, collation, token handling,
tensor provenance — not the encoder or a real training step. The existing
`integration/smoke_test.py --real-encoder` covers the model side and needs
`torch_geometric` + `torch_sparse` on the training environment. Worth running
there before Stage 1 pretraining begins.
