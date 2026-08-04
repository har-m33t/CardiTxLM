"""Materialize BulkFormer's actual model input for the disease-confirmed CVD samples.

`gene_pool_prep/bulkformer_vocab_check.md` found that the project's
`cvd_only_expression.npy` — log2(count + 1) over a 49,231-gene QC universe — is
not what the model receives. BulkFormer is fed TPM -> log1p over a 20,010-gene
ENSG vocabulary, and the two disagree on gene *ordering*, not just scale: none of
the 5 audited samples had matching top-10 rankings.

This script builds the matrix the GT layer should have been reading all along, by
reusing `linear_probe/extract.py`'s transform verbatim rather than reimplementing
it. `normalize_and_align` is imported and called directly, so the GT values and
the tensor the encoder sees come from one piece of code:

    raw ARCHS4 counts -> / gene length (kb) -> TPM -> log1p -> align to 20,010 ENSG

Sample coverage is deliberately identical to `cvd_only_expression.npy` (the same
8,553 disease-confirmed rows, from `cvd_only_sample_index.npy`), so no eligibility
or sampling-plan population shifts as a side effect of this correction.

Outputs, under `qa_generation/bulkformer_input/`:
  * `bulkformer_expression.npy`   [8553, 20010] float32, log1p(TPM)
  * `bulkformer_sample_index.npy` global ARCHS4 sample_index per row
  * `symbol_vocab_map.parquet`    gene symbol -> ENSG -> vocab column
  * `bulkformer_matrix_manifest.json`

The symbol map uses the same reconciliation the audit established: symbols and
ENSG ids are read from the *same* H5 gene axis (`meta/genes/symbol` and
`meta/genes/ensembl_gene`), so the pairing is positional inside one file, not an
inferred or name-matched mapping.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
# `linear_probe` is a namespace package (no __init__.py), so it resolves only
# when the repo root is on sys.path. Running this file directly puts
# qa_generation/ there instead; this keeps both `python <file>` and
# `python -m qa_generation.build_bulkformer_matrix` working.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from linear_probe.extract import (  # noqa: E402
    ArchS4CountReader,
    _decode_h5_bytes,
    load_bulkformer_vocab,
    normalize_and_align,
)
CVD_DIR = REPO / "eda/dataset/cvd_data"
H5_PATH = CVD_DIR / "archs4/human_gene_v2.latest.h5"
SAMPLE_INDEX = CVD_DIR / "gene_pool_prep/cvd_only_sample_index.npy"
OUTDIR = REPO / "qa_generation/bulkformer_input"

N_VOCAB = 20010


def build_symbol_map(h5_path: Path, vocab: list[str]) -> pd.DataFrame:
    """symbol -> ENSG -> vocab column, paired positionally on the H5 gene axis."""
    with h5py.File(h5_path, "r") as f:
        symbols = _decode_h5_bytes(f["meta/genes/symbol"][:])
        ensg = _decode_h5_bytes(f["meta/genes/ensembl_gene"][:])
    vocab_pos = {g: i for i, g in enumerate(vocab)}

    rows = []
    for sym, gid in zip(symbols, ensg):
        if not gid or gid == "nan":
            continue
        bare = gid.split(".")[0]
        pos = vocab_pos.get(bare)
        if pos is not None:
            rows.append((sym, bare, pos))
    out = pd.DataFrame(rows, columns=["gene_symbol", "ensg_id", "vocab_pos"])
    return out.drop_duplicates().reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("build_bulkformer_matrix")
    # normalize_and_align logs once per call; at 34 batches that is just noise.
    quiet = logging.getLogger("quiet")
    quiet.addHandler(logging.NullHandler())
    quiet.propagate = False

    started = datetime.now(timezone.utc)
    for path in (H5_PATH, SAMPLE_INDEX):
        if not path.exists():
            raise SystemExit(f"STOP: missing required input {path}")

    vocab, length_dict = load_bulkformer_vocab(logger)
    if len(vocab) != N_VOCAB:
        raise SystemExit(f"STOP: vocab is {len(vocab)}, expected {N_VOCAB}")

    sample_index = np.load(SAMPLE_INDEX)
    n_samples = len(sample_index)
    logger.info(f"materializing {n_samples} samples x {len(vocab)} vocab genes")

    reader = ArchS4CountReader(H5_PATH, sample_index, logger)
    out = np.empty((n_samples, len(vocab)), dtype=np.float32)
    mask_probs = []
    for start in range(0, n_samples, args.batch_size):
        stop = min(start + args.batch_size, n_samples)
        counts = reader.read_batch(sample_index[start:stop])
        aligned, mask_prob = normalize_and_align(
            counts, reader.h5_gene_symbols, vocab, length_dict, quiet
        )
        out[start:stop] = aligned
        mask_probs.append(mask_prob)
        if start % (args.batch_size * 5) == 0 or stop == n_samples:
            logger.info(f"  {stop}/{n_samples}")

    logger.info("building symbol -> vocab map")
    symbol_map = build_symbol_map(H5_PATH, vocab)

    checks: dict = {}
    failures: list[str] = []
    checks["shape"] = list(out.shape)
    checks["shape_correct"] = out.shape == (n_samples, len(vocab))
    checks["n_nan"] = int(np.isnan(out).sum())
    checks["n_inf"] = int(np.isinf(out).sum())
    checks["no_nan_or_inf"] = checks["n_nan"] == 0 and checks["n_inf"] == 0
    # normalize_and_align writes -10.0 where a vocab gene is absent from the H5.
    # The audit measured 0 missing; anything else means the GT layer would be
    # reporting a sentinel as if it were an expression value.
    checks["n_masked_sentinel"] = int((out == -10.0).sum())
    checks["no_masked_genes"] = checks["n_masked_sentinel"] == 0
    checks["max_mask_prob"] = float(max(mask_probs))
    checks["mask_prob_zero"] = checks["max_mask_prob"] == 0.0
    checks["min_value"] = float(out.min())
    checks["max_value"] = float(out.max())
    checks["min_non_negative"] = bool(out.min() >= 0.0)
    checks["n_symbol_map_rows"] = int(len(symbol_map))
    checks["n_symbols_mapped"] = int(symbol_map.gene_symbol.nunique())
    checks["vocab_cols_covered"] = int(symbol_map.vocab_pos.nunique())
    checks["symbol_map_covers_full_vocab"] = checks["vocab_cols_covered"] == len(vocab)

    for key, val in checks.items():
        if isinstance(val, bool) and not val:
            failures.append(key)
    checks["passed"] = not failures
    checks["failures"] = failures

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.save(args.outdir / "bulkformer_expression.npy", out)
    np.save(args.outdir / "bulkformer_sample_index.npy", sample_index)
    symbol_map.to_parquet(args.outdir / "symbol_vocab_map.parquet", index=False)

    finished = datetime.now(timezone.utc)
    manifest = {
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "elapsed_seconds": round((finished - started).total_seconds(), 1),
        "purpose": (
            "BulkFormer's actual model input for the 8,553 disease-confirmed CVD "
            "samples — the expression source for Stage 1 GT functions."
        ),
        "why_this_exists": (
            "gene_pool_prep/bulkformer_vocab_check.md found the project's "
            "log2(count + 1) matrix disagrees with BulkFormer's TPM -> log1p input "
            "on gene ordering (0 of 5 audited samples had matching top-10s), so GT "
            "computed from it does not describe what the model receives."
        ),
        "transform": {
            "code_reused_from": "linear_probe/extract.py::normalize_and_align",
            "steps": "raw counts / gene_length_kb -> TPM (per-sample, 1e6) -> log1p",
            "vocabulary": "bulkformer_gene_info.csv ensg_id, 20,010 genes",
            "units": "log1p(TPM)",
            "reimplemented": False,
        },
        "identifier_reconciliation": (
            "symbol and ENSG read from the same H5 gene axis "
            "(meta/genes/symbol, meta/genes/ensembl_gene) — positional pairing "
            "within one file, per bulkformer_vocab_check.md section 2"
        ),
        "sample_population": {
            "n_samples": int(n_samples),
            "source": str(SAMPLE_INDEX.relative_to(REPO)),
            "note": (
                "identical rows to cvd_only_expression.npy so no eligibility "
                "population changes as a side effect"
            ),
        },
        "sources": {
            "h5": str(H5_PATH.relative_to(REPO)),
            "vocab": "bulkencoders/checkpoints/bulkformer/support/bulkformer_gene_info.csv",
            "gene_lengths": "bulkencoders/checkpoints/bulkformer/support/gene_length_df.csv",
        },
        "sanity_checks": checks,
        "outputs": {
            "expression": "bulkformer_expression.npy",
            "sample_index": "bulkformer_sample_index.npy",
            "symbol_map": "symbol_vocab_map.parquet",
        },
    }
    (args.outdir / "bulkformer_matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    logger.info(f"wrote {out.shape} -> {args.outdir}")
    if not checks["passed"]:
        logger.error(f"FAILED checks: {failures}")
        return 1
    logger.info(f"all checks passed ({manifest['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
