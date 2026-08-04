"""Write one .npy per sample from the already-materialized BulkFormer matrix.

`stage1_train.json` / `stage2_train.json` reference `<GSM accession>.npy` for
8,553 samples. Those per-sample files are produced here by **slicing rows out of
`qa_generation/bulkformer_input/bulkformer_expression.npy`** — the matrix built
during the correction pass by calling `linear_probe/extract.py`'s
`normalize_and_align` directly.

No transform runs in this script. TPM -> log1p is not recomputed, not
re-derived, and not re-read from the H5: each output file is a byte-exact copy of
one row of the existing matrix. That is deliberate. A second, independently
executed transform is exactly the divergence the correction pass was built to
eliminate — `integration/build_dataset_json.py` would re-read raw counts and
re-run the alignment, which is the right thing when building a set from scratch
but the wrong thing here, where an authoritative materialization already exists.

Output layout (`--outdir`, default `data/cvd_transcriptome`):

    embeddings/<GSM>.npy   <- pass this directory as --image_folder
    text_files/stage{1,2}_train.json

The training JSONs are read, never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from qa_generation import gt_functions as gt

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "data/cvd_transcriptome"
#: Canonical home of the training JSONs. They live inside the bundle so the
#: handoff tree — vectors, text and manifest — is self-contained and there is
#: exactly one copy of each file to keep in sync.
TEXT_FILES = DEFAULT_OUT / "text_files"
TRAIN_JSONS = {1: TEXT_FILES / "stage1_train.json", 2: TEXT_FILES / "stage2_train.json"}
N_VOCAB = 20010


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def referenced_images() -> set[str]:
    """Every distinct `image` value across both training JSONs."""
    out: set[str] = set()
    for path in TRAIN_JSONS.values():
        out |= {entry["image"] for entry in json.loads(path.read_text())}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-sample", type=int, default=500)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    images = referenced_images()
    log(f"{len(images)} distinct image paths referenced by the training JSONs")

    rows = gt._expression_rows()
    matrix = np.load(gt.EXPRESSION_NPY, mmap_mode="r")
    if matrix.shape[1] != N_VOCAB:
        raise SystemExit(f"STOP: matrix has {matrix.shape[1]} genes, expected {N_VOCAB}")

    accessions = {img[:-4] for img in images}
    missing = sorted(accessions - set(rows))
    if missing:
        raise SystemExit(
            f"STOP: {len(missing)} referenced samples have no row in "
            f"{gt.EXPRESSION_NPY.name} (e.g. {missing[:5]}). Refusing to write "
            f"placeholder data for them."
        )

    emb = args.outdir / "embeddings"
    txt = args.outdir / "text_files"
    emb.mkdir(parents=True, exist_ok=True)
    txt.mkdir(parents=True, exist_ok=True)

    written = 0
    for image in sorted(images):
        vector = np.asarray(matrix[rows[image[:-4]]], dtype=np.float32)
        np.save(emb / image, vector)
        written += 1
        if written % 2000 == 0:
            log(f"  {written}/{len(images)}")
    log(f"wrote {written} .npy files -> {emb}")

    # Only copies when writing somewhere other than the canonical bundle; with
    # the default --outdir the JSONs are already at the destination.
    for src in TRAIN_JSONS.values():
        dst = txt / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    log(f"training JSONs in place -> {txt}")

    # Byte-exactness: reload a sample of files and compare to the matrix row.
    log("verifying against source rows")
    rng = np.random.default_rng(0)
    check = sorted(images)
    if args.verify_sample and args.verify_sample < len(check):
        check = [check[i] for i in rng.choice(len(check), args.verify_sample, False)]
    mismatched = []
    for image in check:
        loaded = np.load(emb / image)
        source = np.asarray(matrix[rows[image[:-4]]], dtype=np.float32)
        if loaded.shape != (N_VOCAB,) or loaded.dtype != np.float32:
            mismatched.append((image, "shape/dtype"))
        elif not np.array_equal(loaded, source):
            mismatched.append((image, "content"))

    total_bytes = sum(p.stat().st_size for p in emb.glob("*.npy"))
    checks = {
        "n_referenced": len(images),
        "n_written": written,
        "all_referenced_written": written == len(images),
        "verified_sample": len(check),
        "byte_exact_mismatches": len(mismatched),
        "byte_exact": not mismatched,
        "embeddings_bytes": total_bytes,
    }
    checks["passed"] = checks["all_referenced_written"] and checks["byte_exact"]

    manifest = {
        "started": started.isoformat(),
        "finished": datetime.now(timezone.utc).isoformat(),
        "source_matrix": str(gt.EXPRESSION_NPY.relative_to(REPO)),
        "method": (
            "row slice of the existing materialization; no transform executed, "
            "each file is a byte-exact copy of one matrix row"
        ),
        "transform_provenance": (
            "TPM -> log1p over the 20,010-gene ENSG vocabulary, produced by "
            "linear_probe/extract.py::normalize_and_align during the correction "
            "pass (qa_generation/build_bulkformer_matrix.py)"
        ),
        "image_folder": str(emb.relative_to(REPO)),
        "sanity_checks": checks,
    }
    (args.outdir / "materialization_manifest.json").write_text(json.dumps(manifest, indent=2))

    log(json.dumps(checks))
    if not checks["passed"]:
        log(f"FAILED: {mismatched[:5]}")
        return 1
    log("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
