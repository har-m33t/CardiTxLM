"""Close the one verification gap in the negative-sample encoded cache.

`integration/build_encoded_cache_from_parquet.py --population union` writes the
515-d cache for all 31,032 positives + `neg_hard` negatives by slicing rows out
of `embeddings_BulkFormer-93M.parquet`. Its live-forward check can only run on
samples whose raw [20010] vector exists on disk — and that is the 8,553 Stage-2
positives only. So 22,479 cache members, INCLUDING EVERY NEGATIVE, were written
on the strength of an unproven claim: that their parquet row equals a live
forward of the real tower over their own expression profile.

That gap matters here specifically. Hypothesis B trains a disease-vs-control
discriminative task, so a systematic encoding error on one side of the label
would either manufacture the effect or destroy it, and the probe could not tell
the difference. The claim is cheap to test — the ARCHS4 H5 is local — so it is
tested rather than argued.

Method, deliberately identical to the positives' path so the comparison is
apples-to-apples:
  1. sample N random `is_neg_hard` samples,
  2. read their raw columns from the H5 with `ArchS4CountReader`,
  3. TPM -> log1p -> 20,010-gene vocab via `normalize_and_align`,
  4. forward through the REAL `BulkFormerVisionTower` on CPU,
  5. compare against the cached .npy.

Plus the standing control: the same live output against a ROLLED cache. Without
it a comparison that always returns "equal" is indistinguishable from a working
one. Acceptance is max_abs_diff <= 1e-4 AND control >> that.

    python3 scripts/hypothesis_b/verify_negative_encoding.py --n 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from linear_probe.extract import ArchS4CountReader, normalize_and_align  # noqa: E402
from qa_generation.build_per_sample_de import load_bulkformer_vocab  # noqa: E402
from integration.build_encoded_cache_from_parquet import build_tower  # noqa: E402

H5_PATH = REPO / "eda/dataset/cvd_data/archs4/human_gene_v2.latest.h5"
LABELS = REPO / "linear_probe/probe_sample_labels.parquet"
CACHE = REPO / "data/cvd_transcriptome/embeddings_encoded"
CFG = REPO / "integration/bulkformer_hf_config"
OUT = REPO / "scripts/hypothesis_b/negative_encoding_verification.json"
TOLERANCE = 1e-4
SEED = 20260901


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("verify_neg")
    import torch

    lab = pd.read_parquet(LABELS)
    neg = lab[lab.is_neg_hard]
    # Only samples the cache actually holds; spread across the index range so a
    # localized corruption cannot hide behind a contiguous draw.
    neg = neg[[(CACHE / f"{a}.npy").exists() for a in neg.geo_accession]]
    rng = np.random.default_rng(SEED)
    pick = neg.iloc[np.sort(rng.choice(len(neg), size=args.n, replace=False))]
    log.info(f"drawn {len(pick)} of {len(neg):,} cached negatives")

    idx = pick.sample_index.to_numpy(dtype=np.int64)
    accs = pick.geo_accession.tolist()

    vocab, length_dict = load_bulkformer_vocab(log)
    reader = ArchS4CountReader(H5_PATH, idx, log)
    try:
        counts = reader.read_batch(idx)
        aligned, mask_prob = normalize_and_align(
            counts, reader.h5_gene_symbols, vocab, length_dict, log)
    finally:
        reader.close()
    log.info(f"aligned {aligned.shape}, mask_prob={mask_prob:.6f}")

    tower, cfg, shimmed = build_tower(CFG)
    assert tower._variant == "BulkFormer-93M", tower._variant
    assert tower.embed_dim == cfg.hidden_size == 515

    with torch.no_grad():
        live = tower(torch.from_numpy(aligned.astype(np.float32))
                     ).squeeze(1).float().numpy()
    cached = np.stack([np.load(CACHE / f"{a}.npy") for a in accs])
    assert live.shape == cached.shape == (len(accs), 515), (live.shape, cached.shape)

    d = float(np.max(np.abs(live - cached)))
    # CONTROL: roll the cache by one row. If the comparison is meaningful this
    # must be large; if it also came out ~0 the test proves nothing.
    d_ctrl = float(np.max(np.abs(live - np.roll(cached, 1, axis=0))))
    ok = d <= TOLERANCE and d_ctrl > 100 * max(d, 1e-9)

    log.info(f"\nlive vs cached  max_abs_diff : {d:.3e}   (tolerance {TOLERANCE:.0e})")
    log.info(f"CONTROL rolled  max_abs_diff : {d_ctrl:.3e}   (must be large)")
    log.info(f"mean |value|                 : {float(np.mean(np.abs(cached))):.4f}")
    log.info(f"\n{'PASS' if ok else 'FAIL'}")

    OUT.write_text(json.dumps({
        "purpose": ("live-encoder verification of NEGATIVE (is_neg_hard) cache "
                    "entries — the population build_encoded_cache_from_parquet "
                    "could not check, because negatives have no raw vector on disk"),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_verified": len(accs),
        "n_cached_negatives": int(len(neg)),
        "samples": accs,
        "sample_index": idx.tolist(),
        "source": "raw H5 columns -> normalize_and_align -> real BulkFormerVisionTower (CPU)",
        "mask_prob": float(mask_prob),
        "max_abs_diff": d,
        "control_rolled_cache_max_abs_diff": d_ctrl,
        "tolerance": TOLERANCE,
        "deepspeed_import_shim": bool(shimmed),
        "passed": bool(ok),
    }, indent=2) + "\n")
    log.info(f"wrote {OUT.relative_to(REPO)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
