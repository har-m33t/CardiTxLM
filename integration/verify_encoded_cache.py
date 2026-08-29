"""Confirm the shipped 515-d cache really is what the tower emits — on the pod.

WHY THIS EXISTS
---------------
Stage-2 training feeds pre-encoded [515] vectors through
`BulkFormerVisionTower.forward`'s passthrough instead of raw [20010] ones. Those
vectors were DERIVED from `linear_probe/embeddings/embeddings_BulkFormer-93M.parquet`
rather than produced by a GPU encoder pass — a shortcut that removes an entire
GPU stage, and one that is only valid if the parquet is bitwise what the tower
would have produced.

That was established locally: bitwise-exact (max_abs_diff 0.0) against a live
CPU forward of the real tower over three disjoint sample groups, with a
rolled-cache control proving the comparison discriminates. See
`data/cvd_transcriptome/encoded_cache_manifest.json`.

This re-checks it on the GPU box, on different hardware and a different
BulkFormer checkpoint download, before ~$4/hour of training depends on it.

A full re-encode is impossible here on purpose: the raw [20010] vectors are
668 MB and are deliberately not shipped to the pod — not shipping them is the
whole point of the cache. So a 16-sample slice of raw vectors travels with the
repo (1.3 MB) purely so this check can run. That is enough: the encoder is
deterministic per input, so if 16 samples round-trip exactly, the derivation is
sound; and if they do not, the derivation is broken for all of them.

Run:
    python -m integration.verify_encoded_cache
Exit status is the check result, so a bootstrap can gate on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
RAW_SAMPLE = REPO / "data/cvd_transcriptome/verify_sample_raw"
CACHE = REPO / "data/cvd_transcriptome/embeddings_encoded"
CFG = REPO / "integration/bulkformer_hf_config"
TRAIN_JSON = REPO / "data/cvd_transcriptome/text_files/stage2_train.json"

#: Same acceptance bound the local build used. The local measurement came in at
#: exactly 0.0; CUDA batched reductions vary with batch composition, so a GPU
#: run is allowed a little float slack without weakening the check to
#: meaninglessness (the rolled-vector control below lands at ~0.3-1.0, i.e.
#: four orders of magnitude away).
TOL = 1e-4


def structural_checks() -> list[str]:
    """Cheap invariants that do not need the encoder."""
    failures = []
    files = sorted(CACHE.glob("GSM*.npy"))
    if not files:
        return [f"no cached vectors under {CACHE}"]

    cfg = json.loads((CFG / "config.json").read_text())
    if cfg.get("bulkformer_variant") != "BulkFormer-93M":
        failures.append(f"encoder scale not 93M: {cfg.get('bulkformer_variant')}")
    dim = cfg.get("hidden_size")

    bad_shape = 0
    for f in files[:200]:
        v = np.load(f)
        if v.shape != (dim,) or not np.isfinite(v).all():
            bad_shape += 1
    if bad_shape:
        failures.append(f"{bad_shape} of the first 200 cached vectors are "
                        f"malformed (expected finite shape ({dim},))")

    if TRAIN_JSON.exists():
        wanted = {it["image"] for it in json.loads(TRAIN_JSON.read_text())}
        have = {f.name for f in files}
        missing = wanted - have
        if missing:
            failures.append(
                f"{len(missing)} images referenced by stage2_train.json have no "
                f"cached vector, e.g. {sorted(missing)[:3]}"
            )
    print(f"[structural] {len(files):,} cached vectors, dim {dim}, "
          f"{'OK' if not failures else 'FAILURES'}")
    return failures


def numeric_check(device: str) -> list[str]:
    """Live tower forward on the shipped raw slice, compared to the cache."""
    if not RAW_SAMPLE.exists():
        return [f"raw verification slice missing at {RAW_SAMPLE}"]
    raw_files = sorted(RAW_SAMPLE.glob("GSM*.npy"))
    if not raw_files:
        return [f"no GSM*.npy under {RAW_SAMPLE}"]

    from integration.precompute_encoder_cache import build_tower

    dev = torch.device(device)
    tower, cfg = build_tower(CFG, dev)
    print(f"[numeric] tower {tower._variant} dim {tower.embed_dim} on {dev}")

    x = torch.from_numpy(
        np.stack([np.load(f) for f in raw_files])
    ).to(dev)
    with torch.no_grad():
        live = tower(x).squeeze(1).float().cpu().numpy()

    cached = np.stack([np.load(CACHE / f.name) for f in raw_files])
    max_abs = float(np.abs(live - cached).max())
    scale = float(np.abs(cached).mean())

    # Control: the same comparison against ROLLED cache rows must be large.
    # Without it, a bug that made both sides identically wrong (or that compared
    # an array to itself) would pass silently.
    control = float(np.abs(live - np.roll(cached, 1, axis=0)).max())

    print(f"[numeric] n={len(raw_files)} value_scale={scale:.5f}")
    print(f"[numeric] live vs cache   max_abs={max_abs:.3e}  (tol {TOL:.0e})")
    print(f"[numeric] rolled control  max_abs={control:.3e}  (must be >> tol)")

    failures = []
    if not (max_abs <= TOL):
        failures.append(
            f"cache disagrees with a live tower forward by {max_abs:.3e} "
            f"(tolerance {TOL:.0e}) — do NOT train on it"
        )
    if not (control > TOL * 100):
        failures.append(
            f"rolled-vector control is only {control:.3e}; the comparison is "
            f"not discriminating, so the pass above proves nothing"
        )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-numeric", action="store_true",
                    help="structural checks only (no GPU / no encoder needed)")
    args = ap.parse_args()

    failures = structural_checks()
    if not args.skip_numeric:
        failures += numeric_check(args.device)

    for f in failures:
        print(f"FAIL: {f}")
    if failures:
        print("\nThe pre-encoded cache cannot be trusted. Either fix it or point "
              "--image_folder at raw [20010] vectors and let the tower encode.")
        return 1
    if args.skip_numeric:
        # Say only what was actually checked. Claiming the numeric result here
        # would be the same class of error as the eval instruments in
        # comparison_report.md: a plausible-looking pass that was never measured.
        print("\nPASS (structural only): shapes, finiteness and coverage are "
              "correct. Numeric equivalence to the tower was NOT checked in "
              "this run — rerun without --skip-numeric on a box with the "
              "encoder available.")
    else:
        print("\nPASS: the shipped cache reproduces the live tower forward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
