"""build_encoded_cache_from_parquet.py — materialize the pre-encoded 515-d
Stage-2 vector cache WITHOUT a GPU, by reusing embeddings that already exist.

WHY THIS IS LEGITIMATE (and not a shortcut around the encoder):

`integration/precompute_encoder_cache.py` builds the same cache by running the
frozen BulkFormer tower over every raw [20010] vector — ~13 min on an A100, far
longer on CPU. But the pooled [515] vectors for the whole 57,207-sample probe
population were ALREADY produced, by `linear_probe/extract.py`, which calls
    model(x, mask_prob=mask_prob, output_expr=False).mean(dim=1)
i.e. the identical call and identical gene-axis mean-pool that
`BulkFormerVisionTower.forward` performs. `linear_probe/embeddings/
extraction_manifest.json` records `mask_prob_mean = mask_prob_std = 0.0` for
BulkFormer-93M, so that run's mask_prob was 0.0 everywhere — matching the
tower's hardcoded `mask_prob=0.0`. The two paths are the same computation.

So the 8,553 Stage-2 rows can simply be sliced out of the parquet and written as
per-accession `.npy` files. `BulkFormerVisionTower.forward` detects them by
width (515 == embed_dim) and passes them through untouched.

This script does NOT take that equivalence on faith. `--verify N` instantiates
the REAL BulkFormerVisionTower on CPU from `integration/bulkformer_hf_config`
(locked to BulkFormer-93M, hidden_size 515), runs a live forward on N raw
[20010] vectors, and compares against what was written. Acceptance is
max_abs_diff <= 1e-4 (the prior GPU-built cache measured 7.87e-06 against a
live forward under different batching). A failure is reported, not swallowed.

POPULATIONS
-----------
`--population stage2` (default) reproduces the original build: the 8,553
Stage-2 positives listed in `bulkformer_sample_index.npy`.

`--population union` extends the cache to the full 31,032-sample union of every
`is_positive` (8,725) and every `is_neg_hard` (22,307) sample in
`linear_probe/probe_sample_labels.parquet`. Hypothesis B adds a
disease-vs-no-disease discriminative task, so negatives are now model INPUTS and
need cached vectors too.

Cache membership is NOT training membership. The union deliberately includes the
92 holdout series (1,341 positives + 1,266 neg_hard), because those samples are
the inputs to the held-out binary CVD evaluation. The train/eval split is
enforced separately, at data-generation time, by
`data/cvd_transcriptome/holdout_series.json`.

Writes are skip-if-identical: an existing file is re-derived from the parquet
and left untouched if byte-identical. A mismatch is REPORTED, never overwritten.

Run:
    python3 -m integration.build_encoded_cache_from_parquet \
        --parquet linear_probe/embeddings/embeddings_BulkFormer-93M.parquet \
        --sample-index qa_generation/bulkformer_input/bulkformer_sample_index.npy \
        --raw data/cvd_transcriptome/embeddings \
        --dest data/cvd_transcriptome/embeddings_encoded \
        --verify 8

    python3 -m integration.build_encoded_cache_from_parquet \
        --population union \
        --manifest data/cvd_transcriptome/encoded_cache_manifest_v2.json \
        --verify 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO / "integration" / "bulkformer_hf_config"
TOLERANCE = 1e-4


def load_population(parquet_path):
    """Read the probe embedding parquet.

    Embedding columns come from `eval_binary_comparison.embedding_io.
    embedding_columns`, which is the single correct definition of the
    convention in this repo. It replaces the `c.startswith("e0")` filter this
    function used to carry: that filter silently keeps only e0000..e0999 and has
    already truncated two analyses in this project (see that module's
    docstring). It happens to be harmless at 515-d — every column really does
    start with "e0" — but it must not be propagated by copy-paste.
    """
    import pyarrow.parquet as pq
    from eval_binary_comparison.embedding_io import embedding_columns
    t = pq.read_table(parquet_path).to_pydict()
    ecols = embedding_columns(t.keys())
    n = len(t["geo_accession"])
    X = np.empty((n, len(ecols)), dtype=np.float32)
    for j, c in enumerate(ecols):
        X[:, j] = np.asarray(t[c], dtype=np.float32)
    meta = {k: list(t[k]) for k in
            ("geo_accession", "series_id", "cvd_subtype", "is_positive",
             "is_neg_hard", "is_neg_whole_corpus", "pool", "sample_index")}
    return X, meta


def _shim_deepspeed_if_absent():
    """Let `tinyllava.model` import on a CPU box that has no DeepSpeed.

    `tinyllava/model/__init__.py` -> ... -> `tinyllava/utils/train_utils.py` does
    `from deepspeed import zero` at module scope. DeepSpeed is a TRAINING-only
    dependency (it is used solely by `maybe_zero_3`, for gathering ZeRO-3
    sharded params when saving a checkpoint) and does not install on macOS.
    Nothing in the vision-tower construction or forward path touches it.

    So: if and only if DeepSpeed is genuinely not installed, register empty
    placeholder modules carrying the two names the import statement binds. This
    unblocks an IMPORT. It does NOT stand in for any part of the encoder — the
    real BulkFormer-93M checkpoint, the real interaction graph, the real
    BulkFormerVisionTower class and its real forward all still run below. If
    DeepSpeed *is* installed, this function does nothing.

    Returns True if a shim was installed (recorded in the manifest).
    """
    import importlib.util
    import sys
    import types
    if importlib.util.find_spec("deepspeed") is not None:
        return False
    ds = types.ModuleType("deepspeed")
    zero = types.ModuleType("deepspeed.zero")
    runtime = types.ModuleType("deepspeed.runtime")
    rzero = types.ModuleType("deepspeed.runtime.zero")
    part = types.ModuleType("deepspeed.runtime.zero.partition_parameters")

    def _unavailable(*a, **k):
        raise RuntimeError("DeepSpeed is not installed; this is a CPU-only "
                           "cache-build process and must never reach ZeRO code.")

    zero.GatheredParameters = _unavailable

    class ZeroParamStatus:  # noqa: D401 - placeholder enum stand-in
        NOT_AVAILABLE = "NOT_AVAILABLE"

    part.ZeroParamStatus = ZeroParamStatus
    ds.zero, ds.runtime = zero, runtime
    runtime.zero, rzero.partition_parameters = rzero, part
    # A ModuleSpec is required or `importlib.util.find_spec` raises on these
    # (accelerate probes for deepspeed that way). With a spec present, its
    # probe falls through to `importlib.metadata.metadata("deepspeed")`, which
    # correctly raises PackageNotFoundError — so accelerate still concludes
    # DeepSpeed is unavailable and takes no ZeRO code path.
    import importlib.machinery
    for name, mod in (("deepspeed", ds), ("deepspeed.zero", zero),
                      ("deepspeed.runtime", runtime), ("deepspeed.runtime.zero", rzero),
                      ("deepspeed.runtime.zero.partition_parameters", part)):
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = mod
    print("[verify] deepspeed absent -> installed an import-only placeholder "
          "(training-only dependency, unused by the tower)")
    return True


def build_tower(cfg_dir: Path):
    """Construct the SAME tower class the training path uses, on CPU."""
    import torch
    from transformers import AutoConfig
    shimmed = _shim_deepspeed_if_absent()
    from tinyllava.model.vision_tower.bulkformer import BulkFormerVisionTower

    cfg = AutoConfig.from_pretrained(str(cfg_dir))
    cfg = getattr(cfg, "vision_config", cfg)
    tower = BulkFormerVisionTower(cfg)
    tower.to(torch.device("cpu"))
    tower.eval()
    return tower, cfg, shimmed


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default=str(REPO / "linear_probe/embeddings/embeddings_BulkFormer-93M.parquet"))
    ap.add_argument("--sample-index",
                    default=str(REPO / "qa_generation/bulkformer_input/bulkformer_sample_index.npy"))
    ap.add_argument("--raw", default=str(REPO / "data/cvd_transcriptome/embeddings"),
                    help="dir of raw [20010] .npy vectors (verification only)")
    ap.add_argument("--dest", default=str(REPO / "data/cvd_transcriptome/embeddings_encoded"))
    ap.add_argument("--manifest",
                    default=str(REPO / "data/cvd_transcriptome/encoded_cache_manifest.json"))
    ap.add_argument("--stage2-json",
                    default=str(REPO / "data/cvd_transcriptome/text_files/stage2_train.json"))
    ap.add_argument("--cfg", default=str(DEFAULT_CFG))
    ap.add_argument("--population", choices=("stage2", "union"), default="stage2",
                    help="stage2: the 8,553 Stage-2 positives in --sample-index. "
                         "union: every is_positive + every is_neg_hard sample in "
                         "--labels (cache coverage, NOT training membership).")
    ap.add_argument("--labels",
                    default=str(REPO / "linear_probe/probe_sample_labels.parquet"),
                    help="label frame defining is_positive / is_neg_hard (union only)")
    ap.add_argument("--holdout",
                    default=str(REPO / "data/cvd_transcriptome/holdout_series.json"),
                    help="holdout series list, for reporting cache-vs-holdout overlap")
    ap.add_argument("--verify", type=int, default=8,
                    help="live-forward this many raw vectors through the real tower "
                         "and compare against the written cache (0 disables)")
    ap.add_argument("--verify-offsets", default="0,4000",
                    help="comma-separated start offsets into the population; one "
                         "disjoint verification group is run per offset")
    args = ap.parse_args(argv)

    parquet, dest = Path(args.parquet), Path(args.dest)
    raw_dir = Path(args.raw)

    # ---- the encoder scale is locked project-wide; assert it, don't assume ----
    cfg_json = json.loads((Path(args.cfg) / "config.json").read_text())
    variant = cfg_json.get("bulkformer_variant")
    hidden = cfg_json.get("hidden_size")
    print(f"[cfg] {args.cfg}: variant={variant} hidden_size={hidden}")
    assert variant == "BulkFormer-93M", f"encoder scale is locked to BulkFormer-93M, got {variant!r}"
    assert hidden == 515, f"hidden_size must be 515 for BulkFormer-93M, got {hidden!r}"
    assert "93M" in parquet.name, f"parquet {parquet.name} is not the 93M embedding table"

    # ---- population -------------------------------------------------------
    t0 = time.time()
    X, meta = load_population(parquet)
    print(f"[parquet] {X.shape[0]} rows x {X.shape[1]} dims  ({time.time()-t0:.1f}s)")
    assert X.shape[1] == hidden, f"parquet embedding dim {X.shape[1]} != hidden_size {hidden}"

    by_index = {int(s): i for i, s in enumerate(meta["sample_index"])}

    labels = None
    if args.population == "stage2":
        wanted = [int(s) for s in np.load(args.sample_index)]
        print(f"[index] {len(wanted)} Stage-2 sample_index values")
    else:
        import pyarrow.parquet as pq
        lab = pq.read_table(args.labels).to_pydict()
        labels = {int(si): (bool(p), bool(nh), sid) for si, p, nh, sid in zip(
            lab["sample_index"], lab["is_positive"], lab["is_neg_hard"],
            lab["series_id"])}
        pos = sorted(si for si, (p, nh, _) in labels.items() if p)
        neg = sorted(si for si, (p, nh, _) in labels.items() if nh)
        both = set(pos) & set(neg)
        assert not both, f"{len(both)} samples are both is_positive and is_neg_hard"
        wanted = sorted(set(pos) | set(neg))
        print(f"[labels] {args.labels}: {len(lab['sample_index'])} rows -> "
              f"{len(pos)} is_positive + {len(neg)} is_neg_hard "
              f"= {len(wanted)} union sample_index values")
        # The embedding parquet carries its own copy of these flags. Cross-check
        # rather than trust one of the two silently.
        emb_flag_disagree = sum(
            1 for si, (p, nh, _) in labels.items()
            if si in by_index and (
                bool(meta["is_positive"][by_index[si]]) != p or
                bool(meta["is_neg_hard"][by_index[si]]) != nh))
        print(f"[labels] flag disagreements vs the embedding parquet: {emb_flag_disagree}")

    missing = [int(s) for s in wanted if int(s) not in by_index]
    if missing:
        print(f"[index] !! {len(missing)} sample_index values absent from the parquet: {missing[:20]}")

    rows = [by_index[int(s)] for s in wanted if int(s) in by_index]
    accs = [meta["geo_accession"][r] for r in rows]
    dupes = sorted({a for a in accs if accs.count(a) > 1}) if len(set(accs)) != len(accs) else []
    assert not dupes, f"duplicate geo_accession in the population: {dupes[:10]}"

    # ---- filenames must match the `image` field in stage2_train.json ------
    stage2 = json.loads(Path(args.stage2_json).read_text())
    imgs = {rec["image"] for rec in stage2}
    names = {f"{a}.npy" for a in accs}
    uncovered = sorted(imgs - names)
    print(f"[stage2] {len(stage2)} records, {len(imgs)} unique image names "
          f"(e.g. {sorted(imgs)[0]}); uncovered by this cache: {len(uncovered)}")
    assert not uncovered, f"stage2_train.json references images not in the cache: {uncovered[:10]}"

    # ---- write (skip-if-identical; never silently overwrite) ---------------
    #
    # Every file already in the cache was derived from these same parquet rows,
    # so re-deriving it must reproduce it byte for byte. Compare rather than
    # assume: an existing file that does NOT match is left on disk untouched and
    # reported, because a mismatch means one of the two is wrong and clobbering
    # it would destroy the evidence.
    dest.mkdir(parents=True, exist_ok=True)
    n_written, n_skipped_identical = 0, 0
    mismatched, nonfinite = [], []
    total_bytes = 0
    for r, acc in zip(rows, accs):
        vec = np.ascontiguousarray(X[r], dtype=np.float32)
        assert vec.shape == (hidden,), vec.shape
        if not np.isfinite(vec).all():
            nonfinite.append(acc)
        p = dest / f"{acc}.npy"
        if p.exists():
            old = np.load(p)
            if (old.dtype == vec.dtype and old.shape == vec.shape
                    and old.tobytes() == vec.tobytes()):
                n_skipped_identical += 1
            else:
                mismatched.append({
                    "geo_accession": acc,
                    "existing_dtype": str(old.dtype), "existing_shape": list(old.shape),
                    "max_abs_diff": (float(np.abs(old.astype(np.float64) - vec).max())
                                     if old.shape == vec.shape else None),
                })
                continue  # left untouched on purpose
        else:
            np.save(p, vec)
            n_written += 1
        total_bytes += p.stat().st_size
    print(f"[write] newly written {n_written}, already present and byte-identical "
          f"{n_skipped_identical}, MISMATCHED (left untouched) {len(mismatched)}")
    print(f"[write] {total_bytes} bytes ({total_bytes/1e6:.1f} MB) -> {dest}")
    if mismatched:
        print(f"[write] !! {len(mismatched)} existing files disagree with the parquet: "
              f"{[m['geo_accession'] for m in mismatched[:10]]}")
    if nonfinite:
        print(f"[write] !! {len(nonfinite)} vectors contain NaN/Inf: {nonfinite[:10]}")

    # ---- class + holdout accounting (cache coverage, NOT training membership)
    holdout_series = set(json.loads(Path(args.holdout).read_text())["holdout_series"])
    n_pos = n_neg = n_hold_pos = n_hold_neg = 0
    if labels is not None:
        for s in wanted:
            if int(s) not in by_index:
                continue
            is_pos, is_neg, sid = labels[int(s)]
            in_hold = sid in holdout_series
            n_pos += is_pos
            n_neg += is_neg
            n_hold_pos += is_pos and in_hold
            n_hold_neg += is_neg and in_hold
        print(f"[population] positives={n_pos} negatives={n_neg}; of those, in a "
              f"holdout series: {n_hold_pos} positive + {n_hold_neg} negative "
              f"(present as EVALUATION inputs; the training split is enforced "
              f"separately by {Path(args.holdout).name})")

    # ---- every file in the cache: dtype / shape / finiteness ---------------
    bad_files, cache_files = [], sorted(dest.glob("*.npy"))
    cache_bytes = 0
    for p in cache_files:
        cache_bytes += p.stat().st_size
        a = np.load(p)
        if a.dtype != np.float32 or a.shape != (hidden,) or not np.isfinite(a).all():
            bad_files.append({"file": p.name, "dtype": str(a.dtype),
                              "shape": list(a.shape),
                              "finite": bool(np.isfinite(a).all())})
    print(f"[audit] {len(cache_files)} files in cache, {cache_bytes} bytes "
          f"({cache_bytes/1e6:.1f} MB); dtype/shape/finiteness failures: {len(bad_files)}")
    if bad_files:
        print(f"[audit] !! {bad_files[:10]}")

    # ---- verification against a LIVE forward of the real tower ------------
    #
    # A live forward needs the raw [20010] expression vector. Those were only
    # ever materialized for the Stage-2 positives, so under --population union
    # the verifiable pool is a strict subset of the cache. What IS verified is
    # the parquet->cache code path itself, which is byte-for-byte the same code
    # for a negative as for a positive (same load_population, same X[r] slice,
    # same np.save). What is NOT verified for negatives is the upstream claim
    # that the parquet row equals a live tower forward for THAT sample. Say so
    # in the manifest; do not let the positives' pass be read as covering them.
    verify_pool = [a for a in accs if (raw_dir / f"{a}.npy").exists()]
    n_no_raw = len(accs) - len(verify_pool)
    print(f"[verify] raw [20010] vectors available for {len(verify_pool)}/{len(accs)} "
          f"cache members ({n_no_raw} have none and cannot be live-forwarded)")
    verification = {"attempted": bool(args.verify), "ok": None}
    if args.verify and not verify_pool:
        print("[verify] !! no raw vectors at all — nothing can be live-forwarded")
        verification = {"attempted": True, "ok": False,
                        "error": "no raw [20010] vectors available",
                        "note": "equivalence unverified"}
    elif args.verify:
        try:
            import torch
            tower, cfg, shimmed = build_tower(Path(args.cfg))
            assert tower._variant == "BulkFormer-93M", tower._variant
            assert tower.embed_dim == hidden == cfg.hidden_size
            print(f"[verify] tower variant={tower._variant} embed_dim={tower.embed_dim} device=cpu")

            def encode(names, bs):
                outs = []
                for i in range(0, len(names), bs):
                    b = np.stack([np.load(raw_dir / f"{a}.npy") for a in names[i:i + bs]])
                    assert b.shape[1] == 20010, b.shape
                    with torch.no_grad():
                        outs.append(tower(torch.from_numpy(b)).squeeze(1).float().numpy())
                return np.concatenate(outs)

            groups, worst = [], 0.0
            offsets = [int(o) for o in str(args.verify_offsets).split(",") if o.strip() != ""]
            for off in offsets:
                # Disjoint groups drawn from different parts of the population, so a
                # pass cannot be an artifact of one lucky contiguous block.
                off = off % len(verify_pool)
                picks = (verify_pool + verify_pool)[
                    off:off + min(args.verify, len(verify_pool))]
                n = len(picks)
                print(f"[verify] --- group offset={off} n={n} ---")
                t1 = time.time()
                live = encode(picks, n)
                print(f"[verify] live forward (batch={n}) took {time.time()-t1:.1f}s")
                assert live.shape == (n, hidden), live.shape
                cached = np.stack([np.load(dest / f"{a}.npy") for a in picks])

                max_abs = float(np.abs(live - cached).max())
                mean_abs = float(np.abs(cached).mean())
                # Per-sample forward: bounds how much of any residual is merely
                # batch-composition float drift the LIVE path shows against itself.
                live1 = encode(picks[:2], 1)
                d_bs = float(np.abs(live[:2] - live1).max())
                # CONTROL: same live output vs a rolled cache. Must be LARGE, else
                # the comparison is degenerate (e.g. comparing a thing to itself).
                d_ctrl = float(np.abs(live - np.roll(cached, 1, axis=0)).max())

                worst = max(worst, max_abs)
                print(f"[verify] samples            : {picks}")
                print(f"[verify] mean_abs_value     : {mean_abs:.6f}")
                print(f"[verify] max_abs_diff       : {max_abs:.3e}  "
                      f"(relative {max_abs/max(mean_abs,1e-12):.2e})")
                print(f"[verify] live bs=1 vs bs={n}  : {d_bs:.3e}  (live-vs-live baseline)")
                print(f"[verify] CONTROL rolled     : {d_ctrl:.3e}  (must be large)")
                assert d_ctrl > 100 * TOLERANCE, (
                    f"control diff {d_ctrl} is not large — the comparison has no "
                    "discriminating power and the verification is meaningless")
                groups.append({
                    "offset": off, "n": n, "samples": picks,
                    "max_abs_diff": max_abs, "mean_abs_value": mean_abs,
                    "relative_diff": max_abs / max(mean_abs, 1e-12),
                    "live_batch1_vs_batchN": d_bs,
                    "control_rolled_cache_max_abs_diff": d_ctrl,
                    "batch_size": n,
                })

            passed = worst <= TOLERANCE
            print(f"[verify] threshold          : {TOLERANCE:.1e}")
            print(f"[verify] worst max_abs_diff : {worst:.3e}")
            print(f"[verify] {'PASSED' if passed else 'FAILED'}")
            if not passed:
                print("[verify] !! The parquet vectors are NOT equivalent to the tower "
                      "output. The parquet shortcut is INVALID; rebuild the cache with "
                      "integration/precompute_encoder_cache.py on a GPU.")
            verification = {
                "attempted": True, "ok": passed, "device": "cpu",
                "worst_max_abs_diff": worst, "groups": groups,
                "deepspeed_import_shim": shimmed,
                "verifiable_pool": {
                    "n_cache_members": len(accs),
                    "n_with_raw_20010_vectors": len(verify_pool),
                    "n_without_raw_vectors": n_no_raw,
                    "raw_dir": str(raw_dir),
                },
                "note": ("live forward of the real BulkFormerVisionTower on CPU over "
                         "the raw [20010] vectors, on disjoint sample groups, with a "
                         "rolled-cache control proving the comparison discriminates" +
                         (" (deepspeed absent: import-only placeholder used; the "
                          "encoder itself is fully real)" if shimmed else "")),
                "scope_limitation": (
                    f"{n_no_raw} of {len(accs)} cache members have no raw [20010] "
                    f"vector on disk ({raw_dir}), so NO live tower forward was run "
                    "for them — under --population union these are the is_neg_hard "
                    "negatives and the 172 positives never materialized for Stage 2. "
                    "The live-forward check therefore covers only samples that have "
                    "raw vectors. It does still exercise the exact parquet->cache "
                    "code path used for every sample (same load_population, same row "
                    "slice, same np.save), so the write path is verified for all; "
                    "what is NOT independently verified for the unraw'd samples is "
                    "that their parquet row equals a live tower forward of that "
                    "sample's expression vector."
                ) if n_no_raw else "every cache member had a raw vector available",
            }
        except Exception as e:  # noqa: BLE001 — report the failure, never mask it
            import traceback
            traceback.print_exc()
            print(f"[verify] !! could not instantiate/run the real tower on CPU: "
                  f"{type(e).__name__}: {e}")
            print("[verify] !! EQUIVALENCE IS UNVERIFIED — the cache was written but the "
                  "parquet-vs-tower claim is NOT proven.")
            verification = {"attempted": True, "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                            "note": "tower could not be run on CPU; equivalence unverified"}

    passed = (bool(verification.get("ok")) and not missing
              and not mismatched and not nonfinite and not bad_files)
    manifest = {
        "purpose": "Pre-encoded 515-d BulkFormer-93M vectors for Stage-2 training, "
                   "derived from the linear-probe embedding parquet instead of a GPU "
                   "encoder pass (equivalent computation; see module docstring).",
        "built_by": "integration/build_encoded_cache_from_parquet.py",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "population": args.population,
        "source_parquet": str(Path(args.parquet)),
        "source_sample_index": str(Path(args.sample_index)),
        "encoder": {"variant": variant, "hidden_size": hidden,
                    "config_dir": str(Path(args.cfg)),
                    "checkpoint": "bulkencoders/checkpoints/bulkformer/models/BulkFormer-93M.pt"},
        "dest": str(dest),
        "filename_convention": "<GEO_ACCESSION>.npy, float32 (515,)",
        "n_requested": int(len(wanted)),
        "n_resolved_to_parquet_rows": len(rows),
        "n_newly_written": n_written,
        "n_already_present_byte_identical": n_skipped_identical,
        "n_mismatched_left_untouched": len(mismatched),
        "mismatched": mismatched[:50],
        "bytes_for_this_population": total_bytes,
        "cache_total_files": len(cache_files),
        "cache_total_bytes": cache_bytes,
        "missing_from_parquet": missing,
        "geo_accession_duplicates": dupes,
        "stage2_json": str(Path(args.stage2_json)),
        "stage2_images_uncovered": uncovered,
        "acceptance_threshold_max_abs_diff": TOLERANCE,
        "verification": verification,
        "passed": passed,
    }
    if labels is not None:
        manifest["source_labels"] = str(Path(args.labels))
        manifest["source_holdout"] = str(Path(args.holdout))
        manifest["class_composition"] = {
            "n_is_positive": int(n_pos),
            "n_is_neg_hard": int(n_neg),
            "n_union": int(n_pos + n_neg),
            "flag_disagreements_vs_embedding_parquet": int(emb_flag_disagree),
        }
        manifest["holdout_membership"] = {
            "holdout_series_file": str(Path(args.holdout)),
            "n_holdout_series": len(holdout_series),
            "n_cached_holdout_positive": int(n_hold_pos),
            "n_cached_holdout_neg_hard": int(n_hold_neg),
            "n_cached_holdout_total": int(n_hold_pos + n_hold_neg),
            "IMPORTANT": (
                "Cache membership is NOT training membership. The holdout samples "
                "are cached on purpose: they are the INPUTS to the held-out binary "
                "CVD evaluation, which cannot run without their vectors. Nothing "
                "about a sample being in this cache admits it to training. The "
                "train/eval separation is enforced upstream, at data-generation "
                "time, by holdout_series.json — a leak would be a training-JSON "
                "containing a holdout accession, not a cache file existing."),
        }
        manifest["file_integrity_audit"] = {
            "checked": len(cache_files),
            "requirement": "dtype float32, shape (515,), all finite",
            "failures": len(bad_files),
            "failing_files": bad_files[:50],
            "nonfinite_source_rows": nonfinite[:50],
        }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] wrote {args.manifest}  passed={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
