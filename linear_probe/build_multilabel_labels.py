#!/usr/bin/env python3
"""Build linear_probe/multilabel_labels.parquet + companion manifest JSON.

Deliverable 1 of workstream B2-local. Deterministic, no model fitting.
"""
from __future__ import annotations

import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/Users/harmeetsingh/UofA/TinyLLaVA_Factory")
sys.path.insert(0, str(ROOT / "linear_probe"))
from extract import _decode_h5_bytes  # noqa: E402  (reuse existing decoder)
from labels import SC_PROB_MAX        # noqa: E402  (do not guess the threshold)

H5_PATH = ROOT / "eda/dataset/cvd_data/archs4/human_gene_v2.latest.h5"
MANIFEST_PATH = ROOT / "linear_probe/embeddings/sample_manifest.parquet"
EDA_LABELS = ROOT / "eda/dataset/cvd_data/extended_eda_out/labels/sample_labels.parquet"
OUT_PARQUET = ROOT / "linear_probe/multilabel_labels.parquet"
OUT_JSON = ROOT / "linear_probe/multilabel_labels_manifest.json"

MIN_SUPPORT = 200
OTHER = "__other__"
MISSING = "__missing__"

_PUNCT = re.compile(r"[^a-z0-9]+")


def normalize(s: str) -> str:
    """lowercase, strip, collapse whitespace + punctuation to single spaces."""
    if s is None:
        return MISSING
    t = _PUNCT.sub(" ", str(s).lower()).strip()
    return t if t else MISSING


def collapse_rare(series: pd.Series, min_support: int = MIN_SUPPORT):
    """Everything with support < min_support becomes __other__.

    __missing__ is treated like any other class (it is subject to the same
    rule); its raw count is reported separately in the manifest.
    """
    vc = series.value_counts()
    keep = set(vc[vc >= min_support].index)
    out = series.where(series.isin(keep), OTHER)
    dropped = [c for c in vc.index if c not in keep]
    return out, keep, dropped, vc


def main() -> None:
    report: "OrderedDict[str, dict]" = OrderedDict()

    man = pd.read_parquet(MANIFEST_PATH)
    n = len(man)
    print(f"manifest rows: {n}  unique sample_index: {man.sample_index.nunique()}")
    assert man.sample_index.is_unique, "sample_index not unique in manifest"

    idx = man["sample_index"].to_numpy(dtype=np.int64)

    out = pd.DataFrame({
        "sample_index": idx,
        "geo_accession": man["geo_accession"].to_numpy(),
        "series_id": man["series_id"].to_numpy(),
    })

    # ---------- H5-derived fields --------------------------------------
    with h5py.File(H5_PATH, "r") as f:
        n_h5 = f["meta/samples/geo_accession"].shape[0]
        print(f"H5 sample axis: {n_h5}")
        assert idx.max() < n_h5, "sample_index out of range for H5"

        def pull(field: str) -> np.ndarray:
            arr = f[f"meta/samples/{field}"][:]
            return np.asarray(_decode_h5_bytes(arr[idx]), dtype=object)

        raw_tissue = pull("source_name_ch1")
        raw_plat = pull("platform_id")
        raw_inst = pull("instrument_model")
        h5_gsm = pull("geo_accession")
        sc_prob = f["meta/samples/singlecellprobability"][:][idx]

    # sanity: does the H5 accession agree with the manifest accession?
    mismatch = int((h5_gsm != out["geo_accession"].to_numpy()).sum())
    print(f"geo_accession H5-vs-manifest mismatches: {mismatch}")
    report["_alignment"] = {
        "manifest_rows": int(n),
        "h5_sample_axis": int(n_h5),
        "geo_accession_mismatches_h5_vs_manifest": mismatch,
    }

    # ---------- 1. tissue ----------------------------------------------
    tissue_norm = pd.Series([normalize(x) for x in raw_tissue], name="tissue")
    n_missing_tissue = int((tissue_norm == MISSING).sum())
    tissue, keep, dropped, vc = collapse_rare(tissue_norm)
    out["tissue"] = tissue.to_numpy()
    report["tissue"] = {
        "source_field": "H5 meta/samples/source_name_ch1",
        "kind": "scientific",
        "note": ("free-text sample source; normalized (lowercase, punctuation and "
                 "whitespace collapsed). This is the confound the project worries "
                 "about: strong tissue recovery is informative, not a failure."),
        "n_raw_distinct": int(pd.Series(raw_tissue).nunique()),
        "n_normalized_distinct": int(tissue_norm.nunique()),
        "min_support": MIN_SUPPORT,
        "n_classes_kept": len(keep),
        "n_classes_collapsed_to_other": len(dropped),
        "n_samples_in_other": int((out["tissue"] == OTHER).sum()),
        "n_samples_missing_raw": n_missing_tissue,
        "n_rows_without_label": 0,
        "class_support": out["tissue"].value_counts().to_dict(),
    }

    # ---------- 2. disease_category ------------------------------------
    eda = pd.read_parquet(EDA_LABELS, columns=["sample_index", "disease_category"])
    eda = eda.drop_duplicates("sample_index")
    merged = out[["sample_index"]].merge(eda, on="sample_index", how="left")
    assert len(merged) == n, f"join changed row count: {len(merged)} vs {n}"
    n_unjoined = int(merged["disease_category"].isna().sum())
    print(f"disease_category rows with no join match: {n_unjoined}")
    dc_norm = pd.Series([MISSING if (x is None or (isinstance(x, float) and np.isnan(x)))
                         else normalize(x) for x in merged["disease_category"]])
    n_missing_dc = int((dc_norm == MISSING).sum())
    dc, keep, dropped, vc = collapse_rare(dc_norm)
    out["disease_category"] = dc.to_numpy()
    report["disease_category"] = {
        "source_field": ("eda/dataset/cvd_data/extended_eda_out/labels/"
                         "sample_labels.parquet :: disease_category (joined on sample_index)"),
        "kind": "scientific",
        "n_normalized_distinct": int(dc_norm.nunique()),
        "min_support": MIN_SUPPORT,
        "n_classes_kept": len(keep),
        "n_classes_collapsed_to_other": len(dropped),
        "n_samples_in_other": int((out["disease_category"] == OTHER).sum()),
        "n_rows_without_label": n_unjoined,
        "n_samples_missing_or_empty": n_missing_dc,
        "class_support": out["disease_category"].value_counts().to_dict(),
    }

    # ---------- 3. cvd_subtype -----------------------------------------
    cs_norm = pd.Series([normalize(x) for x in man["cvd_subtype"].to_numpy()])
    n_missing_cs = int((cs_norm == MISSING).sum())
    cs, keep, dropped, vc = collapse_rare(cs_norm)
    out["cvd_subtype"] = cs.to_numpy()
    report["cvd_subtype"] = {
        "source_field": "linear_probe/embeddings/sample_manifest.parquet :: cvd_subtype",
        "kind": "scientific",
        "note": ("empty cvd_subtype (the whole-corpus negative pool) is encoded as "
                 "__missing__, which is a real and meaningful class here, not a defect."),
        "n_normalized_distinct": int(cs_norm.nunique()),
        "min_support": MIN_SUPPORT,
        "n_classes_kept": len(keep),
        "n_classes_collapsed_to_other": len(dropped),
        "n_samples_in_other": int((out["cvd_subtype"] == OTHER).sum()),
        "n_rows_without_label": 0,
        "n_samples_empty_string_raw": n_missing_cs,
        "class_support": out["cvd_subtype"].value_counts().to_dict(),
    }

    # ---------- 4. platform / instrument (TECHNICAL CONTROLS) -----------
    for col, raw, field, desc in [
        ("platform", raw_plat, "H5 meta/samples/platform_id", "GEO platform accession"),
        ("instrument", raw_inst, "H5 meta/samples/instrument_model", "sequencer model"),
    ]:
        norm = pd.Series([normalize(x) for x in raw])
        n_miss = int((norm == MISSING).sum())
        vals, keep, dropped, vc = collapse_rare(norm)
        out[col] = vals.to_numpy()
        report[col] = {
            "source_field": field,
            "kind": "technical_control",
            "note": ("TECHNICAL CONTROL, not a scientific result. Probe AUC here "
                     "measures how much purely technical batch signal the "
                     "representation carries. Never report as a scientific finding."),
            "description": desc,
            "n_normalized_distinct": int(norm.nunique()),
            "min_support": MIN_SUPPORT,
            "n_classes_kept": len(keep),
            "n_classes_collapsed_to_other": len(dropped),
            "n_samples_in_other": int((out[col] == OTHER).sum()),
            "n_rows_without_label": n_miss,
            "class_support": out[col].value_counts().to_dict(),
        }

    # ---------- 5. is_bulk ----------------------------------------------
    n_nan_sc = int(np.isnan(sc_prob).sum())
    out["singlecellprobability"] = sc_prob
    out["is_bulk"] = sc_prob < SC_PROB_MAX
    report["is_bulk"] = {
        "source_field": "H5 meta/samples/singlecellprobability",
        "kind": "scientific",
        "rule": f"singlecellprobability < {SC_PROB_MAX} "
                f"(SC_PROB_MAX imported from linear_probe/labels.py)",
        "n_classes_kept": 2,
        "n_classes_collapsed_to_other": 0,
        "n_rows_without_label": n_nan_sc,
        "class_support": {str(k): int(v) for k, v in
                          out["is_bulk"].value_counts().to_dict().items()},
    }

    # flag degenerate (single-valued) labels so nobody tries to probe them
    for k, v in report.items():
        if k.startswith("_"):
            continue
        n_present = len(v["class_support"])
        v["n_classes_present_in_column"] = n_present
        v["degenerate_single_class"] = bool(n_present < 2)
        if v["degenerate_single_class"]:
            v["warning"] = ("Only one class present across the probe population; "
                            "this label is NOT probeable and must be skipped.")

    # ---------- write ----------------------------------------------------
    cols = ["sample_index", "geo_accession", "series_id", "tissue",
            "disease_category", "cvd_subtype", "platform", "instrument",
            "is_bulk", "singlecellprobability"]
    out = out[cols]
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET} shape={out.shape}")

    meta = {
        "purpose": ("Broad multi-label probe label table for the LLM-latent "
                    "representation-quality comparison (retrain plan Phase 1c/4c). "
                    "Consumed by both the before-fix and after-fix probe runs."),
        "population": str(MANIFEST_PATH),
        "n_rows": int(n),
        "cv_grouping_key": "series_id (geo_accession also carried)",
        "rare_class_rule": f"classes with support < {MIN_SUPPORT} collapsed to '{OTHER}'",
        "sentinels": {"other": OTHER, "missing": MISSING},
        "scientific_labels": [k for k, v in report.items()
                              if v.get("kind") == "scientific"],
        "technical_controls": [k for k, v in report.items()
                               if v.get("kind") == "technical_control"],
        "labels": report,
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2, default=str))
    print(f"wrote {OUT_JSON}")

    for k, v in report.items():
        if k.startswith("_"):
            continue
        print(f"\n== {k} [{v.get('kind')}] kept={v.get('n_classes_kept')} "
              f"collapsed={v.get('n_classes_collapsed_to_other')} "
              f"no_label={v.get('n_rows_without_label')}")
        for c, cnt in list(v["class_support"].items())[:12]:
            print(f"   {c}: {cnt}")


if __name__ == "__main__":
    main()
