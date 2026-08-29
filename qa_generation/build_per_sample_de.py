"""build_per_sample_de.py — workstream B1: REAL per-sample differential expression.

Why this exists
---------------
Stage-2 training answers were degenerate: 86.4% of them were a fixed string,
identical across all 8,553 samples, because
`qa_generation/gt_functions.py::comparative_differential_reasoning` returned a
corpus-level linear-probe ROC-AUC summary (`per_gene_differential: None`) and
`gene_driver_reasoning` returned the same global elastic-net gene ranking for
every sample (self-documented as `sample_role: "eligibility_gate_only"`). A
model trained on that learns to ignore the expression profile entirely.

This script produces the genuinely per-sample quantity those functions should
have been consuming: for each disease-confirmed positive sample, how ITS OWN
transcriptome deviates from a tissue-matched `neg_hard` comparison population.

DATA ONLY. The GT functions are rewritten against this output by another
workstream, so the output schema below is a contract.

Normalization is NOT reimplemented here: `linear_probe.extract` owns the exact,
audited TPM -> log1p -> BulkFormer-vocab pipeline that the positives matrix was
built with, and it is imported and called. The positives manifest's
`"reimplemented": false` property therefore stays true.

Outputs (all under qa_generation/de/):
    per_sample_de.parquet   one row per positive sample
    stable_gene_z.npz       per-sample z/lfc for the stable elastic-net signal genes
    de_reference_stats.npz  per-bucket mu/sd + gene mask (auditable, rerunnable)
    de_manifest.json        timings, populations, exclusions, sanity checks

CLI
---
    python3 qa_generation/build_per_sample_de.py            # full run
    python3 qa_generation/build_per_sample_de.py --limit-ref 512   # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

# MANDATORY reuse — the audited normalization, not a reimplementation.
from linear_probe.extract import (  # noqa: E402
    ArchS4CountReader,
    load_bulkformer_vocab,
    normalize_and_align,
)

H5_PATH = REPO / "eda" / "dataset" / "cvd_data" / "archs4" / "human_gene_v2.latest.h5"
LABELS_PATH = REPO / "linear_probe" / "probe_sample_labels.parquet"
POS_EXPR_PATH = HERE / "bulkformer_input" / "bulkformer_expression.npy"
POS_INDEX_PATH = HERE / "bulkformer_input" / "bulkformer_sample_index.npy"
SYMBOL_MAP_PATH = HERE / "bulkformer_input" / "symbol_vocab_map.parquet"
HOLDOUT_PATH = REPO / "data" / "cvd_transcriptome" / "holdout_series.json"
GENE_SIGNAL_PATH = (REPO / "eda" / "dataset" / "cvd_data" / "elasticnet_out"
                    / "gene_signal" / "gene_signal_ranking.csv")
OUT_DIR = HERE / "de"

SENTINEL = -10.0            # normalize_and_align's fill for genes absent from the H5
MIN_BUCKET_N = 30           # a source_name_ch1 bucket must have >= this to be tissue-matched
SD_FLOOR = 1e-6             # reference genes below this sd give meaningless z-scores
TOP_K = 25
# Ranking gates. These filter WHICH GENES MAY BE NAMED in a top-K list; they do
# NOT touch the deviation-count statistics (n_genes_abs_z_gt2 / frac / tertiles),
# which are a count over all comparable genes and are not distorted the way a
# top-K list is.
LFC_GATE = 0.5      # |x - mu| on a log1p(TPM) scale, i.e. ~1.65x; drops the
                    # near-zero-denominator artifacts that post a huge z off a
                    # trivial fold change.
UNNAMEABLE = None   # set at runtime: vocab entries whose "symbol" is just the ENSG
# Sex-linked genes, excluded from the ranked lists by user decision. Sex is a
# covariate here, not the phenotype under study: excluding sex-linked genes from
# a DE ranking is routine when sex is not the variable of interest. Measured
# first (8.9% of samples had one at rank 1 before this gate) — the number is
# preserved in the manifest's confound_audit so the mitigation stays auditable.
SEX_LINKED_GENES = ("RPS4Y1", "DDX3Y", "UTY", "USP9Y", "KDM5D",
                    "EIF1AY", "NLGN4Y", "ZFY", "XIST", "TSIX")
# Common whole-gene germline DELETION polymorphisms: a large fraction of the
# population is null for each, so extreme expression variance reflects inherited
# copy number, not phenotype. All three carry nonzero_frac == 0.00 and
# abs_mean_coef == 0.0000 in gene_signal_ranking.csv — the elastic net gave them
# a zero coefficient in EVERY fold — so by this project's own measure they hold
# no cardiovascular signal and excluding them removes nothing disease-relevant.
# Exactly these three, nothing inferred: the gate is NOT extended to
# zero-coefficient genes generally, which would collapse the lists toward the
# 1,142 CVD genes and destroy the point of an unbiased DE comparison.
CNV_POLYMORPHISM_GENES = ("GSTM1", "GSTT1", "UGT2B17")
MIN_USABLE_GENES = 1000     # below this a sample's comparison is not worth reporting
READ_BATCH = 256


def _log() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S",
                        stream=sys.stdout)
    return logging.getLogger("build_per_sample_de")


_PUNCT = re.compile(r"[^a-z0-9]+")


def normalize_tissue(raw: str) -> str:
    """lowercase / strip / collapse whitespace+punctuation to a bucket key."""
    if raw is None:
        return ""
    return _PUNCT.sub(" ", str(raw).lower()).strip()


def _decode(arr) -> list[str]:
    return [x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in arr]


# ----- stage 1: populations -----------------------------------------------

def select_populations(logger: logging.Logger, limit_ref: int | None):
    labels = pd.read_parquet(LABELS_PATH)
    holdout_blob = json.loads(HOLDOUT_PATH.read_text())
    holdout_series = set(holdout_blob["holdout_series"])
    logger.info(f"holdout: {len(holdout_series)} series reserved for evaluation")

    neg_hard = labels.loc[labels["is_neg_hard"]].copy()
    n_neg_hard_total = len(neg_hard)
    in_hold = neg_hard["series_id"].isin(holdout_series)
    n_excluded = int(in_hold.sum())
    ref = neg_hard.loc[~in_hold].copy().sort_values("sample_index").reset_index(drop=True)
    # Non-negotiable: holdout-series negatives must not contribute to the
    # reference statistics, or holdout information leaks into TRAINING answers.
    logger.info(f"neg_hard reference: {n_neg_hard_total} total, "
                f"{n_excluded} excluded as holdout-series, {len(ref)} retained")

    if limit_ref is not None:
        ref = ref.iloc[:limit_ref].reset_index(drop=True)
        logger.warning(f"SMOKE TEST: reference truncated to {len(ref)} samples")

    pos_index = np.load(POS_INDEX_PATH)
    pos_meta = (labels.set_index("sample_index")
                      .reindex(pos_index)[["geo_accession", "series_id"]]
                      .reset_index())
    n_missing_meta = int(pos_meta["geo_accession"].isna().sum())
    logger.info(f"positives: {len(pos_meta)} rows, {n_missing_meta} with no label metadata")

    return labels, holdout_series, ref, pos_index, pos_meta, {
        "n_neg_hard_total": n_neg_hard_total,
        "n_neg_hard_excluded_holdout": n_excluded,
        "n_neg_hard_reference": len(ref),
        "n_positive_missing_label_meta": n_missing_meta,
    }


def read_source_names(sample_indices: np.ndarray, logger: logging.Logger) -> list[str]:
    """source_name_ch1 for the given sample columns, straight from the H5."""
    order = np.argsort(sample_indices)
    with h5py.File(H5_PATH, "r") as f:
        raw = f["meta/samples/source_name_ch1"][np.sort(sample_indices)]
    decoded = _decode(raw)
    out = [""] * len(sample_indices)
    for slot, pos in enumerate(order):
        out[pos] = decoded[slot]
    logger.info(f"read source_name_ch1 for {len(out)} samples")
    return out


# ----- stage 2: reference expression matrix -------------------------------

def build_reference_matrix(ref: pd.DataFrame, logger: logging.Logger):
    vocab, length_dict = load_bulkformer_vocab(logger)
    idx = ref["sample_index"].to_numpy(dtype=np.int64)
    reader = ArchS4CountReader(H5_PATH, idx, logger)

    # normalize_and_align logs one line per call; keep it quiet across ~90 batches.
    quiet = logging.getLogger("build_per_sample_de.normalize")
    quiet.setLevel(logging.WARNING)

    n = len(idx)
    mat = np.empty((n, len(vocab)), dtype=np.float32)
    mask_probs = []
    t0 = time.time()
    try:
        for start in range(0, n, READ_BATCH):
            stop = min(start + READ_BATCH, n)
            counts = reader.read_batch(idx[start:stop])
            aligned, mask_prob = normalize_and_align(
                counts, reader.h5_gene_symbols, vocab, length_dict, quiet)
            mat[start:stop] = aligned
            mask_probs.append(mask_prob)
            done = stop
            rate = done / max(time.time() - t0, 1e-9)
            eta = (n - done) / max(rate, 1e-9)
            logger.info(f"reference reads {done}/{n} ({100 * done / n:5.1f}%) "
                        f"{rate:6.1f} samples/s  eta {eta / 60:5.1f} min")
    finally:
        reader.close()

    elapsed = time.time() - t0
    logger.info(f"reference matrix {mat.shape} built in {elapsed / 60:.1f} min "
                f"({n / elapsed:.1f} samples/s)")
    return mat, vocab, float(max(mask_probs)) if mask_probs else 0.0, elapsed


# ----- stage 3: tissue buckets + reference statistics ---------------------

def build_reference_stats(mat: np.ndarray, tissues: list[str], logger: logging.Logger):
    """Per-gene mean/sd for every qualifying tissue bucket and for the whole pool."""
    keys = np.array(tissues, dtype=object)
    buckets: dict[str, np.ndarray] = {}
    for name in pd.unique(keys):
        if not name:                       # blank source_name is not a tissue claim
            continue
        rows = np.flatnonzero(keys == name)
        if len(rows) >= MIN_BUCKET_N:
            buckets[str(name)] = rows

    bucket_names = sorted(buckets)
    logger.info(f"{len(pd.unique(keys))} distinct normalized source_name values; "
                f"{len(bucket_names)} qualify as tissue-matched (n >= {MIN_BUCKET_N}), "
                f"covering {sum(len(buckets[b]) for b in bucket_names)} reference samples")

    g = mat.shape[1]
    bucket_mu = np.empty((len(bucket_names), g), dtype=np.float32)
    bucket_sd = np.empty((len(bucket_names), g), dtype=np.float32)
    bucket_n = np.empty(len(bucket_names), dtype=np.int32)
    for i, name in enumerate(bucket_names):
        rows = buckets[name]
        sub = mat[rows].astype(np.float64)
        bucket_mu[i] = sub.mean(axis=0)
        bucket_sd[i] = sub.std(axis=0)
        bucket_n[i] = len(rows)

    pool = mat.astype(np.float64)
    pool_mu = pool.mean(axis=0).astype(np.float32)
    pool_sd = pool.std(axis=0).astype(np.float32)
    pool_n = np.int32(mat.shape[0])
    del pool
    logger.info(f"pool reference statistics over {int(pool_n)} samples")
    return bucket_names, bucket_mu, bucket_sd, bucket_n, pool_mu, pool_sd, pool_n


def build_gene_mask(mat: np.ndarray, pos_expr: np.ndarray, pool_sd: np.ndarray,
                    logger: logging.Logger):
    """Genes usable in every statistic and every ranked list."""
    ref_sentinel = np.any(mat == SENTINEL, axis=0)
    pos_sentinel = np.any(pos_expr == SENTINEL, axis=0)
    sentinel = ref_sentinel | pos_sentinel
    flat = pool_sd < SD_FLOOR
    mask = ~(sentinel | flat)
    logger.info(f"gene mask: {int(mask.sum())} usable / {mask.size} vocab genes "
                f"({int(sentinel.sum())} sentinel -10.0 [ref {int(ref_sentinel.sum())}, "
                f"pos {int(pos_sentinel.sum())}], "
                f"{int((flat & ~sentinel).sum())} pool sd < {SD_FLOOR})")
    return mask, {
        "n_excluded_sentinel": int(sentinel.sum()),
        "n_excluded_sentinel_reference": int(ref_sentinel.sum()),
        "n_excluded_sentinel_positives": int(pos_sentinel.sum()),
        "n_excluded_zero_sd_pool": int((flat & ~sentinel).sum()),
        "n_genes_usable": int(mask.sum()),
    }


def build_nameable_mask(symbols: np.ndarray, ensg: np.ndarray, logger: logging.Logger):
    """Vocab entries we can actually NAME in a clinical sentence.

    symbol_vocab_map falls back to the ENSG accession when a gene has no symbol,
    so those entries carry an accession in the gene_symbol column. Naming one in
    an answer ("this patient shows elevated ENSG00000269179") is not usable
    supervision: it teaches the model to emit an accession as if it were a
    finding, and a reader cannot check it. A gene we cannot name is a gene we
    should not claim — the project's existing "must be in BulkFormer's vocab"
    rule, extended from in-vocabulary to nameable.

    They stay in the z/lfc matrices and in every statistic; only the ranked
    lists exclude them.
    """
    sym = np.array([str(x).strip() for x in symbols])
    acc = np.array([str(x).strip().upper() for x in ensg])
    nameable = ~((sym == "") | (np.char.upper(sym) == acc)
                 | np.char.startswith(np.char.upper(sym), "ENSG"))
    logger.info(f"nameable genes: {int(nameable.sum())}/{nameable.size} "
                f"({int((~nameable).sum())} vocab entries have no gene symbol)")
    return nameable


def build_symbol_set_mask(symbols: np.ndarray, gene_set: tuple[str, ...],
                          label: str, logger: logging.Logger):
    """True where the vocab entry is one of `gene_set` (matched on symbol).

    Reports which listed genes are absent from the vocab rather than silently
    matching fewer than asked: a gene not in the vocab could never have been
    named anyway, and that is worth stating rather than discovering later.
    """
    sym = np.array([str(x).strip().upper() for x in symbols])
    want = {g.upper() for g in gene_set}
    hit = np.array([x in want for x in sym])
    found = sorted({s_ for s_ in sym if s_ in want})
    missing = sorted(want - set(found))
    logger.info(f"{label} gate: {int(hit.sum())} vocab entries matched "
                f"{len(found)}/{len(want)} listed genes"
                + (f"; not in vocab: {missing}" if missing else ""))
    return hit, found, missing


# ----- stage 5: per-sample differential expression ------------------------

def compute_per_sample(pos_expr, pos_index, pos_meta, pos_tissue, holdout_series,
                       bucket_names, bucket_mu, bucket_sd, bucket_n,
                       pool_mu, pool_sd, pool_n, base_mask, symbols,
                       stable_cols, nameable, is_sex, is_cnv, logger):
    n_pos, g = pos_expr.shape
    bucket_lookup = {name: i for i, name in enumerate(bucket_names)}

    scope = np.empty(n_pos, dtype=object)
    ref_id = np.full(n_pos, -1, dtype=np.int32)      # -1 => pool fallback
    ref_n = np.empty(n_pos, dtype=np.int32)
    for i, t in enumerate(pos_tissue):
        key = normalize_tissue(t)
        bi = bucket_lookup.get(key, -1) if key else -1
        if bi >= 0:
            scope[i] = f"tissue:{key}"
            ref_id[i] = bi
            ref_n[i] = bucket_n[bi]
        else:
            scope[i] = "pool"
            ref_n[i] = pool_n
    n_tissue = int((ref_id >= 0).sum())
    logger.info(f"reference assignment: {n_tissue} tissue-matched, "
                f"{n_pos - n_tissue} pooled fallback")

    mean_abs_z = np.zeros(n_pos, dtype=np.float32)
    max_abs_z = np.zeros(n_pos, dtype=np.float32)
    n_gt2 = np.zeros(n_pos, dtype=np.int32)
    frac_gt2 = np.zeros(n_pos, dtype=np.float32)
    n_genes_used = np.zeros(n_pos, dtype=np.int32)
    status = np.array(["ok"] * n_pos, dtype=object)
    reason = np.array([None] * n_pos, dtype=object)

    up_genes = [None] * n_pos
    down_genes = [None] * n_pos
    up_z = [None] * n_pos
    down_z = [None] * n_pos
    up_lfc = [None] * n_pos
    down_lfc = [None] * n_pos

    stable_z = np.full((n_pos, len(stable_cols)), np.nan, dtype=np.float32)
    stable_lfc = np.full((n_pos, len(stable_cols)), np.nan, dtype=np.float32)
    stable_gate = np.zeros((n_pos, len(stable_cols)), dtype=bool)
    stable_gate_lfc = np.zeros((n_pos, len(stable_cols)), dtype=bool)

    comparable = np.zeros(n_pos, dtype=bool)     # reference-level success, pre-gate
    n_gate_pass = np.zeros(n_pos, dtype=np.int32)
    n_drop_lfc = np.zeros(n_pos, dtype=np.int32)
    n_drop_unnameable = np.zeros(n_pos, dtype=np.int32)
    n_drop_sex = np.zeros(n_pos, dtype=np.int32)
    n_drop_cnv = np.zeros(n_pos, dtype=np.int32)
    presex_up = [[] for _ in range(n_pos)]
    presex_down = [[] for _ in range(n_pos)]
    # ungated top-5, kept ONLY to measure what the gates changed
    ung_up5 = [None] * n_pos
    ung_down5 = [None] * n_pos

    t0 = time.time()
    groups = {}
    for i in range(n_pos):
        groups.setdefault(int(ref_id[i]), []).append(i)

    for gi, (bi, rows) in enumerate(sorted(groups.items())):
        rows = np.asarray(rows, dtype=np.int64)
        if bi < 0:
            mu, sd, this_n = pool_mu, pool_sd, int(pool_n)
        else:
            mu, sd, this_n = bucket_mu[bi], bucket_sd[bi], int(bucket_n[bi])

        # A gene flat within THIS reference has a meaningless z even if the
        # pooled sd was fine, so mask per reference and record the effect.
        mask = base_mask & (sd >= SD_FLOOR)
        valid = np.flatnonzero(mask)

        if this_n < MIN_BUCKET_N or valid.size < MIN_USABLE_GENES:
            for i in rows:
                status[i] = "insufficient_data"
                reason[i] = ("reference_population_too_small"
                             if this_n < MIN_BUCKET_N else "too_few_usable_genes")
                up_genes[i] = []; down_genes[i] = []
                up_z[i] = []; down_z[i] = []
                up_lfc[i] = []; down_lfc[i] = []
            continue

        mu_v = mu[valid].astype(np.float32)
        sd_v = sd[valid].astype(np.float32)
        nameable_v = nameable[valid]
        not_sex_v = ~is_sex[valid]
        not_cnv_v = ~is_cnv[valid]
        stable_valid_pos = np.searchsorted(valid, stable_cols)
        stable_in_valid = ((stable_valid_pos < valid.size)
                           & (valid[np.minimum(stable_valid_pos, valid.size - 1)] == stable_cols))

        for start in range(0, len(rows), 512):
            chunk = rows[start:start + 512]
            x = pos_expr[chunk][:, valid]
            # both x and mu are log1p(TPM), so their difference IS a log fold change
            lfc = x - mu_v
            z = lfc / (sd_v + 1e-6)
            az = np.abs(z)

            mean_abs_z[chunk] = az.mean(axis=1)
            max_abs_z[chunk] = az.max(axis=1)
            over = (az > 2.0).sum(axis=1)
            n_gt2[chunk] = over
            frac_gt2[chunk] = over / valid.size
            n_genes_used[chunk] = valid.size

            comparable[chunk] = True

            # Ungated top-5, purely diagnostic: lets the manifest show what the
            # gates displaced instead of asserting it.
            u5 = np.argpartition(-z, 4, axis=1)[:, :5]
            d5 = np.argpartition(z, 4, axis=1)[:, :5]

            # Selection-time gates. Rank by z as before, but only among genes
            # that clear a real fold change AND can be named.
            pass_lfc = np.abs(lfc) >= LFC_GATE
            gate_presex = pass_lfc & nameable_v           # lfc + nameable only
            gate = gate_presex & not_sex_v & not_cnv_v    # + confound exclusions
            n_gate_pass[chunk] = gate.sum(axis=1)
            n_drop_lfc[chunk] = (~pass_lfc).sum(axis=1)
            n_drop_unnameable[chunk] = (pass_lfc & ~nameable_v).sum(axis=1)
            n_drop_sex[chunk] = (gate_presex & ~not_sex_v).sum(axis=1)
            n_drop_cnv[chunk] = (gate_presex & ~not_cnv_v).sum(axis=1)
            z_up = np.where(gate, z, -np.inf)
            z_dn = np.where(gate, z, np.inf)
            # same ranking with lfc+nameable but WITHOUT either exclusion gate,
            # kept only to measure what the exclusion gates displaced — the
            # confound findings stay live measurements, not hardcoded numbers
            z_up_ps = np.where(gate_presex, z, -np.inf)
            z_dn_ps = np.where(gate_presex, z, np.inf)
            n_presex = gate_presex.sum(axis=1)

            for r, i in enumerate(chunk):
                ung_up5[i] = [symbols[valid[j]] for j in u5[r][np.argsort(-z[r, u5[r]])]]
                ung_down5[i] = [symbols[valid[j]] for j in d5[r][np.argsort(z[r, d5[r]])]]

                nps = int(n_presex[r])
                if nps > 0:
                    kp = min(TOP_K, nps)
                    hp = np.argpartition(-z_up_ps[r], kp - 1)[:kp]
                    hp = hp[np.argsort(-z_up_ps[r, hp])]
                    lp = np.argpartition(z_dn_ps[r], kp - 1)[:kp]
                    lp = lp[np.argsort(z_dn_ps[r, lp])]
                    presex_up[i] = [symbols[valid[j]] for j in hp]
                    presex_down[i] = [symbols[valid[j]] for j in lp]

                npass = int(n_gate_pass[i])
                if npass == 0:
                    # no nameable gene moved enough to be worth claiming
                    status[i] = "insufficient_data"
                    reason[i] = "no_genes_pass_ranking_gate"
                    up_genes[i] = []; down_genes[i] = []
                    up_z[i] = []; down_z[i] = []
                    up_lfc[i] = []; down_lfc[i] = []
                    continue
                k = min(TOP_K, npass)
                hi = np.argpartition(-z_up[r], k - 1)[:k]
                hi = hi[np.argsort(-z_up[r, hi])]
                lo = np.argpartition(z_dn[r], k - 1)[:k]
                lo = lo[np.argsort(z_dn[r, lo])]
                up_genes[i] = [symbols[valid[j]] for j in hi]
                down_genes[i] = [symbols[valid[j]] for j in lo]
                up_z[i] = z[r, hi].astype(np.float32)
                down_z[i] = z[r, lo].astype(np.float32)
                up_lfc[i] = lfc[r, hi].astype(np.float32)
                down_lfc[i] = lfc[r, lo].astype(np.float32)

            if stable_in_valid.any():
                cols = stable_valid_pos[stable_in_valid]
                dest = np.flatnonzero(stable_in_valid)
                stable_z[np.ix_(chunk, dest)] = z[:, cols]
                stable_lfc[np.ix_(chunk, dest)] = lfc[:, cols]
                # the consumer gates at selection time; this is that gate,
                # precomputed, so it never has to re-derive it
                stable_gate[np.ix_(chunk, dest)] = gate[:, cols]
                stable_gate_lfc[np.ix_(chunk, dest)] = pass_lfc[:, cols]

        logger.info(f"scored group {gi + 1}/{len(groups)} "
                    f"({'pool' if bi < 0 else bucket_names[bi]}): "
                    f"{len(rows)} samples, {valid.size} usable genes, "
                    f"{time.time() - t0:.1f}s elapsed")

    ok = status == "ok"
    # Tertiles of frac_genes_abs_z_gt2 computed ONCE across all samples, over
    # every COMPARABLE sample. Deliberately not over post-gate `ok`: the ranking
    # gates must not shift a deviation-count statistic, or magnitude_reasoning
    # stops being comparable across runs.
    cut_lo, cut_hi = np.quantile(frac_gt2[comparable], [1 / 3, 2 / 3])
    bucket_label = np.array([""] * n_pos, dtype=object)
    for i in range(n_pos):
        if not comparable[i]:
            # never fabricate a magnitude for a sample we could not compare
            bucket_label[i] = "unknown"
            continue
        bucket_label[i] = ("minimal" if frac_gt2[i] <= cut_lo
                           else ("moderate" if frac_gt2[i] <= cut_hi else "large"))
    logger.info(f"magnitude tertile cut points: {cut_lo:.6f} / {cut_hi:.6f}")

    in_holdout = pos_meta["series_id"].isin(holdout_series).to_numpy()

    return dict(
        scope=scope, ref_n=ref_n, status=status, reason=reason,
        mean_abs_z=mean_abs_z, max_abs_z=max_abs_z, n_gt2=n_gt2, frac_gt2=frac_gt2,
        n_genes_used=n_genes_used, bucket_label=bucket_label,
        up_genes=up_genes, down_genes=down_genes, up_z=up_z, down_z=down_z,
        up_lfc=up_lfc, down_lfc=down_lfc, stable_z=stable_z, stable_lfc=stable_lfc,
        in_holdout=in_holdout, cut_lo=float(cut_lo), cut_hi=float(cut_hi),
        n_tissue_matched=n_tissue, ref_id=ref_id, comparable=comparable,
        stable_gate=stable_gate, stable_gate_lfc=stable_gate_lfc,
        n_gate_pass=n_gate_pass, n_drop_sex=n_drop_sex, n_drop_cnv=n_drop_cnv,
        presex_up=presex_up, presex_down=presex_down,
        n_drop_lfc=n_drop_lfc, n_drop_unnameable=n_drop_unnameable,
        ung_up5=ung_up5, ung_down5=ung_down5,
    )


# ----- main ----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-ref", type=int, default=None,
                    help="smoke test: cap the reference population size")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    logger = _log()
    started = datetime.now(timezone.utc)
    t_start = time.time()
    timings = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- stage 1
    t = time.time()
    labels, holdout_series, ref, pos_index, pos_meta, pop = select_populations(logger, args.limit_ref)
    ref_tissue_raw = read_source_names(ref["sample_index"].to_numpy(np.int64), logger)
    pos_tissue_raw = read_source_names(pos_index, logger)
    ref_tissue = [normalize_tissue(t_) for t_ in ref_tissue_raw]
    timings["stage1_populations_s"] = round(time.time() - t, 1)

    # --- stage 2
    mat, vocab, max_mask_prob, read_s = build_reference_matrix(ref, logger)
    timings["stage2_reference_reads_s"] = round(read_s, 1)

    # --- stage 3
    t = time.time()
    (bucket_names, bucket_mu, bucket_sd, bucket_n,
     pool_mu, pool_sd, pool_n) = build_reference_stats(mat, ref_tissue, logger)
    timings["stage3_reference_stats_s"] = round(time.time() - t, 1)

    # --- stage 4
    t = time.time()
    pos_expr = np.load(POS_EXPR_PATH)
    logger.info(f"positives matrix {pos_expr.shape} {pos_expr.dtype}")
    base_mask, gene_excl = build_gene_mask(mat, pos_expr, pool_sd, logger)
    del mat

    symbol_map = pd.read_parquet(SYMBOL_MAP_PATH).sort_values("vocab_pos")
    symbols = symbol_map["gene_symbol"].astype(str).to_numpy()
    if len(symbols) != pos_expr.shape[1]:
        raise RuntimeError(f"symbol map covers {len(symbols)} of {pos_expr.shape[1]} vocab columns")
    sym_to_col = {}
    for s, c in zip(symbols, symbol_map["vocab_pos"].to_numpy()):
        sym_to_col.setdefault(s, int(c))

    ensg = symbol_map["ensg_id"].astype(str).to_numpy()
    nameable = build_nameable_mask(symbols, ensg, logger)
    is_sex, sex_found, sex_missing = build_symbol_set_mask(
        symbols, SEX_LINKED_GENES, "sex-linked", logger)
    is_cnv, cnv_found, cnv_missing = build_symbol_set_mask(
        symbols, CNV_POLYMORPHISM_GENES, "cnv-polymorphism", logger)

    ranking = pd.read_csv(GENE_SIGNAL_PATH)
    stable = ranking.loc[ranking["nonzero_frac"] == 1.0].copy()
    stable["rank"] = np.arange(len(stable), dtype=np.int32)
    stable["col"] = stable["gene_symbol"].astype(str).map(sym_to_col)
    n_stable_raw = len(stable)
    stable = stable.dropna(subset=["col"])
    n_stable_in_vocab = len(stable)
    stable["col"] = stable["col"].astype(int)
    stable = stable.loc[base_mask[stable["col"].to_numpy()]]
    stable = stable.drop_duplicates(subset=["col"]).sort_values("col").reset_index(drop=True)
    stable_cols = stable["col"].to_numpy(dtype=np.int64)
    logger.info(f"stable signal genes: {n_stable_raw} with nonzero_frac==1.0 -> "
                f"{n_stable_in_vocab} in BulkFormer vocab -> {len(stable)} after gene mask")
    timings["stage4_masks_and_inputs_s"] = round(time.time() - t, 1)

    # --- stage 5
    t = time.time()
    res = compute_per_sample(pos_expr, pos_index, pos_meta, pos_tissue_raw, holdout_series,
                             bucket_names, bucket_mu, bucket_sd, bucket_n,
                             pool_mu, pool_sd, pool_n, base_mask, symbols,
                             stable_cols, nameable, is_sex, is_cnv, logger)
    timings["stage5_per_sample_de_s"] = round(time.time() - t, 1)

    # --- write outputs
    t = time.time()
    f32 = pa.list_(pa.float32())
    schema = pa.schema([
        ("sample_index", pa.int64()), ("geo_accession", pa.string()),
        ("series_id", pa.string()), ("in_holdout", pa.bool_()),
        ("status", pa.string()), ("reason", pa.string()),
        ("reference_scope", pa.string()), ("reference_n", pa.int32()),
        ("mean_abs_z", pa.float32()), ("max_abs_z", pa.float32()),
        ("n_genes_abs_z_gt2", pa.int32()), ("frac_genes_abs_z_gt2", pa.float32()),
        ("n_genes_compared", pa.int32()), ("magnitude_bucket", pa.string()),
        ("top_up_genes", pa.list_(pa.string())), ("top_down_genes", pa.list_(pa.string())),
        ("top_up_z", f32), ("top_down_z", f32),
        ("top_up_lfc", f32), ("top_down_lfc", f32),
    ])
    table = pa.table({
        "sample_index": pa.array(pos_index, pa.int64()),
        "geo_accession": pa.array(pos_meta["geo_accession"].astype(object), pa.string()),
        "series_id": pa.array(pos_meta["series_id"].astype(object), pa.string()),
        "in_holdout": pa.array(res["in_holdout"], pa.bool_()),
        "status": pa.array(res["status"], pa.string()),
        "reason": pa.array(res["reason"], pa.string()),
        "reference_scope": pa.array(res["scope"], pa.string()),
        "reference_n": pa.array(res["ref_n"], pa.int32()),
        "mean_abs_z": pa.array(res["mean_abs_z"], pa.float32()),
        "max_abs_z": pa.array(res["max_abs_z"], pa.float32()),
        "n_genes_abs_z_gt2": pa.array(res["n_gt2"], pa.int32()),
        "frac_genes_abs_z_gt2": pa.array(res["frac_gt2"], pa.float32()),
        "n_genes_compared": pa.array(res["n_genes_used"], pa.int32()),
        "magnitude_bucket": pa.array(res["bucket_label"], pa.string()),
        "top_up_genes": pa.array([list(x) for x in res["up_genes"]], pa.list_(pa.string())),
        "top_down_genes": pa.array([list(x) for x in res["down_genes"]], pa.list_(pa.string())),
        "top_up_z": pa.array([list(map(float, x)) for x in res["up_z"]], f32),
        "top_down_z": pa.array([list(map(float, x)) for x in res["down_z"]], f32),
        "top_up_lfc": pa.array([list(map(float, x)) for x in res["up_lfc"]], f32),
        "top_down_lfc": pa.array([list(map(float, x)) for x in res["down_lfc"]], f32),
    }, schema=schema)
    parquet_path = args.out_dir / "per_sample_de.parquet"
    pq.write_table(table, parquet_path)
    logger.info(f"wrote {parquet_path} ({len(table)} rows)")

    direction = np.where(stable["mean_coef"].to_numpy() >= 0, "up_in_disease", "down_in_disease")
    np.savez_compressed(
        args.out_dir / "stable_gene_z.npz",
        z=res["stable_z"], lfc=res["stable_lfc"],
        sample_index=pos_index.astype(np.int64),
        genes=stable["gene_symbol"].astype(str).to_numpy(),
        vocab_pos=stable_cols.astype(np.int64),
        rank=stable["rank"].to_numpy(np.int32),
        mean_coef=stable["mean_coef"].to_numpy(np.float32),
        direction=direction,
        in_clingen_hcvd=stable["in_clingen_hcvd"].to_numpy(bool),
        # selection-time ranking gate, precomputed per (sample, stable gene):
        # |lfc| >= LFC_GATE and the gene is nameable. z/lfc keep the full
        # 1,142-gene row; the consumer gates when it RANKS, not when it reads.
        # COMBINED gate (lfc AND nameable AND not_sex_linked):
        rank_gate=res["stable_gate"],
        # ...and each component SEPARATELY, so any single gate can be revisited
        # later without recomputing the others:
        gate_lfc_pass=res["stable_gate_lfc"],      # [n_samples, n_stable] bool
        nameable=nameable[stable_cols],            # [n_stable] bool
        not_sex_linked=~is_sex[stable_cols],       # [n_stable] bool
        not_cnv_polymorphism=~is_cnv[stable_cols],  # [n_stable] bool
        lfc_gate=np.float32(LFC_GATE),             # the scalar threshold
    )
    np.savez_compressed(
        args.out_dir / "de_reference_stats.npz",
        bucket_names=np.array(bucket_names, dtype=object).astype(str),
        bucket_mu=bucket_mu, bucket_sd=bucket_sd, bucket_n=bucket_n,
        pool_mu=pool_mu, pool_sd=pool_sd, pool_n=np.int32(pool_n),
        gene_mask=base_mask, nameable_mask=nameable, sex_linked_mask=is_sex,
        cnv_polymorphism_mask=is_cnv,
        gene_symbols=symbols.astype(str),
        vocab_ensg=symbol_map["ensg_id"].astype(str).to_numpy(),
    )
    timings["stage6_write_outputs_s"] = round(time.time() - t, 1)

    # --- sanity checks
    ok = res["status"] == "ok"
    distinct_top_up = len({tuple(x) for x in res["up_genes"] if len(x)})

    # stable_gene_z NaN is a DELIBERATE "not comparable" marker, not a gap: a
    # stable gene with zero variance inside a sample's own tissue reference has
    # no defined z. Verify every NaN is exactly such a case rather than a bug.
    nan_mask = np.isnan(res["stable_z"])
    expected_nan = np.zeros_like(nan_mask)
    for i in range(len(pos_index)):
        if not ok[i]:
            expected_nan[i] = True
            continue
        bi = int(res["ref_id"][i])
        sd = pool_sd if bi < 0 else bucket_sd[bi]
        expected_nan[i] = ~(base_mask[stable_cols] & (sd[stable_cols] >= SD_FLOOR))
    nan_explained = bool(np.array_equal(nan_mask, expected_nan))

    all_up_lfc = np.concatenate([np.asarray(x) for x in res["up_lfc"] if len(x)])
    all_up_z = np.concatenate([np.asarray(x) for x in res["up_z"] if len(x)])

    # --- confound measurement (measured, not assumed; nothing is removed) ---
    SEX_GENES = {"RPS4Y1", "DDX3Y", "UTY", "USP9Y", "KDM5D", "EIF1AY",
                 "NLGN4Y", "ZFY", "XIST", "TSIX"}

    def _rank1_counts(lists, n=15):
        c = pd.Series([x[0] for x in lists if len(x)]).value_counts()
        return {str(k): int(v) for k, v in c.head(n).items()}

    def _sex_counts(lists):
        r1 = sum(1 for x in lists if len(x) and x[0] in SEX_GENES)
        t5 = sum(1 for x in lists if set(x[:5]) & SEX_GENES)
        any_ = sum(1 for x in lists if set(x) & SEX_GENES)
        return {"rank1": r1, "in_top5": t5, "anywhere_in_top25": any_,
                "n_lists": sum(1 for x in lists if len(x))}

    confounds = {
        "note": ("Sex-linked genes are real biology but are CONFOUNDS, not "
                 "cardiovascular signal. Measured first, then excluded from the "
                 "ranked lists by user decision. The pre-gate figures are kept "
                 "here deliberately: the finding stays auditable even though it "
                 "is now mitigated."),
        "headline_finding_before_mitigation": (
            "8.9% of samples had a sex-linked gene as their single most notable "
            "elevated gene; 9.4% had one somewhere in the top 25. RPS4Y1 alone "
            "drove the large majority. Top-down was essentially clean."),
        "mitigation": "sex_linked_gene_gate (see ranking_gates.exclude_sex_linked)",
        "sex_linked_genes_checked": sorted(SEX_GENES),
        "top_up": {
            "rank1_ungated": _rank1_counts(res["ung_up5"]),
            "rank1_before_sex_gate": _rank1_counts(res["presex_up"]),
            "rank1_after_sex_gate": _rank1_counts(res["up_genes"]),
            "sex_linked_ungated": _sex_counts(res["ung_up5"]),
            "sex_linked_before_sex_gate": _sex_counts(res["presex_up"]),
            "sex_linked_after_sex_gate": _sex_counts(res["up_genes"]),
        },
        "top_down": {
            "rank1_ungated": _rank1_counts(res["ung_down5"]),
            "rank1_before_sex_gate": _rank1_counts(res["presex_down"]),
            "rank1_after_sex_gate": _rank1_counts(res["down_genes"]),
            "sex_linked_ungated": _sex_counts(res["ung_down5"]),
            "sex_linked_before_sex_gate": _sex_counts(res["presex_down"]),
            "sex_linked_after_sex_gate": _sex_counts(res["down_genes"]),
        },
        "caveat": ("'ungated' lists hold only 5 entries (diagnostic), so "
                   "'anywhere_in_top25' is not meaningful for them. "
                   "'before_sex_gate' is the full top-25 with the lfc and "
                   "nameable gates applied but NOT the sex gate — it is the "
                   "live measurement of what the sex gate displaced."),
    }

    checks = {
        "n_rows": int(len(table)),
        "n_rows_correct": bool(len(table) == len(pos_index)),
        "n_ok": int(ok.sum()),
        "n_insufficient_data": int((~ok).sum()),
        "insufficient_reasons": {str(k): int(v) for k, v in
                                 pd.Series([r for r in res["reason"] if r]).value_counts().items()},
        "distinct_top_up_gene_lists": distinct_top_up,
        "distinct_top_up_first_gene": len({x[0] for x in res["up_genes"] if len(x)}),
        "degeneracy_fixed": bool(distinct_top_up > 0.5 * int(ok.sum())),
        "n_nan_mean_abs_z": int(np.isnan(res["mean_abs_z"][ok]).sum()),
        "no_nan_summary_stats": bool(np.isnan(res["mean_abs_z"][ok]).sum() == 0),
        "n_nan_stable_z": int(nan_mask[ok].sum()),
        "frac_nan_stable_z": float(nan_mask[ok].mean()),
        "n_stable_genes_with_any_nan": int(nan_mask[ok].any(axis=0).sum()),
        "stable_z_nan_fully_explained_by_zero_variance": nan_explained,
        "magnitude_bucket_counts": {str(k): int(v) for k, v in
                                    pd.Series(res["bucket_label"]).value_counts().items()},
        "all_top_genes_in_vocab": bool(set().union(*[set(x) for x in res["up_genes"] if len(x)])
                                       <= set(symbols.tolist())),
        "no_holdout_series_in_reference": bool(
            not set(ref["series_id"]).intersection(holdout_series)),
        "max_ref_mask_prob": max_mask_prob,
        "reference_mask_prob_zero": bool(max_mask_prob == 0.0),
        "n_lists_shorter_than_top_k": int(sum(
            1 for x in list(res["up_genes"]) + list(res["down_genes"])
            if 0 < len(x) < TOP_K)),
        # backfill assertion: slots must fall through to the next passing gene,
        # never leave a list short. Asserted, not assumed.
        "no_ranked_list_short": bool(all(
            len(x) == TOP_K for x in list(res["up_genes"]) + list(res["down_genes"])
            if len(x))),
        "all_top_genes_nameable": bool(
            not (set().union(*[set(x) for x in res["up_genes"] if len(x)])
                 | set().union(*[set(x) for x in res["down_genes"] if len(x)]))
            .intersection(set(symbols[~nameable].tolist()))),
        "all_top_lfc_pass_gate": bool(float(np.abs(all_up_lfc).min()) >= LFC_GATE),
        "n_sex_linked_in_top_up": int(sum(
            len(set(x) & set(SEX_LINKED_GENES)) for x in res["up_genes"])),
        "n_sex_linked_in_top_down": int(sum(
            len(set(x) & set(SEX_LINKED_GENES)) for x in res["down_genes"])),
        "no_sex_linked_genes_in_ranked_lists": bool(
            sum(len(set(x) & set(SEX_LINKED_GENES))
                for x in list(res["up_genes"]) + list(res["down_genes"])) == 0),
        "low_variance_tail": {
            "note": ("share of top-up entries whose z is large but whose log-fold-change "
                     "is trivial \u2014 the residue of the mandated sd < 1e-6 floor"),
            "frac_top_up_abs_lfc_lt_0.10": round(float((np.abs(all_up_lfc) < 0.10).mean()), 5),
            "frac_top_up_abs_lfc_lt_0.25": round(float((np.abs(all_up_lfc) < 0.25).mean()), 5),
            "frac_rank1_abs_lfc_lt_0.25": round(
                float(np.mean([abs(x[0]) < 0.25 for x in res["up_lfc"] if len(x)])), 5),
            "max_abs_z_observed_after_gate": round(float(np.abs(all_up_z).max()), 1),
        },
        "confound_audit": confounds,
    }
    checks["failures"] = [k for k in (
        "n_rows_correct", "degeneracy_fixed", "no_nan_summary_stats",
        "stable_z_nan_fully_explained_by_zero_variance",
        "all_top_genes_in_vocab", "no_holdout_series_in_reference", "reference_mask_prob_zero",
        "all_top_genes_nameable", "all_top_lfc_pass_gate",
        "no_sex_linked_genes_in_ranked_lists", "no_ranked_list_short",
    ) if not checks[k]]
    checks["passed"] = not checks["failures"]

    finished = datetime.now(timezone.utc)
    manifest = {
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "timings": timings,
        "purpose": ("REAL per-sample differential expression: how each disease-confirmed "
                    "sample's own transcriptome deviates from a tissue-matched neg_hard "
                    "reference population."),
        "why_this_exists": ("Stage-2 answers were 86.4% a single fixed string because "
                            "comparative_differential_reasoning returned a corpus-level "
                            "probe summary (per_gene_differential: None) and "
                            "gene_driver_reasoning returned the same global elastic-net "
                            "ranking for every sample (sample_role: eligibility_gate_only). "
                            "This file supplies the per-sample quantity they should consume."),
        "transform": {
            "code_reused_from": "linear_probe/extract.py::normalize_and_align + ArchS4CountReader",
            "steps": "raw counts / gene_length_kb -> TPM (per-sample, 1e6) -> log1p -> vocab align",
            "units": "log1p(TPM); lfc = x - mu is therefore already a log fold change",
            "reimplemented": False,
        },
        "populations": pop,
        "holdout_exclusion": {
            "holdout_series_file": str(HOLDOUT_PATH.relative_to(REPO)),
            "n_holdout_series": len(holdout_series),
            "n_neg_hard_excluded": pop["n_neg_hard_excluded_holdout"],
            "rationale": ("holdout-series negatives contributing to reference statistics "
                          "would leak holdout information into TRAINING answers"),
            "verified_no_holdout_series_in_reference": checks["no_holdout_series_in_reference"],
        },
        "tissue_matching": {
            "bucket_key": "normalized source_name_ch1 (lowercase, punctuation/whitespace collapsed)",
            "min_bucket_n": MIN_BUCKET_N,
            "n_distinct_reference_source_names": int(len(set(ref_tissue))),
            "n_qualifying_buckets": len(bucket_names),
            "n_reference_samples_in_qualifying_buckets": int(bucket_n.sum()),
            "n_positives_tissue_matched": int(res["n_tissue_matched"]),
            "n_positives_pooled_fallback": int(len(pos_index) - res["n_tissue_matched"]),
            "note": "reference_scope records the choice per sample; never silently substituted",
        },
        "gene_exclusions": {**gene_excl, "sentinel_value": SENTINEL, "sd_floor": SD_FLOOR,
                            "note": "per-reference sd floor is applied again inside each bucket"},
        "stable_signal_genes": {
            "source": str(GENE_SIGNAL_PATH.relative_to(REPO)),
            "criterion": "nonzero_frac == 1.0",
            "n_raw": int(n_stable_raw),
            "n_in_bulkformer_vocab": int(n_stable_in_vocab),
            "n_final_after_gene_mask": int(len(stable)),
        },
        "magnitude_tertiles": {
            "metric": "frac_genes_abs_z_gt2",
            "cut_low": res["cut_lo"], "cut_high": res["cut_hi"],
            "labels": ["minimal", "moderate", "large"],
            "computed_over": "the 'ok' samples, once, across the whole positive set",
        },
        "ranking_gates": {
            "applies_to": "top_up_genes / top_down_genes and their z/lfc arrays ONLY",
            "does_not_apply_to": ("mean_abs_z, max_abs_z, n_genes_abs_z_gt2, "
                                  "frac_genes_abs_z_gt2, magnitude_bucket, and the "
                                  "z/lfc matrices in stable_gene_z.npz — a deviation "
                                  "COUNT is not distorted by the artifact the way a "
                                  "top-K named-gene list is"),
            "min_abs_lfc": LFC_GATE,
            "min_abs_lfc_rationale": ("0.5 on a log1p(TPM) scale is ~1.65x. Below it, a "
                                      "gene whose reference sd sits just above the 1e-6 "
                                      "floor can post an enormous z off a trivial fold "
                                      "change; 'GENE elevated (z=10548)' is supervision "
                                      "that teaches the model to describe noise."),
            "require_gene_symbol": True,
            "require_gene_symbol_rationale": ("symbol_vocab_map falls back to the ENSG "
                                              "accession when a gene has no symbol. A gene "
                                              "we cannot name is a gene we should not claim."),
            "n_vocab_entries_without_symbol": int((~nameable).sum()),
            "exclude_sex_linked": True,
            "sex_linked_genes": list(SEX_LINKED_GENES),
            "sex_linked_genes_matched_in_vocab": sex_found,
            "sex_linked_genes_not_in_vocab": sex_missing,
            "n_vocab_entries_sex_linked": int(is_sex.sum()),
            "exclude_sex_linked_rationale": (
                "Sex is a covariate here, not the phenotype under study. Excluding "
                "sex-linked genes from a DE ranking is routine when sex is not the "
                "variable of interest. Without it, 8.9% of samples would have had "
                "chromosomal sex as the headline finding of their generated answer "
                "— per-sample and numerically correct, but not cardiovascular "
                "signal. Applied by user decision after the confound was measured."),
            "mean_genes_dropped_per_sample_by_sex_gate": round(
                float(res["n_drop_sex"][res["comparable"]].mean()), 3),
            "mean_genes_dropped_per_sample_by_lfc_gate": round(
                float(res["n_drop_lfc"][res["comparable"]].mean()), 1),
            "mean_nameable_genes_dropped_per_sample_by_symbol_gate": round(
                float(res["n_drop_unnameable"][res["comparable"]].mean()), 1),
            "mean_genes_eligible_for_ranking_per_sample": round(
                float(res["n_gate_pass"][res["comparable"]].mean()), 1),
            "min_genes_eligible_for_ranking": int(res["n_gate_pass"][res["comparable"]].min()),
            "stable_gene_z_note": ("z/lfc keep the full 1,142-gene row as required; the "
                                   "same gate is supplied precomputed as `rank_gate` for "
                                   "selection-time use."),
        },
        "ranking_semantics": {
            "what_top_k_is": ("The 25 most extreme genes by z among those passing the "
                              "ranking gates. It is a RANKING, not a significance test."),
            "what_top_k_is_not": ("NOT '25 genes that passed a significance bar'. A list "
                                  "can be entirely below |z| = 2 — that simply means the "
                                  "sample has no strongly deviating genes in that "
                                  "direction, which is an honest finding."),
            "consumer_rule": ("Do not render a |z| below 2 as 'significantly elevated' or "
                              "'significantly reduced'. Check the z value before choosing "
                              "the wording; n_genes_abs_z_gt2 is the count that speaks to "
                              "how many genes actually clear |z| > 2."),
            "lists_may_be_shorter_than_25": ("When fewer than 25 genes clear the gates the "
                                             "list is shorter. It is never padded."),
        },
        "nan_semantics": {
            "stable_gene_z.npz z/lfc": ("NaN means NOT COMPARABLE for that sample: the gene "
                                        "had zero variance inside that sample's own reference, "
                                        "so no z is defined. Never impute it \u2014 the consuming "
                                        "GT function must omit the gene for that sample."),
        },
        "outputs": {
            "per_sample": "per_sample_de.parquet",
            "stable_gene_z": "stable_gene_z.npz",
            "reference_stats": "de_reference_stats.npz",
        },
        "sanity_checks": checks,
    }
    if args.limit_ref is not None:
        manifest["SMOKE_TEST"] = f"reference truncated to {args.limit_ref} samples — NOT a real run"
    (args.out_dir / "de_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"manifest written; sanity checks passed={checks['passed']} "
                f"failures={checks['failures']}")
    logger.info(f"TOTAL {(time.time() - t_start) / 60:.1f} min")
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
