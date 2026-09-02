#!/bin/bash
# Phase 4 — the full evaluation suite for the Hypothesis B retrain.
#
# Extends scripts/pod/run_eval.sh with the one thing that was not measurable
# before: a real held-out binary CVD evaluation. That eval only became
# meaningful once the model was actually trained on the task, which is what
# this condition adds.
#
# Every probe reuses the SAME methodology as the previous two conditions —
# identical folds (StratifiedGroupKFold(5, seed 20260707) grouped by series_id),
# identical PCA-matched dimensionality control, identical holdout — so the
# three training conditions are directly comparable. Anything else would make
# the comparison an artifact of the harness rather than the model.
#
# Usage (on the pod):
#   bash scripts/pod/run_eval_hypb.sh 2>&1 | tee /workspace/eval_hypb.log

set -euo pipefail

REPO="${REPO:-/workspace/CardioLLM}"
VENV="${VENV:-/workspace/venv}"
PY="${VENV}/bin/python"
CKPT="${CKPT:-${REPO}/checkpoints/stage2-lora-bulkformer-93M-hypb}"
EMB="${REPO}/linear_probe/embeddings"
TABLES="${REPO}/stage2_regen_report/tables"
PLOTS="${REPO}/stage2_regen_report/plots"

export HF_HOME="${HF_HOME:-/workspace/hf}"
# This pod's peer-to-peer NCCL transport is broken (a bare all-reduce hangs on
# the first collective and completes with P2P off). Single-GPU eval does not
# use collectives, but the flag is harmless and keeps the environment identical
# to the training run.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
cd "$REPO"
mkdir -p "$TABLES" "$PLOTS"

step() { echo; echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

step "extract latents from the Hypothesis-B checkpoint"
# Input is the probe's own 515-d parquet, so the encoder is never re-run: those
# vectors ARE the tower's pooled output and the passthrough takes them by width.
if [ ! -f "${EMB}/embeddings_LLM-latent-hypb-imgtok.parquet" ]; then
    "$PY" -m eval_binary_comparison.extract_llm_latents \
        --lora-ckpt "$CKPT" \
        --embeddings "${EMB}/embeddings_BulkFormer-93M.parquet" \
        --out-prefix "${EMB}/embeddings_LLM-latent-hypb" \
        --batch-size 64
else
    echo "hypb latents already extracted"
fi

step "4.1 three-way probe (encoder vs linear vs MLP) on the clean holdout"
"$PY" -m eval_binary_comparison.run_regen_eval \
    --llm-latents "${EMB}/embeddings_LLM-latent-hypb-imgtok.parquet" \
    --encoder "${EMB}/embeddings_BulkFormer-93M.parquet" \
    --out "${TABLES}/probe_three_way_hypb.json"

step "4.2 multi-label probing on the Hypothesis-B latents"
"$PY" -m eval_binary_comparison.run_multilabel_probe \
    --features "${EMB}/embeddings_LLM-latent-hypb-imgtok.parquet" \
    --name "LLM-latent-hypb-imgtok" \
    --out "${TABLES}/multilabel_probe_hypb.csv"

step "4.3 held-out binary CVD evaluation (forced-choice log-probability)"
# THE number to read here is within-series AUC, not pooled AUC. Zero non-holdout
# series contains both classes, so a model can score on the pooled metric by
# recognizing batch signature alone. All 92 holdout series ARE mixed, so
# comparing inside a series holds batch/platform/lab/tissue fixed and only real
# per-sample biology separates the classes. High pooled + ~0.5 within-series
# means the shortcut was taken. The encoder's own within-series AUC (0.765) is
# the bar, NOT its pooled 0.668.
"$PY" -m eval_binary_comparison.run_binary_cvd_eval \
    --lora-ckpt "$CKPT" \
    --encoder-embeddings "${EMB}/embeddings_BulkFormer-93M.parquet" \
    --stage2-json "${REPO}/data/cvd_transcriptome/text_files/stage2_train_hypb.json" \
    --tables-dir "$TABLES" --plots-dir "$PLOTS" \
    --out-tag binary_cvd_eval_hypb

step "done"
ls -la "$TABLES"
