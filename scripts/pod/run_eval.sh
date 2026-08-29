#!/bin/bash
# Phase 4 — everything that has to happen on the GPU box after the retrain.
#
#   1. extract the retrained model's latents for the full 57,207-sample probe
#      population (imgtok and meanpool);
#   2. three-way probe on the clean holdout: encoder vs linear vs MLP;
#   3. broad multi-label probing on the retrained latents AND on the pre-fix
#      ones, so the before/after comparison is made in ONE environment.
#
# Step 3's "before" run is the reason this is a script rather than three
# commands. The pre-fix latents already exist from the 2026-08-13 session, but
# they were probed with a different sklearn build; re-probing both here means
# the before/after difference cannot be an artifact of the library version.
# That is also why the pre-fix parquet has to be uploaded rather than the old
# numbers simply being quoted.
#
# Usage (on the pod):
#   bash scripts/pod/run_eval.sh 2>&1 | tee /workspace/eval.log

set -euo pipefail

REPO="${REPO:-/workspace/CardioLLM}"
VENV="${VENV:-/workspace/venv}"
PY="${VENV}/bin/python"
CKPT="${CKPT:-${REPO}/checkpoints/stage2-lora-bulkformer-93M}"
EMB="${REPO}/linear_probe/embeddings"
TABLES="${REPO}/stage2_regen_report/tables"

export HF_HOME="${HF_HOME:-/workspace/hf}"
cd "$REPO"
mkdir -p "$TABLES"

step() { echo; echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

step "extract latents from the retrained checkpoint"
# Input is the probe's own 515-d parquet, so the encoder is never re-run: those
# vectors ARE the tower's pooled output and the passthrough takes them by width.
if [ ! -f "${EMB}/embeddings_LLM-latent-regen-imgtok.parquet" ]; then
    "$PY" -m eval_binary_comparison.extract_llm_latents \
        --lora-ckpt "$CKPT" \
        --embeddings "${EMB}/embeddings_BulkFormer-93M.parquet" \
        --out-prefix "${EMB}/embeddings_LLM-latent-regen" \
        --batch-size 64
else
    echo "regen latents already extracted"
fi

step "three-way probe (encoder vs linear vs MLP) on the clean holdout"
"$PY" -m eval_binary_comparison.run_regen_eval \
    --llm-latents "${EMB}/embeddings_LLM-latent-regen-imgtok.parquet" \
    --encoder "${EMB}/embeddings_BulkFormer-93M.parquet" \
    --out "${TABLES}/probe_three_way.json"

step "multi-label probing — AFTER (retrained latents)"
"$PY" -m eval_binary_comparison.run_multilabel_probe \
    --features "${EMB}/embeddings_LLM-latent-regen-imgtok.parquet" \
    --name "LLM-latent-regen-imgtok" \
    --out "${TABLES}/multilabel_probe_after.csv"

step "multi-label probing — BEFORE (pre-fix latents)"
# Same environment, same sklearn build, same folds as the "after" run above.
# Quoting the 2026-08-13 numbers instead would leave the library version as an
# uncontrolled variable in the one comparison this phase exists to make.
if [ -f "${EMB}/embeddings_LLM-latent-imgtok.parquet" ]; then
    "$PY" -m eval_binary_comparison.run_multilabel_probe \
        --features "${EMB}/embeddings_LLM-latent-imgtok.parquet" \
        --name "LLM-latent-prefix-imgtok" \
        --out "${TABLES}/multilabel_probe_before.csv"
else
    echo "SKIP: pre-fix latents not present — upload"
    echo "      linear_probe/embeddings/embeddings_LLM-latent-imgtok.parquet"
    echo "      to run the before/after comparison in one environment."
fi

step "done"
ls -la "$TABLES"
