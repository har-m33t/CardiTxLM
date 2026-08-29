#!/bin/bash
# Package everything worth keeping off the pod into one archive.
#
# Run this BEFORE deleting the pod. Deleting a pod destroys its volume, and the
# artifacts below are the entire product of the GPU spend — the checkpoint, the
# loss history the before/after plot needs, the probe results, and the extracted
# latents that make any future re-probe possible without paying for the LLM
# forward again.
#
# The LoRA checkpoint is small (~0.6 GB) because only adapters and the connector
# are saved; the frozen base weights are not duplicated into it.
#
# Usage (on the pod):
#   bash scripts/pod/collect_results.sh
# then, from the local machine:
#   scp -P <port> root@<ip>:/workspace/regen_results.tar.gz .

set -euo pipefail

REPO="${REPO:-/workspace/CardioLLM}"
OUT="${OUT:-/workspace/regen_results.tar.gz}"
STAGE="${STAGE:-/workspace/_collect}"

rm -rf "$STAGE"; mkdir -p "$STAGE"/{checkpoint,logs,tables,latents}

echo "=== checkpoint ==="
CKPT="$REPO/checkpoints/stage2-lora-bulkformer-93M"
if [ -d "$CKPT" ]; then
    cp -r "$CKPT"/* "$STAGE/checkpoint/" 2>/dev/null || true
    du -sh "$STAGE/checkpoint"
else
    echo "WARNING: no checkpoint at $CKPT"
fi

echo "=== logs ==="
# trainer_state.json carries log_history — this is what the before/after loss
# plot is built from, so it matters more than the raw log.
for f in /workspace/stage2_regen_train.log /workspace/bootstrap.log /workspace/eval.log; do
    [ -f "$f" ] && cp "$f" "$STAGE/logs/" || true
done
[ -f "$CKPT/trainer_state.json" ] && cp "$CKPT/trainer_state.json" \
    "$STAGE/logs/stage2_regen_trainer_state.json" || \
    echo "WARNING: no trainer_state.json — the loss curve cannot be plotted without it"
ls -la "$STAGE/logs/"

echo "=== tables ==="
cp -r "$REPO/stage2_regen_report/tables/"* "$STAGE/tables/" 2>/dev/null || \
    echo "(no tables yet)"

echo "=== latents ==="
# Worth carrying back: re-extracting these means paying for the LLM forward over
# 57,207 samples again. Compresses poorly (dense float32), so they dominate the
# archive size — drop them by setting SKIP_LATENTS=1 if bandwidth matters more.
if [ "${SKIP_LATENTS:-0}" != "1" ]; then
    cp "$REPO"/linear_probe/embeddings/embeddings_LLM-latent-regen-*.parquet \
       "$STAGE/latents/" 2>/dev/null || echo "(no regen latents)"
fi
du -sh "$STAGE/latents" 2>/dev/null || true

echo "=== archive ==="
tar czf "$OUT" -C "$STAGE" .
ls -la "$OUT"
echo
echo "Now, from the local machine:"
echo "  scp -P <port> root@<ip>:$OUT ."
echo "THEN delete the pod. Stopping is not enough — a stopped pod still bills"
echo "for its volume, and nothing here is needed once this archive is down."
