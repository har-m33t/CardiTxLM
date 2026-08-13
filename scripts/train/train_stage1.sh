#!/bin/bash
# Stage 1 — connector-only alignment for the BulkFormer tower.
#
#   Trainable: the connector (transcript_linear) ONLY.
#   Frozen:    the BulkFormer encoder AND the Vicuna-7B LLM.
#
# This is the standalone Stage-1 launcher. It calls tinyllava/train/train.py
# directly (NOT custom_finetune.py — that entrypoint requires
# --pretrained_model_path and loads an already-built TinyLLaVA via
# AutoModelForCausalLM, which is the Stage-2 shape, not Stage 1).
#
# Every flag below is verified against tinyllava/utils/arguments.py
# (ModelArguments / DataArguments / TrainingArguments) — see the verification
# report in integration/train_stage1_verification.md.
#
# ---------------------------------------------------------------------------
# HARDWARE TARGET: 4x A100-80GB (320 GB total), DeepSpeed ZeRO-2.
# Retuned 2026-08-11 — see scripts/train/hardware_retune_report.md.
#
# Sizing rationale (this is NOT parameter-dominated):
#   Trainable set is the connector alone (515->4096 = 2.11 M params), so
#   gradients + Adam states are ~32 MB before sharding — numerically irrelevant.
#   Under ZeRO-2 the frozen fp16 7B weights are NOT sharded: ~13.5 GB/GPU, flat,
#   independent of batch size.
#   The batch-scaling term is the frozen BulkFormer forward, which materializes
#   [B, 20010, 515] in fp32 under no_grad — ~41.2 MB/sample for the output alone
#   and a few hundred MB/sample at the block's transient peak. (515 = the locked
#   BulkFormer-93M width, dim 512 + 3; the retune report's 643/51.5 MB figures
#   were for the retired 127M variant, so this term is now ~20% cheaper.)
#   The LLM forward is cheap here: the tower mean-pools to exactly ONE token per
#   sample (bulkformer.py::forward), so sequence length is just the text
#   (measured on the real corpus: mean 237, p95 660 tokens).
#   => batch is bounded by the TOWER's fp32 activations, not by the LLM and not
#      by parameter count. Hence 32 -> 64 (2x), not a 4x GPU-proportional bump.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Usage
#   Real run (CUDA box):   bash scripts/train/train_stage1.sh
#   Plumbing dry run:      DRY_RUN=1 bash scripts/train/train_stage1.sh
#
# DRY_RUN=1 swaps in: a stubbed BulkFormer encoder, a tiny random Llama, a
# synthetic 8-sample dataset, CPU/fp32, no deepspeed, and --max_steps 5.
# It verifies flags + the training loop; it does NOT verify the real encoder or
# the real backbone. Never produce a real checkpoint from it.
# ---------------------------------------------------------------------------

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---- encoder scale (LOCKED) -------------------------------------------------
# BulkFormer-93M is the final, fixed encoder scale for this project — not a
# tunable, not pending any further comparison. Justification: the completed
# 5-variant linear-probe sweep in linear_probe/results/ (93M is the top variant
# on both negative pools; 127M/147M do not improve on it). See
# llm_training_plan.md §2.
BULKFORMER_SCALE="93M"
BULKFORMER_VARIANT="BulkFormer-${BULKFORMER_SCALE}"

# ---- model -----------------------------------------------------------------
# Vision tower: the "bulkformer" prefix before ':' selects BulkFormerVisionTower
# via VisionTowerFactory's substring match; the path after ':' is the local HF
# config dir exposing hidden_size=515 + bulkformer_variant (BulkFormer-93M).
# (VisionTowerFactory splits on ':' [0] for the name; train.py's
# _load_vision_settings splits on ':' [-1] for the path.)
VT_CONFIG_DIR="${REPO_ROOT}/integration/bulkformer_hf_config"
VT_VERSION="${VT_VERSION:-bulkformer:${VT_CONFIG_DIR}}"

# The variant the tower actually instantiates comes from that config dir, not
# from this script — so assert the two agree instead of assuming it.
if ! grep -q "\"bulkformer_variant\": \"${BULKFORMER_VARIANT}\"" "${VT_CONFIG_DIR}/config.json"; then
  echo "ERROR: ${VT_CONFIG_DIR}/config.json does not select ${BULKFORMER_VARIANT}." >&2
  echo "       The encoder scale is locked at ${BULKFORMER_VARIANT}; fix the config." >&2
  exit 1
fi
VT_VERSION2=""
CN_VERSION=transcript_linear
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048
# Stage 1 uses the 'pretrain' template (tinyllava/data/template/pretrain_template.py),
# which drops the user turn entirely and supervises only the assistant text.
#
# ⚠ READ integration/train_stage1_verification.md §6 BEFORE THE REAL RUN.
# 'pretrain' is a valid, registered template, but it DISCARDS the question text.
# stage1_train.json is gene-specific QA (~23 differently-answered pairs per
# expression vector), so under 'pretrain' the target is not determined by the
# input and the model can only learn the marginal answer distribution.
# Set CONV_VERSION=llama to keep the question in the prompt (Vicuna v1 format).
CONV_VERSION="${CONV_VERSION:-pretrain}"

# ---- mode-dependent settings -----------------------------------------------
if [ "$DRY_RUN" = "1" ]; then
    SCRATCH="${SCRATCH:-${REPO_ROOT}/integration/_stage1_dryrun}"
    python -m integration.make_synthetic_stage1_data --outdir "$SCRATCH" --n 8

    # Dir name must contain "tinyllama" so the substring-matching LLMFactory
    # picks the Llama loader for this tiny random stand-in.
    LLM_VERSION="${SCRATCH}/tinyllama_stub"
    python - "$LLM_VERSION" <<'PY'
import sys, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
dest = pathlib.Path(sys.argv[1])
if not (dest / "config.json").exists():
    src = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    AutoModelForCausalLM.from_pretrained(src).save_pretrained(dest)
    AutoTokenizer.from_pretrained(src).save_pretrained(dest)
print(f"[dry-run] tiny LLM stub at {dest}")
PY

    DATA_PATH="${SCRATCH}/pretrain.json"
    IMAGE_PATH="${SCRATCH}/images"
    OUTPUT_DIR="${SCRATCH}/checkpoint"
    LAUNCHER=(python -m integration.dryrun_stage1)
    DEEPSPEED_FLAGS=()
    # --use_cpu True: on Apple silicon the Trainer otherwise picks MPS, where the
    # stub encoder's [B, 20010, 515] intermediate faults (Bus error: 10).
    PRECISION_FLAGS=(--fp16 False --attn_implementation eager --use_cpu True)
    MAX_STEPS=5
    NUM_EPOCHS=1
    BATCH_SIZE=2
    GRAD_ACCUM=1
    NUM_WORKERS=0
    REPORT_TO=none
    GRAD_CKPT=False
else
    # The real Stage-1 corpus: 199,954 QA pairs over 8,553 per-sample [20010]
    # float32 log1p(TPM) vectors (verified present; see materialization_manifest.json).
    DATA_PATH="${DATA_PATH:-data/cvd_transcriptome/text_files/stage1_train.json}"
    IMAGE_PATH="${IMAGE_PATH:-data/cvd_transcriptome/embeddings}"
    # LLM backbone: Vicuna-7B, the confirmed standard backbone. Selected by the
    # "vicuna" substring via tinyllava/model/llm/vicuna.py.
    LLM_VERSION="${LLM_VERSION:-lmsys/vicuna-7b-v1.5}"
    # Fixed Stage-1 output dir. scripts/train/train_stage2_lora.sh hard-codes
    # exactly this path as its --pretrained_model_path — keep the two in sync.
    # (Must not contain the substring "lora"; see train_stage2_lora.sh note 3.)
    OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/stage1-connector-${BULKFORMER_SCALE}}"
    # 4x A100-80GB. NUM_GPUS is derived from GPUS so the two can never disagree.
    GPUS="${GPUS:-0,1,2,3}"
    NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPUS")
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPUS}"
    LAUNCHER=(deepspeed --include "localhost:${GPUS}" --master_port 29501 tinyllava/train/train.py)
    # ZeRO-2 (was zero3.json). With only the connector trainable there is no
    # optimizer state worth sharding, and 80 GB absorbs the unsharded 13.5 GB
    # fp16 weights easily — so ZeRO-3's per-layer all-gather is pure overhead.
    # Also matches the PI's specified config and both Stage-2 scripts.
    DEEPSPEED_FLAGS=(--deepspeed ./scripts/zero2.json)
    PRECISION_FLAGS=(--fp16 True --attn_implementation flash_attention_2)
    MAX_STEPS="${MAX_STEPS:--1}"   # -1 => use --num_train_epochs
    NUM_EPOCHS=1
    # Batch is bounded by SEQUENCE LENGTH, and Stage-1 sequence length depends on
    # CONV_VERSION — so the safe batch does too. Measured on the real corpus with
    # the real Vicuna tokenizer: mean 237, p50 168, p95 660, max 3458 (clipped to
    # model_max_length 2048). The collator pads to the longest item IN THE BATCH,
    # so a large batch is very likely to contain a long-tail sample and pad
    # everything up to it.
    #
    # Estimated peak/GPU (ZeRO-2, 4x A100-80GB), worst-case padded batch:
    #   pretrain, B=64, S~320   ~44 GB   OK
    #   llama,    B=64, S=2048  ~91 GB   OOM
    #   llama,    B=32, S=2048  ~52 GB   OK
    # The two big batch-scaling terms at long S are the grad-checkpoint layer
    # boundaries (32 x tokens x 4096 x 2B) and the logits, which materialize
    # BOTH a bf16/fp16 [B,S,32000] tensor and an fp32 copy for the CE loss
    # (LlamaForCausalLM upcasts with `logits = logits.float()`) = 192 kB/token.
    #
    # 'pretrain' drops the user turn entirely (answer only => short), 'llama'
    # keeps question + answer (long tail to 2048). Hence the split below.
    if [ "$CONV_VERSION" = "pretrain" ]; then
        BATCH_SIZE="${BATCH_SIZE:-64}"   # was 32; global batch 4x64x1 = 256, unchanged
        GRAD_ACCUM="${GRAD_ACCUM:-1}"    # was 2
    else
        BATCH_SIZE="${BATCH_SIZE:-32}"   # long-sequence template — do NOT raise without measuring
        GRAD_ACCUM="${GRAD_ACCUM:-2}"    # global batch 4x32x2 = 256, unchanged
    fi
    NUM_WORKERS=8
    REPORT_TO=tensorboard          # requires `pip install tensorboard`
    GRAD_CKPT=True
fi

RUN_NAME="tiny-llava-bulkformer-${BULKFORMER_SCALE}-pretrain"

# NOTE: ${A[@]+"${A[@]}"} — bash 3.2 (macOS) treats an empty array as unset
# under `set -u`, so DEEPSPEED_FLAGS must be expanded defensively.
"${LAUNCHER[@]}" \
    ${DEEPSPEED_FLAGS[@]+"${DEEPSPEED_FLAGS[@]}"} \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_PATH" \
    --is_multimodal True \
    --conv_version "$CONV_VERSION" \
    --model_name_or_path "$LLM_VERSION" \
    --vision_tower "$VT_VERSION" \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type "$CN_VERSION" \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    "${PRECISION_FLAGS[@]}" \
    --training_recipe "$TRAIN_RECIPE" \
    --tune_type_llm frozen \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --gradient_checkpointing "$GRAD_CKPT" \
    --dataloader_num_workers "$NUM_WORKERS" \
    --lazy_preprocess True \
    --report_to "$REPORT_TO" \
    --tokenizer_use_fast False \
    --run_name "$RUN_NAME"
