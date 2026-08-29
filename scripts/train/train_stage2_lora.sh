#!/bin/bash
# Stage 2 — LoRA instruction tuning on the CVD transcriptome QA set.
#
# Starts from the Stage-1 connector-alignment checkpoint and trains LoRA
# adapters on the Vicuna-7B LLM while the connector is tuned in full and the
# BulkFormer tower stays frozen (standing project decision).
#
# Every flag below was verified against this repo's actual LoRA path
# (tinyllava/training_recipe/lora_recipe.py, tinyllava/utils/arguments.py) and
# exercised by a 5-step dry run — see integration/stage2_lora_dryrun_result.md.
#
# ---------------------------------------------------------------------------
# THREE NON-OBVIOUS CONSTRAINTS THIS REPO IMPOSES (do not "clean these up")
# ---------------------------------------------------------------------------
# 1. --conv_version is `llama`, NOT `vicuna_v1`. `vicuna_v1` is not registered
#    here (TemplateFactory raises); tinyllava/data/template/llama_template.py IS
#    the Vicuna v1.5 format ("A chat between a curious user...", USER:/ASSISTANT:,
#    </s> separator). Confirmed in integration/vicuna_smoke_test_result.md.
#
# 2. OUTPUT_DIR must contain the substring "lora". tinyllava/model/load_model.py
#    selects its LoRA-merging load path by testing `'lora' in model_name_or_path`;
#    a Stage-2 LoRA checkpoint saved to a dir without "lora" in the name silently
#    loads as a plain (adapter-less) model at eval time.
#
# 3. STAGE1_CKPT must NOT contain the substring "lora". The mirror-image test in
#    tinyllava/training_recipe/base.py::load would otherwise try to
#    PeftModel.from_pretrained + merge the Stage-1 dir instead of loading its
#    connector normally.
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# HARDWARE TARGET: 4x A100-80GB (320 GB total), DeepSpeed ZeRO-2.
# This is the configuration matching the PI's spec exactly: next-token
# prediction / causal CE loss, LoRA weights, 4x A100, ZeRO-2.
# Retuned 2026-08-11 — see scripts/train/hardware_retune_report.md.
#
# Sizing rationale:
#   ZeRO-2 leaves the fp16 base weights unsharded: ~13.5 GB/GPU, flat.
#   Trainable set is ~322 M (LoRA r=128 + the 515->4096 connector, 2.11 M), so
#   fp32 Adam states are ~3.9 GB sharded 4 ways => ~1 GB/GPU. Small, and it does
#   not scale with batch.
#   Batch-scaling terms: the frozen BulkFormer forward's fp32 [B, 20010, 515]
#   intermediates (the dominant one), then LLM activations. The tower mean-pools
#   to ONE token per sample, so the LLM sees text only — and Stage-2 sequences are
#   short and tight (measured on the real bundle: mean 141, p95 190, max 199
#   tokens), which is why the LLM side stays cheap even at batch 32.
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- data -----------------------------------------------------------------
# Materialized Stage-2 bundle: 19,793 items over all three categories
# (disease_subtype_classification, comparative_differential_reasoning bound to
# the neg_hard comparison group, gene_driver_reasoning).
DATA_PATH="${DATA_PATH:-data/cvd_transcriptome/text_files/stage2_train.json}"
# Per-sample .npy expression vectors. Point this at the output of
# integration/precompute_encoder_cache.py to feed pre-encoded [dim+3] vectors
# instead of raw [20010] ones — BulkFormerVisionTower.forward detects the width
# and passes them through. Keep it consistent with whatever Stage 1 used.
IMAGE_PATH="${IMAGE_PATH:-data/cvd_transcriptome/embeddings}"

# ---- encoder scale (LOCKED) -------------------------------------------------
# BulkFormer-93M is the final, fixed encoder scale for this project — not a
# tunable, not pending any further comparison. Justification: the completed
# 5-variant linear-probe sweep in linear_probe/results/ (93M is the top variant
# on both negative pools; 127M/147M do not improve on it). See
# llm_training_plan.md §2.
BULKFORMER_SCALE="93M"
BULKFORMER_VARIANT="BulkFormer-${BULKFORMER_SCALE}"

# ---- model ----------------------------------------------------------------
LLM_VERSION=lmsys/vicuna-7b-v1.5
VT_CONFIG_DIR=$(pwd)/integration/bulkformer_hf_config
VT_VERSION=bulkformer:${VT_CONFIG_DIR}
VT_VERSION2=""
CN_VERSION=transcript_linear
CONV_VERSION=llama                     # see constraint (1) above
MODEL_MAX_LENGTH=2048

# Attention kernel. flash_attention_2 is the default and what the 2026-08-13 run
# used, but it is the one dependency that can fail to build on a fresh pod. The
# fallback is cheap here: Stage-2 sequences are short (measured on the real
# bundle: mean 141, p95 190, max 199 tokens), so attention is not the bottleneck
# and `sdpa` costs little. Set ATTN_IMPL=sdpa when flash-attn is unavailable
# rather than blocking the run on it.
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

# The variant the tower actually instantiates comes from that config dir, not
# from this script — so assert the two agree instead of assuming it.
if ! grep -q "\"bulkformer_variant\": \"${BULKFORMER_VARIANT}\"" "${VT_CONFIG_DIR}/config.json"; then
  echo "ERROR: ${VT_CONFIG_DIR}/config.json does not select ${BULKFORMER_VARIANT}." >&2
  echo "       The encoder scale is locked at ${BULKFORMER_VARIANT}; fix the config." >&2
  exit 1
fi

# ---- checkpoints ----------------------------------------------------------
# Stage-1 output dir. Must contain language_model/, vision_tower/, connector/
# subdirs (written by BaseTrainingRecipe.save). The connector weights in
# connector/pytorch_model.bin are the Stage-1-trained ones that Stage 2 starts
# from — this flag does NOT merely re-init a fresh connector.
#
# Fixed path: this is exactly what scripts/train/train_stage1.sh writes with the
# encoder scale locked at 93M (its OUTPUT_DIR is
# checkpoints/stage1-connector-${BULKFORMER_SCALE}). No substitution needed.
STAGE1_CKPT=${STAGE1_CKPT:-./checkpoints/stage1-connector-93M}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/stage2-lora-bulkformer-93M}

case "$STAGE1_CKPT" in *lora*)
  echo "ERROR: STAGE1_CKPT must not contain 'lora' — see constraint (3)." >&2; exit 1;; esac
case "$OUTPUT_DIR" in *lora*) ;; *)
  echo "ERROR: OUTPUT_DIR must contain 'lora' — see constraint (2)." >&2; exit 1;; esac

if [ ! -d "$STAGE1_CKPT/connector" ]; then
  echo "ERROR: no Stage-1 connector at $STAGE1_CKPT/connector — run scripts/train/train_stage1.sh first." >&2
  exit 1
fi

# Intermediate-checkpoint policy — see the same block in train_stage1.sh. A full
# HF Trainer checkpoint here is tens of GB (frozen base weights + ZeRO optimizer
# state) on top of the ~14 GB the LoRA recipe writes at the end. On a volume that
# cannot hold both, set SAVE_STRATEGY=no and take the final save only.
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-50000}"

RUN_NAME=$(basename "$OUTPUT_DIR")

# ---- hardware -------------------------------------------------------------
GPUS="${GPUS:-0,1,2,3}"                       # 4x A100-80GB
NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPUS")
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPUS}"
# 8 -> 32 per device, accum 4 -> 1. Global batch UNCHANGED at 128
# (4 x 32 x 1 == 4 x 8 x 4).
PER_DEVICE_BS="${PER_DEVICE_BS:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
STEP_ARGS=(--num_train_epochs 1)

# MAX_STEPS=N runs a short probe instead of a full epoch (used by the
# memory-verification dry run; see hardware_retune_report.md).
if [ -n "${MAX_STEPS:-}" ]; then STEP_ARGS=(--max_steps "$MAX_STEPS"); fi

deepspeed --include "localhost:${GPUS}" --master_port 29501 tinyllava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_PATH" \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower "$VT_VERSION" \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation "$ATTN_IMPL" \
    --bf16 True \
    --training_recipe lora \
    --tune_type_llm lora \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --bits 16 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --group_by_modality_length False \
    --pretrained_model_path "$STAGE1_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    "${STEP_ARGS[@]}" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --evaluation_strategy "no" \
    --save_strategy "$SAVE_STRATEGY" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --mm_projector_lr 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name "$RUN_NAME"

# Notes on the hyperparameters above:
#   --lora_r 128 / --lora_alpha 256 mirror scripts/train/lora/finetune_lora.sh
#     (this repo's own reference LoRA config). lora_r/lora_alpha are plain ints
#     and lora_dropout a float, passed straight into peft.LoraConfig; there is no
#     validated range beyond PEFT's own. --lora_bias must be one of
#     none|all|lora_only (peft LoraConfig).
#   --bits 16 is required to be 16 for the fp16/bf16 cast in
#     LoRATrainingRecipe.training_model_converse to run (it is also the default).
#     Values other than 16 belong to the separate `qlora` recipe.
#   --tune_type_llm lora is what removes 'language_model' from the recipe's
#     lora_skip_module list, i.e. it is the flag that decides LoRA is applied to
#     the LLM at all. --tune_type_connector full keeps the connector fully
#     trainable on top of the adapters (re-enabled after PEFT freezes everything).
#   --mm_projector_lr 2e-5 gives the already-aligned Stage-1 connector a gentler
#     LR than the 2e-4 the fresh adapters get.
#   --bf16: A100-class and newer. Switch to --fp16 True on V100-class GPUs.
