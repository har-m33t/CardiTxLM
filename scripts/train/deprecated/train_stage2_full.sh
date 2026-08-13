#!/bin/bash
# =============================================================================
# Stage 2 — FULL fine-tune of the 7B LLM + connector, BulkFormer tower frozen.
# Target: 4x A100-80GB (320 GB total) via DeepSpeed ZeRO-2. NO CPU offload.
#
# This is the full-fine-tune comparison arm for scripts/train/train_stage2_lora.sh.
# Both branches now fork from the SAME Stage-1 checkpoint and run the SAME ZeRO
# stage (ZeRO-2), so the only intended difference between them is LoRA vs full FT.
#
#   !! GPU MEMORY STILL UNMEASURED !!
#   Retuned 2026-08-11 on a machine with no NVIDIA GPU. The budget below is
#   ARITHMETIC, NOT MEASUREMENT — no A100 was reachable at retune time.
#   Run the VERIFY probe (below) on the real 4xA100 box BEFORE queuing training.
#   See scripts/train/hardware_retune_report.md.
#
# Differences vs scripts/train/finetune.sh are all deliberate and listed in the
# "DEVIATIONS" section. Do not "fix" them silently.
# =============================================================================
#
# -----------------------------------------------------------------------------
# MEMORY BUDGET (estimated — supersedes the previous 2x4090 / ZeRO-3-offload
#                budget, which no longer applies)
# -----------------------------------------------------------------------------
# Full FT of Vicuna-7B, mixed precision, standard ~16 B/param accounting:
#   fp16 params 14 GB + fp16 grads 14 GB + Adam(fp32 master + m + v) 84 GB
#   = ~112 GB of model states total.
#
# Under ZeRO-2 on 4 GPUs, params are NOT sharded but grads and optimizer are:
#   params   fp16, unsharded         ~13.5 GB/GPU
#   grads    fp16, sharded /4        ~3.4 GB/GPU  (+ reduce bucket)
#   Adam     fp32 master+m+v, /4     ~21 GB/GPU
#   -------------------------------------------------
#   model states                     ~38-42 GB/GPU, flat in batch size
#
# That leaves ~38 GB/GPU of the 80 GB for activations — which is why CPU offload
# is no longer needed. Offload was LOAD-BEARING at 2x24 GB; at 4x80 GB it is pure
# PCIe slowdown for no benefit, so it is gone along with the ~128 GB host-RAM
# requirement it imposed.
#
# Batch-scaling terms (same as the other two scripts):
#   - the frozen BulkFormer forward's fp32 [B, 20010, 643] intermediates — the
#     dominant per-sample cost;
#   - LLM activations, which are cheap here: the tower mean-pools to ONE token
#     per sample, and Stage-2 sequences are short (measured on the real bundle:
#     mean 141, p95 190, max 199 tokens).
# Budgeted ~1 GB/sample => batch 8 lands near ~50 GB/GPU, ~30 GB under the
# ceiling. Deliberately conservative: this is the tightest of the three scripts
# and the one number nobody has measured yet.
#
# -----------------------------------------------------------------------------
# VERIFY PROBE — run this first, on the 4xA100 box, BEFORE any real training
# -----------------------------------------------------------------------------
#   Terminal A:  VERIFY=1 bash scripts/train/train_stage2_full.sh
#   Terminal B:  nvidia-smi --query-gpu=index,memory.used,memory.total \
#                  --format=csv -l 1 | tee /tmp/stage2_full_mem.log
#
# VERIFY=1 forces --max_steps 2, writes to a throwaway output dir and disables
# reporting, but KEEPS the real per-device batch size — the point is to measure
# the batch you will actually train with. If it OOMs, report that rather than
# silently reverting to ZeRO-3 + offload.
# =============================================================================

set -euo pipefail

# ---- data -------------------------------------------------------------------
# The real materialized Stage-2 bundle — IDENTICAL to train_stage2_lora.sh.
# (Was integration/data/finetune.json, the synthetic smoke-test fixture from
# build_dataset_json.py. That path does not exist in this checkout, and training
# the full-FT arm on different data than the LoRA arm would make the two
# incomparable regardless.)
DATA_PATH=${DATA_PATH:-data/cvd_transcriptome/text_files/stage2_train.json}
IMAGE_PATH=${IMAGE_PATH:-data/cvd_transcriptome/embeddings}

# ---- model ------------------------------------------------------------------
LLM_VERSION=lmsys/vicuna-7b-v1.5
VT_VERSION=bulkformer:$(pwd)/integration/bulkformer_hf_config
VT_VERSION2=""
CN_VERSION=transcript_linear
CONV_VERSION=llama          # this repo's name for the Vicuna v1 template
VERSION=bulkformer-127m
TRAIN_RECIPE=common         # NOT 'lora' — see DEVIATIONS/tune-type note below
MODEL_MAX_LENGTH=2048

# -----------------------------------------------------------------------------
# STAGE-1 CHECKPOINT CONSISTENCY (load-bearing — do not "simplify")
# -----------------------------------------------------------------------------
# This script and train_stage2_lora.sh MUST fork from the same Stage-1 output,
# or the full-FT vs LoRA comparison is confounded at the starting point.
# The default below is byte-identical to train_stage2_lora.sh's STAGE1_CKPT
# default, which in turn matches train_stage1.sh's OUTPUT_DIR. Keep all three in
# sync if any one changes; the guards below fail loudly if they drift.
#
# NOTE: the old PRETRAINED_PATH was built from
#   VT_VARIANT="${VT_VERSION#*/}"
# where VT_VERSION is "bulkformer:/abs/path/to/integration/bulkformer_hf_config".
# `#*/` strips only through the FIRST slash, so VT_VARIANT expanded to most of an
# absolute path and the checkpoint dir name became a nested garbage path that
# matched nothing Stage 1 writes. Same bug train_stage1.sh already fixed by
# dropping the variant expansion entirely. Removed here for the same reason.
# -----------------------------------------------------------------------------
STAGE1_CKPT=${STAGE1_CKPT:-./checkpoints/tiny-llava-vicuna-7b-${VERSION}-pretrain}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/tiny-llava-vicuna-7b-${VERSION}-stage2-full}
RUN_NAME=$(basename "$OUTPUT_DIR")

# Mirror of train_stage2_lora.sh constraint (3): a Stage-1 dir containing "lora"
# sends training_recipe/base.py::load down the PeftModel-merge path.
case "$STAGE1_CKPT" in *lora*)
  echo "ERROR: STAGE1_CKPT must not contain 'lora' — base.py::load would try to merge adapters." >&2; exit 1;; esac
# Inverse of the LoRA script's constraint (2): this is NOT a LoRA run, so the
# output dir must NOT contain "lora" or load_model.py:35-39 will take the
# adapter-merging load path on a checkpoint that has no adapters.
case "$OUTPUT_DIR" in *lora*)
  echo "ERROR: OUTPUT_DIR must not contain 'lora' — this is the full-FT arm; load_model.py would take the adapter path." >&2; exit 1;; esac

if [ ! -d "$STAGE1_CKPT/connector" ]; then
  echo "ERROR: no Stage-1 connector at $STAGE1_CKPT/connector — run scripts/train/train_stage1.sh first." >&2
  exit 1
fi

# Both Stage-2 arms must start from the same Stage-1 checkpoint. If the LoRA
# script is present, resolve ITS default and refuse on drift. The default line
# there is  STAGE1_CKPT=${STAGE1_CKPT:-<literal>}  — pull out <literal> and
# expand it with only VERSION in scope.
_LORA_SH="$(dirname "${BASH_SOURCE[0]}")/train_stage2_lora.sh"
if [ -f "$_LORA_SH" ]; then
  _RAW=$(sed -n 's/^STAGE1_CKPT=\${STAGE1_CKPT:-\(.*\)}$/\1/p' "$_LORA_SH" | head -1)
  if [ -n "$_RAW" ]; then
    _LORA_CKPT="${_RAW//\$\{VERSION\}/$VERSION}"
    _LORA_CKPT="${_LORA_CKPT//\$VERSION/$VERSION}"
    if [ "$_LORA_CKPT" != "$STAGE1_CKPT" ]; then
      echo "ERROR: Stage-1 checkpoint drift between the two Stage-2 arms:" >&2
      echo "         full-FT : $STAGE1_CKPT" >&2
      echo "         LoRA    : $_LORA_CKPT" >&2
      echo "       They must fork from the same Stage 1, or the comparison is confounded." >&2
      echo "       Override STAGE1_CKPT on both if the divergence is intentional." >&2
      exit 1
    fi
  fi
fi

# ---- hardware / batch / step config ------------------------------------------
GPUS="${GPUS:-0,1,2,3}"                       # 4x A100-80GB (was 0,1 on 2x4090)
NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPUS")
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPUS}"
# 2 -> 16 per device, accum 16 -> 2. Estimated peak ~42 GB/GPU (see budget above),
# leaving ~38 GB of the 80 GB free.
#
# NOTE — this DOES change the global batch: 4 x 16 x 2 = 128, was 2 x 2 x 16 = 64.
# Deliberate, and the one non-hardware change in this retune:
#   - it matches train_stage2_lora.sh's global batch (128), so the LoRA vs
#     full-FT comparison differs only in the tuning method;
#   - 128 is also upstream scripts/train/finetune.sh's global batch, which the
#     old 2x4090 config had halved only because that hardware was too slow.
# To restore the old global batch of 64, run with GRAD_ACCUM=1.
PER_DEVICE_BS="${PER_DEVICE_BS:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NUM_EPOCHS=1
STEP_ARGS=(--num_train_epochs ${NUM_EPOCHS})
REPORT_TO=tensorboard

if [[ "${VERIFY:-0}" == "1" ]]; then
    echo "=== VERIFY MODE: 2 steps at the REAL per-device batch ${PER_DEVICE_BS}, throwaway output ==="
    GRAD_ACCUM=1
    STEP_ARGS=(--max_steps 2)
    OUTPUT_DIR=/tmp/stage2_full_verify_out
    RUN_NAME=stage2-full-verify
    REPORT_TO=none
fi

deepspeed --include "localhost:${GPUS}" --master_port 29501 tinyllava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower frozen \
    --tune_type_connector full \
    --group_by_modality_length True \
    --pretrained_model_path "$STAGE1_CKPT" \
    --output_dir $OUTPUT_DIR \
    "${STEP_ARGS[@]}" \
    --per_device_train_batch_size $PER_DEVICE_BS \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to $REPORT_TO \
    --tokenizer_use_fast False \
    --run_name $RUN_NAME

# =============================================================================
# DEVIATIONS from scripts/train/finetune.sh — each is intentional
# =============================================================================
# 1. --deepspeed scripts/zero2.json (was zero3_offload.json; finetune.sh: zero3.json)
#      At 4x80 GB the ZeRO-2 model-state budget is ~38-42 GB/GPU, so neither
#      ZeRO-3 nor CPU offload is needed. ZeRO-2 also matches the PI's specified
#      config and train_stage2_lora.sh, making the two Stage-2 arms comparable.
#      scripts/zero3_offload.json is retained as a documented fallback only.
#
# 2. --include localhost:0,1,2,3 (was 4,5,6,7 upstream; 0,1 at the 2x4090 retune)
#      finetune.sh hardcodes an 8-GPU box's last 4 devices. This targets the
#      4xA100 box's devices 0-3. Override with GPUS=...
#
# 3. --bf16 True (was --fp16 True)
#      A100 is Ampere and supports bf16 natively. Full FT of 7B under fp16 with
#      dynamic loss scaling is prone to scaler collapse; bf16 removes that
#      failure mode at equal memory cost.
#      NOTE: bf16/fp16 in zero2.json are both "auto" — HF Trainer sets whichever
#      flag is passed here, so the config needs no edit to switch back.
#
# 4. --per_device_train_batch_size 8, grad accum 2 (was 2 / 16)
#      Global batch is unchanged at 64 (4 x 8 x 2 == 2 x 2 x 16) — this is a pure
#      hardware retune, not an optimization change. Note the LoRA arm's global
#      batch is 128; see hardware_retune_report.md for that open discrepancy.
#
# 5. --output_dir under ./checkpoints (was /mnt/data/sata/yinghu/...)
#      That path is the upstream authors' machine and does not exist here.
#
# 6. --dataloader_num_workers 4 (was 8), --per_device_eval_batch_size 1 (was 4)
#      Kept at 4 from the offload-era retune. With offload gone the host-RAM
#      pressure that motivated it is gone too; 8 (matching the other two
#      scripts) is now safe if dataloading shows up as a bottleneck.
#
# 7. Dropped --tune_vision_tower_from_layer 0
#      Only read on the 'partially-tune' path (training_recipe/base.py:57-74);
#      inert when tune_type_vision_tower is 'frozen'.
#
# TUNE-TYPE NOTE (verified in source, not just docs):
#   --tune_type_llm full hits training_recipe/base.py:43-44 ->
#   model.language_model.requires_grad_(True), i.e. every LLM parameter trains.
#   LoRA is a SEPARATE axis: it requires --training_recipe lora (arguments.py:50
#   defaults to 'common') AND tune_type_llm == 'lora' (lora_recipe.py:29). With
#   TRAIN_RECIPE=common above, there is no LoRA path reachable from this script.
#   The frozen encoder is controlled by the separate --tune_type_vision_tower.
#
# EXPECTED TRAINABLE PARAMS (task 5 — check this in the VERIFY run's log):
#   ~6.7B LLM + a few M connector = essentially the full 7B trainable;
#   BulkFormer tower ~0 trainable. If you instead see ~M-scale trainable, the
#   run has silently taken a LoRA/frozen path and must not be treated as full FT.
#   Under ZeRO-2 params are NOT sharded, so each rank should report the FULL
#   ~6.7B trainable — unlike the old ZeRO-3 setup where a per-rank ~3.4B shard
#   count was the expected signature. A ~M-scale count still means the run has
#   silently taken a LoRA/frozen path and must not be treated as full FT.
# =============================================================================
