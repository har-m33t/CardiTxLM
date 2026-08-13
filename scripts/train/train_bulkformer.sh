#!/bin/bash
# ⚠ SUPERSEDED (2026-08-13). The pipeline now runs as two standalone launchers:
#     Stage 1  -> scripts/train/train_stage1.sh
#     Stage 2  -> scripts/train/train_stage2_lora.sh   (LoRA-only)
#   This script's stage 2 is a FULL fine-tune via finetune.sh, which is the
#   retired branch (see scripts/train/deprecated/). Kept for provenance; it is
#   not the script to launch. See llm_training_plan.md.
#
# Two-stage BulkFormer training launcher — mirrors scripts/train/train_phi.sh.
#
#   Stage 1 (pretrain.sh):  connector-only alignment. BulkFormer tower + LLM
#                           frozen; only the connector trains.
#   Stage 2 (finetune.sh):  instruction tuning. LLM + connector train; the
#                           BulkFormer tower stays frozen.
#
# The BulkFormer encoder is frozen throughout (standing project decision) — the
# two reference stages already keep --tune_type_vision_tower frozen, so no extra
# flags are needed. See integration/repo_findings.md §9.
#
# NOTE: the shared pretrain.sh/finetune.sh assume a CUDA box (deepspeed
# --include localhost:4,5,6,7, flash_attention_2, --fp16). Adapt those to your
# target machine (GPU indices, output_dir, attn impl). BulkFormer's forward is
# CPU-only today (torch_sparse GCNConv has no MPS/where-needed kernel), so a
# real run belongs on a CUDA machine where the frozen tower runs on CPU/GPU as
# available.

# Data built by:  python -m integration.build_dataset_json --n-per-class 200
DATA_PATH=integration/data/pretrain.json          # stage-1 alignment annotations
FINETUNE_DATA_PATH=integration/data/finetune.json  # stage-2 instruction annotations
IMAGE_PATH=integration/data/images                 # dir of per-sample .npy expression vectors
FINETUNE_IMAGE_PATH=integration/data/images

# LLM backbone: Vicuna-7B (lmsys/vicuna-7b-v1.5), the confirmed standard backbone.
# Wiring verified end-to-end in integration/vicuna_smoke_test_result.md (real
# vicuna loader + real tokenizer + real hidden dim 4096, loss + gradients OK).
# Two enablers were added to make this a supported path in this repo:
#   * LLM loader: tinyllava/model/llm/vicuna.py registers `vicuna` ->
#     LlamaForCausalLM, so LLMFactory selects it via the "vicuna" substring in
#     the path below.
#   * Conv template: CONV_VERSION=llama is THIS repo's Vicuna v1 format
#     (tinyllava/data/template/llama_template.py) — note it is NOT named
#     `vicuna_v1` here.
# COMPUTE (do not silently change batch sizes here): Stage-2 full fine-tune of 7B
# under ZeRO-3 (scripts/zero3.json, 4 GPUs, no offload) needs ~40 GB-class GPUs
# x4; it will NOT fit 24 GB without gradient checkpointing / ZeRO-3 offload.
# Confirm the target env's per-GPU VRAM before queuing — see the compute-budget
# section of integration/vicuna_smoke_test_result.md.
LLM_VERSION=lmsys/vicuna-7b-v1.5
# Vision tower: the "bulkformer" prefix selects BulkFormerVisionTower via the
# substring-matching factory; the path is the local HF config dir exposing
# hidden_size=515 and bulkformer_variant (BulkFormer-93M — the locked scale).
VT_VERSION=bulkformer:$(pwd)/integration/bulkformer_hf_config
VT_VERSION2=""
CN_VERSION=transcript_linear
CONV_VERSION=llama   # Vicuna v1 template (this repo's name for it); used in stage 2
VERSION=bulkformer-93m
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048

bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
