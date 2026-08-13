# Deprecated — retired Stage-2 full fine-tune branch

Archived 2026-08-12. **Nothing in here is part of the active pipeline.** These
files are kept for provenance only; do not launch them and do not reference them
from new scripts.

| File | Was | Why retired |
|---|---|---|
| `train_stage2_full.sh` | Stage-2 full fine-tune of Vicuna-7B + connector (the comparison arm to `../train_stage2_lora.sh`) | Stage 2 is now **LoRA-only** by direct instruction. The full-FT arm is not being run, so it is no longer maintained. |
| `zero3_offload.json` | DeepSpeed ZeRO-3 + CPU-offload config | Only ever needed for the full fine-tune's ~112 GB optimizer state. The active pipeline (Stage 1 and Stage-2 LoRA) runs **ZeRO-2** via `scripts/zero2.json`. |

Two further reasons they must not be reused as-is:

1. `train_stage2_full.sh` still carries `VERSION=bulkformer-127m` and the 643-dim
   tower assumption. The project has since **locked BulkFormer-93M** (515-dim
   tower embedding) across the pipeline; this script was deliberately not updated
   because it is retired. See `llm_training_plan.md` §2.
2. It also carries an unresolved `!! GPU MEMORY UNMEASURED !!` banner — its
   budget was arithmetic, never measured on an A100.

The active scripts are `scripts/train/train_stage1.sh` and
`scripts/train/train_stage2_lora.sh`.
