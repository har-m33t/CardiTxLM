# LLM training plan — CardioLLM (BulkFormer → connector → Vicuna-7B)

> **Provenance note.** No `llm_training_plan.md` existed in this repository or in
> `git log --all` before 2026-08-13; this file was created on that date to hold
> the decisions that were previously scattered across the training scripts and
> the `integration/` verification reports. Section numbering starts here — there
> is no earlier "§2" this supersedes.

Scope of this document: what gets trained, at what scale, with which arguments,
on what hardware. Per-file argument verification lives in
`integration/train_stage1_verification.md` and
`integration/stage2_lora_dryrun_result.md`; the finalization audit lives in
`scripts/train/stage2_lora_finalization_report.md`.

---

## 1. Pipeline shape

Three components, one of which is frozen throughout:

| Component | Implementation | Stage 1 | Stage 2 |
|---|---|---|---|
| Encoder ("vision tower") | `tinyllava/model/vision_tower/bulkformer.py`, **BulkFormer-93M** | frozen | frozen |
| Connector | `tinyllava/model/connector/transcript_linear.py`, `nn.Linear(515 → 4096)` | **full** | **full** |
| LLM | Vicuna-7B v1.5 (`lmsys/vicuna-7b-v1.5`) | frozen | **LoRA adapters only** |

Entry point for both stages: `tinyllava/train/train.py`, which parses
`ModelArguments` / `DataArguments` / `TrainingArguments`
(`tinyllava/utils/arguments.py`) and runs `LLaVATrainer`
(`tinyllava/train/tinyllava_trainer.py`, a `transformers.Trainer` subclass).
The tune-type/LoRA behaviour is selected by `--training_recipe`
(`tinyllava/training_recipe/`).

Launchers:

- `scripts/train/train_stage1.sh` — connector alignment
- `scripts/train/train_stage2_lora.sh` — LoRA instruction tuning

---

## 2. Encoder scale: **BulkFormer-93M — LOCKED**

**93M is the final choice, not a recommendation pending confirmation.** No
further scale comparison is planned, and no script exposes the scale as a
tunable. Concretely this means:

- `integration/bulkformer_hf_config/config.json` →
  `"bulkformer_variant": "BulkFormer-93M"`, `"hidden_size": 515`
- Both launchers hard-code `BULKFORMER_SCALE="93M"` and **fail at launch** if
  that config dir does not select `BulkFormer-93M`
- `BulkFormerVisionTower`'s fallback variant is `BulkFormer-93M`
- Stage-1 checkpoint path is fixed: `checkpoints/stage1-connector-93M`

### Justification (from the completed sweep, not from assertion)

All five variants (37M / 50M / 93M / 127M / 147M) were run to completion under
one protocol — same manifest, same grouped 5-fold splits by series, same seed,
same probe hyperparameters. Source: `linear_probe/results/variant_comparison_table.csv`
and `linear_probe/writeup.md` §5, §8.

Linear-probe ROC-AUC (mean ± std over 5 folds):

| Variant | Pool (a) whole-corpus non-CVD | Pool (b) tissue-matched hard negatives |
|---|---|---|
| 37M | 0.925 ± 0.037 | 0.781 ± 0.105 |
| 50M | 0.928 ± 0.036 | 0.724 ± 0.110 |
| **93M** | **0.944 ± 0.023** | **0.801 ± 0.187** |
| 127M | 0.943 ± 0.027 | 0.720 ± 0.215 |
| 147M | 0.941 ± 0.024 | 0.773 ± 0.180 |

The finding that decides it, quoted from `linear_probe/writeup.md` §5:

> **Pool (a):** "rises from 37M through 93M, then **plateaus** — 93M/127M/147M
> are statistically indistinguishable from each other (ROC-AUC 0.941–0.944,
> PR-AUC 0.888–0.898, all well inside one another's std bars) … Scaling past
> ~93M buys nothing further on this pool."
>
> **Pool (b):** "no clean scaling trend at all. 93M is the single best point
> estimate (ROC-AUC 0.801, PR-AUC 0.672), but 127M … is the *worst* … At n=5
> folds … this reads as **noise dominating any true scale effect**."

So: **93M is the smallest variant at the plateau.** Above it, pool (a) is flat
and pool (b) is noise. Two things follow, and it is worth being precise about
which is load-bearing:

1. **Decisive:** nothing above 93M performs better. 93M is best or tied-best on
   ROC-AUC in both pools, and 127M/147M never beat it outside the std bars.
2. **Tie-breaking, not evidence of superiority:** at equal measured performance,
   93M is the cheaper encoder — 304 MB vs 451 MB checkpoint, and a 515-wide
   embedding vs 643, which shrinks the dominant `[B, 20010, dim]` fp32 activation
   term in both stages by ~20%.

**Honest limits of this justification.** 93M's *lead* on pool (b) is not
statistically separable from the other variants (std bars of ±0.11 to ±0.27 on
PR-AUC overlap every mean). The defensible claim is "performance plateaus at 93M
and larger encoders add nothing," **not** "93M is significantly better than
127M." Also note `writeup.md` §8: at 93M–147M the frozen embedding does not yet
beat the elastic-net baseline (PR-AUC 0.873) on pool (a) — clearing that bar is
the job of the connector + LLM, and is the target the trained pipeline is
measured against.

---

## 3. Stage 1 — connector alignment

`scripts/train/train_stage1.sh`. Trainable: the `transcript_linear` connector
only (515 → 4096 = 2.11 M params). Encoder and LLM frozen.

- Recipe: `--training_recipe common`, `--tune_type_llm frozen`,
  `--tune_type_vision_tower frozen`, `--tune_type_connector full`
- Data: `data/cvd_transcriptome/text_files/stage1_train.json`
  (199,954 QA pairs over 8,553 `[20010]` float32 log1p-TPM `.npy` vectors)
- Output: **`checkpoints/stage1-connector-93M`** — writes `language_model/`,
  `vision_tower/`, `connector/` subdirs via `BaseTrainingRecipe.save`
- Hardware: 4×A100-80GB, DeepSpeed ZeRO-2 (`scripts/zero2.json`)

Open decision carried from `integration/train_stage1_verification.md` §6:
`--conv_version pretrain` discards the question text, which is wrong for
gene-specific QA. Set `CONV_VERSION=llama` before the real run unless the
answer-marginal behaviour is intended.

---

## 4. Stage 2 — **LoRA only**

`scripts/train/train_stage2_lora.sh`. The full fine-tune arm has been retired;
its script and the ZeRO-3-offload config it needed are archived under
`scripts/train/deprecated/` and are not maintained.

- Starts from `--pretrained_model_path ./checkpoints/stage1-connector-93M`
  (loads the Stage-1-*trained* connector, not a fresh init)
- Recipe: `--training_recipe lora`, `--tune_type_llm lora`,
  `--tune_type_connector full`, `--tune_type_vision_tower frozen`, `--bits 16`
- LoRA: `--lora_r 128 --lora_alpha 256 --lora_dropout 0.05 --lora_bias none`
  → `peft.LoraConfig` in `tinyllava/training_recipe/lora_recipe.py`. Target
  modules are computed by `find_all_linear_names`, not passed on the CLI.
- Trainable at full depth: ~322 M of ~7.06 B = **4.56%** (LoRA 319.8 M +
  connector 2.11 M); base LLM and encoder receive no gradients
- `--conv_version llama` — this repo's name for the Vicuna v1.5 template;
  `vicuna_v1` is not registered
- Two silent naming constraints, enforced by guards in the script: `OUTPUT_DIR`
  **must** contain `lora`; `STAGE1_CKPT` must **not**

---

## 5. Hardware / DeepSpeed

4×A100-80GB, **ZeRO-2** for both stages via `--deepspeed ./scripts/zero2.json`.
ZeRO-3 and ZeRO-3+offload are not used: with only the connector (Stage 1) or
adapters + connector (Stage 2) trainable, there is no optimizer state worth
sharding, and 80 GB absorbs the unsharded fp16 weights.

Per-GPU VRAM has never been measured on real A100s — every budget in the scripts
and in `scripts/train/hardware_retune_report.md` is arithmetic. Run the VERIFY
probe on the target box before queuing.

---

## 6. Evaluation

Target: beat the linear-probe floor in `linear_probe/results/BulkFormer-93M/`
(both negative pools), and on pool (a) beat the elastic-net baseline
(PR-AUC 0.873). Comparison arms are **Stage-2 LoRA vs. that baseline** — there is
no full-fine-tune arm to compare against.

**Not yet implemented.** No `evaluate_and_compare.py` exists in this repo. The
protocol for turning generated answers into a comparable score (generation
settings, label extraction from free text, and whether the LLM is scored on the
probe's exact grouped 5-fold splits) has not been specified anywhere, and is a
prerequisite for writing it.

---

## 7. Open gates before any real run

1. **Real-encoder gate — re-opened by the 93M lock.** `python -m
   integration.smoke_test --real-encoder` passed on a CUDA box on 2026-08-11,
   but against **127M** (`integration/smoke_test_result.md`). It must be re-run
   now that 93M is locked; the test needs no edit (it reads the variant from the
   config dir) and should report `(B, 1, 515)`. `torch_sparse` is absent on the
   macOS dev host, so this can only run on the training environment.
2. **Stage 1 must actually run** — no real Stage-1 checkpoint exists yet; the
   Stage-2 dry run used a randomly-initialised placeholder of the correct shapes.
3. **Vicuna-7B weights** have never been downloaded on the dev host (dry runs use
   a 2-layer stand-in with the real config and tokenizer).
4. **VRAM measurement** on the real 4×A100 box.
5. **`CONV_VERSION` decision** for Stage 1 (§3).
