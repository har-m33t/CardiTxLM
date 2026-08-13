"""stage2_lora_dryrun.py — 5-step dry run of scripts/train/train_stage2_lora.sh.

Drives the REAL entrypoint (`tinyllava.train.train.train()`) through the REAL
CLI parser with the REAL Stage-2 LoRA flag set, so that argument names, the
`lora` training recipe, the PEFT wrap, the Stage-1 checkpoint load, the
`llama` conv template and the `.npy` data path are all exercised as they would
be on the cluster. It then asserts that what is trainable is LoRA adapters +
connector, not the whole LLM.

Four host-imposed substitutions (each already an accepted, separately-tracked
boundary in this project — nothing LoRA-specific is faked):

  1. BulkFormer encoder -> shape-correct STUB. `torch_sparse` is not installed on
     this host, so the real encoder cannot be constructed (the support tensors,
     including esm2_feature_concat.pt, ARE present now). The real-encoder gate is
     tracked in integration/smoke_test_result.md. The STUB still resolves the
     variant, its dims and its checkpoint path from the real `_VARIANTS` table,
     and this harness separately reads the real BulkFormer-93M `.pt` off disk to
     confirm the locked scale — see `verify_locked_scale`.
  2. Vicuna-7B depth 32 -> 2 layers. Real hidden_size (4096), vocab (32000),
     intermediate (11008) and the real Vicuna tokenizer are kept. Same
     concession as integration/vicuna_smoke_test.py; 24 GB host.
  3. Stage-1 checkpoint -> PLACEHOLDER with randomly initialised weights of the
     correct shapes (see --stage1-ckpt / _make_placeholder_stage1). It proves the
     load PATH, not that any real Stage-1 training happened.
  4. deepspeed/CUDA/flash-attn/bf16 -> single-process CPU, eager attention, fp32.

Run:  python -m integration.stage2_lora_dryrun
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

from integration.smoke_test import N_GENES, CFG_DIR, REPO, _patch_encoder

VICUNA_HF = "lmsys/vicuna-7b-v1.5"
REDUCED_LAYERS = 2
SCRIPT = REPO / "scripts" / "train" / "train_stage2_lora.sh"

# The encoder scale is LOCKED project-wide. Anything else is a bug, not an option.
LOCKED_VARIANT = "BulkFormer-93M"
LOCKED_VISION_HIDDEN = 515          # BulkFormer-93M dim 512 + 3


# --------------------------------------------------------------------------
# locked-scale verification (independent of the stubbed encoder)
# --------------------------------------------------------------------------
def verify_locked_scale():
    """Prove the pipeline selects BulkFormer-93M, and that the real 93M
    checkpoint on disk is what the tower would load.

    The encoder itself is stubbed here (no torch_sparse), so this does the next
    strongest thing: it resolves the variant exactly the way
    BulkFormerVisionTower.__init__ does, then opens the real `.pt` the resolved
    spec names and checks its tensor widths are the 93M architecture (dim=512),
    not 127M's (dim=640).
    """
    import tinyllava.model.vision_tower.bulkformer as bf
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(CFG_DIR))
    cfg = getattr(cfg, "vision_config", cfg)
    variant = getattr(cfg, "bulkformer_variant", None)
    report("locked-scale/config", dir=CFG_DIR.name, variant=variant,
           hidden_size=cfg.hidden_size)
    assert variant == LOCKED_VARIANT, \
        f"tower config selects {variant!r}, not the locked {LOCKED_VARIANT!r}"
    assert cfg.hidden_size == LOCKED_VISION_HIDDEN, \
        f"config hidden_size {cfg.hidden_size} != {LOCKED_VISION_HIDDEN} (93M dim+3)"

    spec = bf._VARIANTS[variant]
    ckpt = bf._CKPT_ROOT / "models" / spec["ckpt"]
    report("locked-scale/spec", dim=spec["dim"], p_repeat=spec["p_repeat"],
           ckpt=spec["ckpt"], exists=ckpt.exists())
    assert spec["dim"] + 3 == LOCKED_VISION_HIDDEN
    assert ckpt.exists(), f"locked checkpoint missing: {ckpt}"

    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    widths = {tuple(v.shape)[-1] for v in raw.values()
              if hasattr(v, "shape") and v.dim() >= 1}
    n_params = sum(v.numel() for v in raw.values() if hasattr(v, "numel"))
    report("locked-scale/checkpoint", file=ckpt.name,
           n_tensors=len(raw), n_params=f"{n_params:,}",
           encoder_dim_present=spec["dim"] in widths,
           mb=f"{ckpt.stat().st_size / 1e6:.0f}")
    assert spec["dim"] in widths, \
        f"{ckpt.name} has no dim-{spec['dim']} tensors — wrong variant on disk"
    # 127M/147M are the only wider variants (dim 640); their width must be absent.
    wider = sorted({s["dim"] for s in bf._VARIANTS.values() if s["dim"] > spec["dim"]})
    assert not (widths & set(wider)), \
        f"{ckpt.name} carries width(s) {sorted(widths & set(wider))} — that is a " \
        f"larger variant ({wider}), not {variant}"
    del raw
    return variant, spec


# --------------------------------------------------------------------------
# tiny synthetic Stage-2 dataset — one item per Stage-2 category
# --------------------------------------------------------------------------
def make_stage2_dataset(root: Path, repeats: int = 4):
    """Stage-2-format items covering all three categories.

    Phrasing/structure is copied from the real materialized bundle
    (data/cvd_transcriptome/text_files/stage2_train.json) so the tokenized
    shapes are representative — in particular the comparative item carries the
    neg_hard comparison group (the only group gt_functions.py accepts for
    comparative_differential_reasoning) bound into both question and answer,
    and its answer is the long multi-sentence form.
    """
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    templates = [
        # --- disease_subtype_classification ---------------------------------
        ("disease_subtype_classification",
         "<image>\nBased on the gene expression profile, which cardiovascular "
         "condition is most probable?",
         "This profile is consistent with coronary artery disease."),
        # --- comparative_differential_reasoning (neg_hard comparison group) --
        ("comparative_differential_reasoning",
         "<image>\nWhich transcriptomic features distinguish coronary artery "
         "disease from tissue-matched samples without a confirmed "
         "cardiovascular diagnosis?",
         "When compared to tissue-matched samples without confirmed "
         "cardiovascular disease (n = 22,307, from the neg_hard pool), "
         "disease-confirmed cardiovascular samples (n = 8,725) are separable "
         "with a ROC-AUC of 0.7806 (sd 0.1055, linear probe ROC-AUC, grouped "
         "5-fold by series, BulkFormer-37M). This sample is labeled coronary "
         "artery disease. A per-gene differential expression analysis has not "
         "been performed for this comparison, so no individual genes are "
         "attributed to the difference."),
        # --- gene_driver_reasoning ------------------------------------------
        ("gene_driver_reasoning",
         "<image>\nBased on the gene expression ranking, identify the most "
         "informative genes for cardiovascular disease.",
         "Top 10 stable elastic-net signal genes for cardiovascular disease "
         "vs. random bulk tissue (nonzero in all 5 CV folds): FCGR2A, GFRA1, "
         "BGN, TTN, ROBO1, CHCHD2, HSPG2, SLC6A8, CELF2, NAV2."),
    ]

    records = []
    for r in range(repeats):
        for c, (category, q, a) in enumerate(templates):
            sid = f"GSM{r}{c:02d}"
            np.save(img_dir / f"{sid}.npy", rng.normal(size=N_GENES).astype(np.float32))
            records.append({
                "id": sid, "image": f"{sid}.npy", "category": category,
                "conversations": [{"from": "human", "value": q},
                                  {"from": "gpt", "value": a}],
            })
    out = root / "stage2_tiny.json"
    out.write_text(json.dumps(records, indent=1))
    return out, img_dir, records


# --------------------------------------------------------------------------
# backbone + PLACEHOLDER Stage-1 checkpoint
# --------------------------------------------------------------------------
def _make_vicuna_backbone(scratch: Path, n_layers: int = REDUCED_LAYERS):
    """Reduced-depth, real-config Vicuna in a dir whose name contains 'vicuna'
    so LLMFactory's substring match picks the real vicuna loader."""
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

    out = scratch / "vicuna_stub"
    cfg = AutoConfig.from_pretrained(VICUNA_HF)
    real_layers = cfg.num_hidden_layers
    cfg.num_hidden_layers = n_layers
    if not (out / "config.json").exists():
        LlamaForCausalLM(cfg).save_pretrained(out)
    AutoTokenizer.from_pretrained(VICUNA_HF).save_pretrained(out)
    return out, cfg, real_layers


def _make_placeholder_stage1(scratch: Path, backbone: Path, vision_hidden: int,
                             llm_hidden: int) -> Path:
    """PLACEHOLDER Stage-1 checkpoint — NOT a trained model.

    Reproduces exactly the layout BaseTrainingRecipe.save() writes for the
    pretrain stage, so BaseTrainingRecipe.add_args/load exercise their real
    paths:  <ckpt>/language_model/{config.json,weights}
            <ckpt>/connector/pytorch_model.bin   (keys: _connector.{weight,bias})
            <ckpt>/vision_tower/                 (ignored by BulkFormerVisionTower)

    The connector tensors are RANDOM, with a deliberately distinctive scale so
    the dry run can prove they were actually read back into the live model.
    """
    from transformers import LlamaForCausalLM
    import shutil

    ckpt = scratch / "stage1_placeholder_pretrain"   # note: no 'lora' in the name
    (ckpt / "vision_tower").mkdir(parents=True, exist_ok=True)
    (ckpt / "connector").mkdir(parents=True, exist_ok=True)

    lm_dir = ckpt / "language_model"
    if not (lm_dir / "config.json").exists():
        lm_dir.mkdir(parents=True, exist_ok=True)
        LlamaForCausalLM.from_pretrained(str(backbone)).save_pretrained(lm_dir)
    for f in ("tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"):
        if (backbone / f).exists():
            shutil.copy(backbone / f, lm_dir / f)

    torch.manual_seed(1234)
    weight = torch.randn(llm_hidden, vision_hidden) * 0.01234
    bias = torch.randn(llm_hidden) * 0.01234
    torch.save({"_connector.weight": weight, "_connector.bias": bias},
               ckpt / "connector" / "pytorch_model.bin")
    return ckpt, weight, bias


# --------------------------------------------------------------------------
# cross-check: the flags we run are the flags the shell script ships
# --------------------------------------------------------------------------
# Flags the dry run must diverge on, with why. Everything else has to match.
HOST_OVERRIDES = {
    "--deepspeed":            "no multi-GPU/deepspeed launcher on this host",
    "--attn_implementation":  "flash_attention_2 requires CUDA",
    "--bf16":                 "CPU run is fp32",
    "--num_train_epochs":     "replaced by --max_steps 5",
    "--per_device_train_batch_size": "1 to fit 24 GB host",
    "--gradient_accumulation_steps": "1 for a 5-step run",
    "--dataloader_num_workers": "0 for a single-process run",
    "--report_to":            "no tensorboard for a dry run",
    "--save_strategy":        "nothing is saved by a dry run",
    "--save_steps":           "nothing is saved by a dry run",
    "--save_total_limit":     "nothing is saved by a dry run",
    "--gradient_checkpointing": "disabled: needs a real backward graph on GPU",
    "--per_device_eval_batch_size": "no eval in a dry run",
}


def script_flags() -> set:
    text = SCRIPT.read_text()
    # start after "train.py" so deepspeed's own launcher flags (--include,
    # --master_port) are not mistaken for train.py arguments
    body = text.split("tinyllava/train/train.py", 1)[1].split("\n\n", 1)[0]
    return set(re.findall(r"(--[a-z0-9_]+)", body))


def check_flag_parity(argv: list) -> tuple:
    ours = set(t for t in argv if t.startswith("--"))
    theirs = script_flags()
    missing = sorted(theirs - ours - set(HOST_OVERRIDES))
    extra = sorted(ours - theirs - {"--max_steps", "--use_cpu", "--fp16"})
    return missing, extra


def report(tag, **kv):
    print(f"[{tag}] " + " ".join(f"{k}={v}" for k, v in kv.items()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--n-layers", type=int, default=REDUCED_LAYERS)
    args = ap.parse_args(argv)

    bar = "=" * 74
    print(bar)
    print("STAGE-2 LoRA DRY RUN — real train.py, real CLI parser, real lora recipe")
    print(f"  encoder=STUB(scale-locked {LOCKED_VARIANT})  llm-depth=reduced")
    print("  stage1-ckpt=PLACEHOLDER  device=cpu")
    print(bar)

    # scale lock, checked against the real checkpoint on disk, before anything runs
    locked_variant, locked_spec = verify_locked_scale()

    _patch_encoder()

    scratch = REPO / "integration" / "_stage2_lora_dryrun_tmp"
    scratch.mkdir(parents=True, exist_ok=True)

    # single-process gloo group: tinyllava.utils.logging.log() calls dist.get_rank()
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:29517")

    from transformers import AutoConfig

    backbone, vcfg, real_layers = _make_vicuna_backbone(scratch, args.n_layers)
    tower_cfg = AutoConfig.from_pretrained(str(CFG_DIR))
    vision_hidden = getattr(tower_cfg, "vision_config", tower_cfg).hidden_size
    assert vision_hidden == LOCKED_VISION_HIDDEN
    stage1, conn_w, conn_b = _make_placeholder_stage1(
        scratch, backbone, vision_hidden, vcfg.hidden_size)
    report("setup", real_vicuna_layers=real_layers, dryrun_layers=args.n_layers,
           hidden=vcfg.hidden_size, vision_hidden=vision_hidden,
           stage1_ckpt=stage1.name)

    data_json, img_dir, records = make_stage2_dataset(scratch)
    cats = sorted({r["category"] for r in records})
    report("dataset", n_items=len(records), categories=",".join(cats))
    assert len(cats) == 3, "all three Stage-2 categories must be present"
    assert any("neg_hard" in r["conversations"][1]["value"] for r in records
               if r["category"] == "comparative_differential_reasoning"), \
        "comparative item must carry the neg_hard comparison group"

    out_dir = scratch / "out-stage2-lora"   # 'lora' in the name, per constraint (2)

    train_argv = [
        "train.py",
        "--data_path", str(data_json),
        "--image_folder", str(img_dir),
        "--is_multimodal", "True",
        "--conv_version", "llama",
        "--model_name_or_path", str(backbone),
        "--vision_tower", f"bulkformer:{CFG_DIR}",
        "--vision_tower2", "",
        "--connector_type", "transcript_linear",
        "--mm_vision_select_layer", "-2",
        "--image_aspect_ratio", "square",
        "--attn_implementation", "eager",
        "--fp16", "False",
        "--training_recipe", "lora",
        "--tune_type_llm", "lora",
        "--tune_type_vision_tower", "frozen",
        "--tune_vision_tower_from_layer", "0",
        "--tune_type_connector", "full",
        "--bits", "16",
        "--lora_r", "128",
        "--lora_alpha", "256",
        "--lora_dropout", "0.05",
        "--lora_bias", "none",
        "--group_by_modality_length", "False",
        "--pretrained_model_path", str(stage1),
        "--output_dir", str(out_dir),
        "--max_steps", str(args.max_steps),
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--evaluation_strategy", "no",
        "--save_strategy", "no",
        "--learning_rate", "2e-4",
        "--mm_projector_lr", "2e-5",
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "False",
        "--model_max_length", "2048",
        "--gradient_checkpointing", "False",
        "--dataloader_num_workers", "0",
        "--lazy_preprocess", "True",
        "--report_to", "none",
        "--tokenizer_use_fast", "False",
        "--run_name", "stage2-lora-dryrun",
        "--use_cpu", "True",
    ]

    missing, extra = check_flag_parity(train_argv)
    report("flag-parity", script_flags=len(script_flags()),
           missing_from_dryrun=missing or "none", not_in_script=extra or "none")
    assert not missing, f"dry run does not exercise script flags: {missing}"
    assert not extra, f"dry run uses flags the script does not ship: {extra}"

    # ---- run the real entrypoint ------------------------------------------
    # tinyllava.train.__init__ re-exports train(), which shadows the submodule
    # under plain `import ... as` — go through importlib to get the module.
    import importlib
    train_mod = importlib.import_module("tinyllava.train.train")

    captured = {}
    orig_trainer_cls = train_mod.LLaVATrainer

    class _ProbedTrainer(orig_trainer_cls):
        def __init__(self, model=None, **kw):
            captured["model"] = model
            super().__init__(model=model, **kw)

    train_mod.LLaVATrainer = _ProbedTrainer

    old_argv = sys.argv
    sys.argv = train_argv
    try:
        train_mod.train()
    finally:
        sys.argv = old_argv
        train_mod.LLaVATrainer = orig_trainer_cls

    model = captured["model"]

    # ---- assertions on WHAT trained ---------------------------------------
    from peft import PeftModel
    assert isinstance(model, PeftModel), f"model was not PEFT-wrapped: {type(model)}"

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_n = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad and "lora_" in n)
    conn_n = sum(p.numel() for n, p in model.named_parameters()
                 if p.requires_grad and ".connector." in n)
    llm_base_n = sum(p.numel() for n, p in model.named_parameters()
                     if p.requires_grad and "language_model" in n and "lora_" not in n)
    tower_n = sum(p.numel() for n, p in model.named_parameters()
                  if p.requires_grad and "vision_tower" in n)

    report("trainable", total=total, trainable=trainable,
           pct=f"{100 * trainable / total:.3f}%", lora=lora_n, connector=conn_n,
           llm_base=llm_base_n, tower=tower_n)
    assert lora_n > 0, "no LoRA parameters are trainable"
    assert llm_base_n == 0, "base LLM weights are trainable — this is not LoRA"
    assert tower_n == 0, "frozen BulkFormer tower has trainable params"
    assert trainable == lora_n + conn_n, "unexpected trainable params outside lora+connector"

    # LoRA must live on the LLM only, never on tower/connector
    lora_mods = {n.split(".lora_")[0] for n, _ in model.named_parameters() if "lora_" in n}
    assert all("language_model" in m for m in lora_mods), \
        f"LoRA applied outside the LLM: {sorted(m for m in lora_mods if 'language_model' not in m)}"
    report("lora-placement", n_adapted_modules=len(lora_mods),
           all_in_llm=True, example=sorted(lora_mods)[0])

    # ---- project the same config to the REAL 32-layer 7B -------------------
    # LoRA params scale exactly linearly in depth (identical target modules per
    # layer), so the reduced-depth count extrapolates without approximation.
    per_layer_lora = lora_n // args.n_layers
    full_lora = per_layer_lora * real_layers
    h, inter, vocab = vcfg.hidden_size, vcfg.intermediate_size, vcfg.vocab_size
    full_base = (real_layers * (4 * h * h + 3 * h * inter + 2 * h)
                 + 2 * vocab * h + h)
    full_trainable = full_lora + conn_n
    full_total = full_base + full_trainable
    report("full-depth-projection", layers=real_layers,
           base_llm_params=f"{full_base:,}", lora=f"{full_lora:,}",
           connector=f"{conn_n:,}", trainable=f"{full_trainable:,}",
           pct_of_model=f"{100 * full_trainable / full_total:.2f}%")
    assert abs(full_base - 6_738_415_616) < 1_000_000, \
        f"projected base is not Vicuna-7B: {full_base:,}"
    assert full_trainable / full_total < 0.10, "trainable fraction is not LoRA-sized"

    # ---- the LIVE model is the locked scale -------------------------------
    base = model.base_model.model
    live_variant = base.vision_tower._variant
    live_embed = base.vision_tower.embed_dim
    report("live-tower", variant=live_variant, embed_dim=live_embed,
           would_load=locked_spec["ckpt"],
           config_variant=base.config.vision_config.bulkformer_variant)
    assert live_variant == LOCKED_VARIANT == locked_variant, \
        f"live tower is {live_variant}, not the locked {LOCKED_VARIANT}"
    assert live_embed == LOCKED_VISION_HIDDEN

    # ---- Stage-1 connector really was loaded ------------------------------
    live_w = base.connector._connector.weight.detach().float().cpu()
    live_b = base.connector._connector.bias.detach().float().cpu()
    # 5 optimizer steps moved the connector slightly; it must still be the
    # placeholder Stage-1 tensor, not a fresh random init.
    w_delta = (live_w - conn_w).abs().max().item()
    report("stage1-connector-load", shape=tuple(live_w.shape),
           max_abs_delta_vs_ckpt=f"{w_delta:.3e}",
           bias_delta=f"{(live_b - conn_b).abs().max().item():.3e}")
    assert tuple(live_w.shape) == (vcfg.hidden_size, vision_hidden)
    assert w_delta < 1e-2, "connector in the live model is not the Stage-1 checkpoint"

    print("\n" + bar)
    print("STAGE-2 LoRA DRY RUN PASSED")
    print(f"  encoder scale locked: {LOCKED_VARIANT} "
          f"(embed {LOCKED_VISION_HIDDEN}, ckpt {locked_spec['ckpt']})")
    print(f"  {args.max_steps} optimizer steps completed through the real train.py")
    print(f"  trainable {trainable:,} / {total:,} params = {100*trainable/total:.3f}%")
    print("  (LoRA adapters + connector only; base LLM and tower frozen)")
    print(bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
