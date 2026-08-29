"""Reconstruct `<stage1-ckpt>/language_model/` from the base Vicuna weights.

WHY THIS IS EXACT, NOT AN APPROXIMATION
---------------------------------------
`training_recipe/base.py:27` points the Stage-2 launch at
`<pretrained_model_path>/language_model`, and `load_llm` then loads the LLM from
there instead of from `--model_name_or_path`. That directory is written by the
Stage-1 run — but Stage 1 trains the CONNECTOR ONLY:

    scripts/train/train_stage1.sh:  --tune_type_llm frozen
    modeling_tinyllava.load_llm:    self.language_model.requires_grad_(False)

so the weights it saves are the base `lmsys/vicuna-7b-v1.5` weights, unchanged.
Rebuilding the directory from the same HF checkpoint therefore reproduces it
exactly; there is nothing else it could contain.

This is needed because `checkpoints/*` is gitignored (`.gitignore:59`), so a
fresh pod gets the 4.2 MB trained connector by direct transfer but not the
13.5 GB of frozen LLM weights that sit beside it — which would be pointless to
carry when the identical bytes are already being downloaded from HuggingFace.

The connector is the ONLY Stage-1 artifact that carries training. It must be
copied, never regenerated, and this script refuses to run if it is absent
rather than producing a checkpoint dir that looks complete but has a
randomly-initialised connector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO / "checkpoints/stage1-connector-93M"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--llm", default="lmsys/vicuna-7b-v1.5")
    args = ap.parse_args()

    connector = args.ckpt / "connector/pytorch_model.bin"
    if not connector.exists():
        raise SystemExit(
            f"refusing to proceed: {connector} is missing.\n"
            f"The connector is the only Stage-1 artifact that carries training. "
            f"Copy it from the machine that has it — regenerating it here would "
            f"silently produce an untrained connector."
        )
    print(f"connector present: {connector.stat().st_size/1e6:.1f} MB")

    dest = args.ckpt / "language_model"
    if (dest / "pytorch_model.bin").exists():
        print(f"{dest} already materialized")
        return 0
    dest.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.llm} (frozen in Stage 1, so identical to what it saved)")
    model = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.float16)
    # Stage 1 saved via `model.language_model.named_parameters()` into a plain
    # torch .bin, and `config.text_config.save_pretrained` beside it. Match that
    # layout exactly — from_pretrained needs both.
    torch.save(model.state_dict(), dest / "pytorch_model.bin")
    AutoConfig.from_pretrained(args.llm).save_pretrained(dest)

    size = (dest / "pytorch_model.bin").stat().st_size / 1e9
    print(f"wrote {dest}/pytorch_model.bin ({size:.1f} GB) + config.json")

    (args.ckpt / "language_model_provenance.json").write_text(json.dumps({
        "reconstructed_from": args.llm,
        "why_exact": ("Stage 1 ran with --tune_type_llm frozen and load_llm calls "
                      "requires_grad_(False), so the LLM weights it saved are the "
                      "base checkpoint unchanged"),
        "not_reconstructed": "connector/pytorch_model.bin — the only trained artifact",
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
