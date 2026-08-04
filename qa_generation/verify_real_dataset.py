"""Task 4 — run REAL training entries through TinyLLaVA's actual data pipeline.

Supersedes the original Step 0 fake-sample check. That used hand-written records
and a synthetic `.npy`; this uses real generated QA pairs and the real
materialized expression vectors, through the same `LazySupervisedDataset` and
`DataCollatorForSupervisedDataset` the trainer uses.

Checks, in order:
  1. the loader parses the real JSON and returns tensors
  2. the collated batch has the expected shapes, including [B, 20010] images
  3. the `<image>` token is stripped from the text and replaced by the
     IMAGE_TOKEN_INDEX sentinel in `input_ids` (template logic, not raw text)
  4. multi-turn entries — the same `image` appearing in several rows — load as
     independent examples with different token content
  5. the loaded vector matches the file on disk, and the file matches the
     source matrix row

Reads only. Slices are written to a scratch dir; the training JSONs are never
modified.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data/cvd_transcriptome"
IMAGE_FOLDER = DATA / "embeddings"
TEXT_FILES = DATA / "text_files"
N_GENES = 20010
TINY_LLM = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def report(tag: str, **kv) -> None:
    print(f"[{tag}] " + " ".join(f"{k}={v}" for k, v in kv.items()))


def pick(path: Path, n: int) -> list[dict]:
    """n real entries, deliberately including repeats of one image."""
    data = json.loads(path.read_text())
    by_image: dict[str, list[dict]] = {}
    for entry in data:
        by_image.setdefault(entry["image"], []).append(entry)
    # an image with several QA pairs, so multi-turn behaviour is exercised
    repeated = max(by_image, key=lambda k: len(by_image[k]))
    picked = by_image[repeated][: min(3, len(by_image[repeated]))]
    for entry in data:
        if len(picked) >= n:
            break
        if entry["image"] != repeated:
            picked.append(entry)
    return picked[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from tinyllava.data.dataset import (
        DataCollatorForSupervisedDataset,
        LazySupervisedDataset,
    )
    from tinyllava.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

    tokenizer = AutoTokenizer.from_pretrained(TINY_LLM, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    failures: list[str] = []
    summary: dict = {}

    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        for stage, name in ((1, "stage1_train.json"), (2, "stage2_train.json")):
            src = TEXT_FILES / name
            picked = pick(src, args.n)
            slice_path = scratch / f"slice_stage{stage}.json"
            slice_path.write_text(json.dumps(picked))

            data_args = SimpleNamespace(
                conv_version="pretrain",
                image_folder=str(IMAGE_FOLDER),
                image_processor=None,  # unused on the .npy branch
                image_aspect_ratio="square",
                is_multimodal=True,
                image_grid_pinpoints=None,
                data_path=str(slice_path),
            )
            ds = LazySupervisedDataset(str(slice_path), tokenizer, data_args)
            items = [ds[i] for i in range(len(ds))]
            collate = DataCollatorForSupervisedDataset(tokenizer)
            batch = collate(items)

            imgs = batch["images"]
            ids = batch["input_ids"]
            labels = batch["labels"]
            report(
                f"stage{stage}",
                n=len(ds),
                images=tuple(imgs.shape),
                input_ids=tuple(ids.shape),
                labels=tuple(labels.shape),
                dtype=str(imgs.dtype),
            )

            # (2) batch shape
            if tuple(imgs.shape) != (len(picked), N_GENES):
                failures.append(f"stage{stage}: images shape {tuple(imgs.shape)} != ({len(picked)}, {N_GENES})")
            if imgs.dtype != torch.float32:
                failures.append(f"stage{stage}: images dtype {imgs.dtype} != float32")
            if ids.shape[0] != len(picked):
                failures.append(f"stage{stage}: input_ids batch {ids.shape[0]} != {len(picked)}")
            if not torch.isfinite(imgs).all():
                failures.append(f"stage{stage}: non-finite values in collated images")

            # (3) <image> token handling
            n_img_tok = int((ids == IMAGE_TOKEN_INDEX).sum())
            literal = tokenizer.batch_decode(ids.clamp(min=0))
            leaked = sum(1 for t in literal if "<image>" in t)
            report(
                f"stage{stage}-imagetoken",
                sentinel_count=n_img_tok,
                expected=len(picked),
                literal_leak=leaked,
            )
            if n_img_tok != len(picked):
                failures.append(
                    f"stage{stage}: found {n_img_tok} IMAGE_TOKEN_INDEX sentinels, expected {len(picked)}"
                )
            if leaked:
                failures.append(f"stage{stage}: literal '<image>' survived into {leaked} decoded sequences")
            if not (labels != IGNORE_INDEX).any():
                failures.append(f"stage{stage}: every label is IGNORE_INDEX — nothing to train on")

            # (4) multi-turn: same image, independent examples
            same = [i for i, e in enumerate(picked) if e["image"] == picked[0]["image"]]
            distinct_text = len({tuple(ids[i].tolist()) for i in same})
            identical_vectors = all(torch.equal(imgs[same[0]], imgs[i]) for i in same)
            report(
                f"stage{stage}-multiturn",
                image=picked[0]["image"],
                rows_sharing_image=len(same),
                distinct_token_sequences=distinct_text,
                same_vector_reused=identical_vectors,
            )
            if len(same) < 2:
                failures.append(f"stage{stage}: no repeated image in the slice — multi-turn not exercised")
            elif distinct_text != len(same):
                failures.append(
                    f"stage{stage}: {len(same)} rows share an image but only {distinct_text} distinct "
                    f"token sequences — entries are not independent"
                )
            if not identical_vectors:
                failures.append(f"stage{stage}: rows sharing an image got different vectors")

            # (5) tensor matches disk, disk matches source matrix
            from qa_generation import gt_functions as gt

            rows = gt._expression_rows()
            matrix = np.load(gt.EXPRESSION_NPY, mmap_mode="r")
            drift = 0
            for i, entry in enumerate(picked):
                on_disk = np.load(IMAGE_FOLDER / entry["image"])
                source = np.asarray(matrix[rows[entry["image"][:-4]]], dtype=np.float32)
                if not np.array_equal(on_disk, source):
                    drift += 1
                if not np.array_equal(imgs[i].numpy(), on_disk):
                    drift += 1
            report(f"stage{stage}-provenance", checked=len(picked), mismatches=drift)
            if drift:
                failures.append(f"stage{stage}: {drift} vector mismatches loader vs disk vs source matrix")

            summary[f"stage{stage}"] = {
                "entries": len(picked),
                "images_shape": list(imgs.shape),
                "input_ids_shape": list(ids.shape),
                "image_sentinels": n_img_tok,
                "rows_sharing_one_image": len(same),
                "distinct_token_sequences": distinct_text,
                "provenance_mismatches": drift,
            }

    print()
    if failures:
        print("TASK 4 RESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("TASK 4 RESULT: PASS — real entries load, collate, and tokenize correctly")
    (REPO / "qa_generation/real_loader_verification.json").write_text(
        json.dumps({"result": "pass", "detail": summary}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
