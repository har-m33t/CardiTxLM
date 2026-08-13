"""make_synthetic_stage1_data.py — tiny synthetic Stage-1 dataset.

Emits a handful of `[20010]` float32 `.npy` expression vectors plus a
`pretrain.json` in TinyLLaVA's confirmed schema (see
`integration/repo_findings.md` §1), so `scripts/train/train_stage1.sh` can be
launched end-to-end without touching the real 199,954-pair corpus.

This is a *plumbing* fixture only — the vectors are random noise, not real
ARCHS4-derived expression. Use `integration/build_dataset_json.py` for real data.

    python -m integration.make_synthetic_stage1_data --outdir /tmp/stage1_dry --n 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_GENES = 20010

# A small closed set of answers, so a 5-step run has something learnable and the
# loss curve is readable rather than pure noise.
_ANSWERS = [
    "This transcriptome shows a cardiovascular disease signature.",
    "This transcriptome shows a healthy control signature.",
]


def build(outdir: Path, n: int, seed: int = 0) -> Path:
    img_dir = outdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    records = []
    for i in range(n):
        cls = i % len(_ANSWERS)
        # class-separated means so the connector has real signal to fit
        vec = (rng.normal(loc=float(cls), scale=1.0, size=N_GENES)).astype(np.float32)
        np.save(img_dir / f"syn{i}.npy", vec)
        records.append({
            "id": f"syn{i}",
            "image": f"syn{i}.npy",
            "conversations": [
                {"from": "human", "value": "<image>"},
                {"from": "gpt", "value": _ANSWERS[cls]},
            ],
        })

    json_path = outdir / "pretrain.json"
    json_path.write_text(json.dumps(records, indent=2))
    print(f"[synthetic] wrote {n} samples -> {json_path} (images in {img_dir})")
    return json_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.outdir, args.n, args.seed)


if __name__ == "__main__":
    main()
