"""Collect every deliverable of the Stage-2 regeneration into `results/`.

One folder, so the run can be handed to someone without them needing to know
where each artifact happened to be produced. Re-runnable: it copies from the
canonical locations rather than being the place things are written to, so
`results/` can be deleted and rebuilt at any time and can never drift from the
sources.

Layout:
    results/
      SUMMARY.md               the write-up: what was done, what was found
      STAGE2_DATA_ANALYSIS.md  breakdown of the regenerated corpus
      README.md           generated from the artifacts, numbers only
      loss_curves/        stage 1 (replot) and stage 2 (before vs after)
      plots/              three-way probe, multi-label before/after
      tables/             every CSV/JSON the report cites
      data/               the manifests that make the corpus auditable
      logs/               trainer state and run logs
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results"

#: (source, destination-subdir). Missing sources are reported, not fatal — the
#: folder should still assemble if one optional artifact was not produced.
ITEMS: list[tuple[str, str]] = [
    # --- figures ---
    ("stage2_regen_report/loss_curves/stage1_loss.png", "loss_curves"),
    ("stage2_regen_report/loss_curves/stage2_loss_before_after.png", "loss_curves"),
    ("stage2_regen_report/plots/probe_three_way.png", "plots"),
    ("stage2_regen_report/plots/multilabel_before_after.png", "plots"),
    # --- tables ---
    ("stage2_regen_report/tables/probe_comparison.csv", "tables"),
    ("stage2_regen_report/tables/probe_three_way.json", "tables"),
    ("stage2_regen_report/tables/multilabel_before_after.csv", "tables"),
    ("stage2_regen_report/tables/multilabel_probe_before.csv", "tables"),
    ("stage2_regen_report/tables/multilabel_probe_after.csv", "tables"),
    ("stage2_regen_report/tables/encoder_scale_sweep.csv", "tables"),
    ("stage2_regen_report/tables/encoder_scale_sweep_note.md", "tables"),
    # --- provenance: how the corpus was built and what it contains ---
    ("qa_generation/stage2_bundle_stats.json", "data"),
    ("qa_generation/de/de_manifest.json", "data"),
    ("qa_generation/stage2_regen_plan_stats.json", "data"),
    ("qa_generation/step3_regen_stats.json", "data"),
    ("qa_generation/regen_usage_summary.json", "data"),
    ("qa_generation/stage2_regen_rejected.json", "data"),
    ("data/cvd_transcriptome/holdout_series.json", "data"),
    ("data/cvd_transcriptome/encoded_cache_manifest.json", "data"),
    ("linear_probe/multilabel_labels_manifest.json", "data"),
    # --- logs ---
    ("runlogs/stage2_regen_trainer_state.json", "logs"),
    ("runlogs/stage2_trainer_state.json", "logs"),
    ("runlogs/pod_bootstrap.log", "logs"),
    ("runlogs/stage2_regen_train.log", "logs"),
    ("runlogs/probe_pca.log", "logs"),
    ("runlogs/ml_before.log", "logs"),
    ("runlogs/ml_after.log", "logs"),
    # --- the corpus itself, so the folder is self-contained ---
    ("data/cvd_transcriptome/text_files/stage2_train.zip", "data"),
    # --- corpus breakdown (scripts/analyze_stage2_corpus.py) ---
    ("results/tables/corpus_composition.csv", "tables"),
    ("results/tables/corpus_gene_frequency.csv", "tables"),
    ("results/tables/corpus_de_statistics.csv", "tables"),
    ("results/plots/corpus_composition.png", "plots"),
    # --- the generated, numbers-only report ---
    ("stage2_regen_report/README.md", "."),
]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    copied, missing = 0, []
    for src, sub in ITEMS:
        s = REPO / src
        if not s.exists():
            missing.append(src)
            continue
        d = OUT / sub
        d.mkdir(parents=True, exist_ok=True)
        dest = d / s.name
        # analyze_stage2_corpus.py writes its outputs straight into results/,
        # so those entries are already in place. Listed here anyway so the
        # manifest and the layout docstring stay complete.
        if s.resolve() != dest.resolve():
            shutil.copy2(s, dest)
        copied += 1

    print(f"copied {copied} artifacts into {OUT.relative_to(REPO)}/")
    for sub in sorted({s for _, s in ITEMS}):
        p = OUT / sub
        if p.exists():
            names = sorted(f.name for f in p.iterdir() if f.is_file())
            print(f"  {sub + '/':16s} {len(names)} files")
    if missing:
        print("\nnot found (skipped):")
        for m in missing:
            print(f"  {m}")

    # A machine-readable index so a consumer does not have to guess what each
    # file is; the SUMMARY covers the same ground in prose.
    (OUT / "MANIFEST.json").write_text(json.dumps({
        "purpose": "Stage-2 regeneration and retrain — all deliverables",
        "rebuild_with": "python3 scripts/build_results_folder.py",
        "note": ("copied from canonical locations; delete and rebuild freely, "
                 "this folder is never the source of truth"),
        "contents": {
            sub: sorted(f.name for f in (OUT / sub).iterdir() if f.is_file())
            for _, sub in {(None, s) for _, s in ITEMS}
            if (OUT / sub).exists()
        },
        "missing": missing,
    }, indent=2) + "\n")
    print(f"\nwrote {(OUT / 'MANIFEST.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
