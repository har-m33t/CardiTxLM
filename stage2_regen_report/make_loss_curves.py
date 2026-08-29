"""Phase 5a — loss curves for both stages.

Stage 1 is REPLOTTED from the existing 2026-08-13 logs. It was not retrained
(its data was never the problem), so this is a regeneration of the plot from
already-saved numbers, not a new run.

Stage 2 is plotted new-vs-old on one axis. That comparison is the point, and it
should be read carefully: a HIGHER loss floor on the corrected data is evidence
the fix worked, not evidence of a worse model. The original run collapsed
2.883 -> 0.166 in 155 steps because 86.4% of its targets were a fixed string,
which is trivially predictable. Once the targets carry real per-sample content
the task is genuinely harder, and the loss should not fall as far or as fast.
A corrected run that reproduced the old curve would be the alarming outcome.

Inputs:
  runlogs/stage1_full.log            step/loss lines from the Stage-1 run
  runlogs/stage2_trainer_state.json  log_history for the ORIGINAL Stage-2 run
  <new>/trainer_state.json           log_history for the corrected retrain
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUNLOGS = REPO / "runlogs"
OUT = REPO / "stage2_regen_report/loss_curves"

LOSS_LINE = re.compile(r"\{'loss': ([\d.]+),.*?'epoch': ([\d.]+)\}")


def from_trainer_state(path: Path) -> tuple[list[int], list[float]]:
    hist = json.loads(path.read_text()).get("log_history", [])
    steps, loss = [], []
    for h in hist:
        if "loss" in h and "step" in h:
            steps.append(int(h["step"]))
            loss.append(float(h["loss"]))
    return steps, loss


def from_log(path: Path) -> tuple[list[int], list[float]]:
    steps, loss = [], []
    for i, m in enumerate(LOSS_LINE.finditer(path.read_text(errors="ignore")), 1):
        steps.append(i)
        loss.append(float(m.group(1)))
    return steps, loss


def annotate_convergence(ax, steps, loss, frac=0.35):
    """Mark where the curve has effectively flattened."""
    if not steps:
        return
    k = max(1, int(len(steps) * frac))
    ax.axvline(steps[k - 1], color="0.55", ls="--", lw=1)
    ax.annotate(f"~{int(frac*100)}% of epoch\nloss {loss[k-1]:.3f}",
                xy=(steps[k - 1], loss[k - 1]),
                xytext=(0.42, 0.72), textcoords="axes fraction",
                fontsize=8, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.55", lw=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-state", type=Path,
                    default=RUNLOGS / "stage2_regen_trainer_state.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # --- Stage 1 (replot only) -------------------------------------------
    s1 = RUNLOGS / "stage1_full.log"
    if s1.exists():
        steps, loss = from_log(s1)
        if steps:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.plot(steps, loss, lw=0.8, color="#2E86AB")
            annotate_convergence(ax, steps, loss, 0.35)
            ax.set_xlabel("logged step")
            ax.set_ylabel("training loss")
            ax.set_title("Stage 1 — connector alignment (not retrained; replotted)")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(OUT / "stage1_loss.png", dpi=160)
            plt.close(fig)
            print(f"stage1_loss.png — {len(steps)} points, "
                  f"{loss[0]:.3f} -> {loss[-1]:.3f}")
    else:
        print(f"skip stage 1: {s1} not found")

    # --- Stage 2 before/after --------------------------------------------
    old_p = RUNLOGS / "stage2_trainer_state.json"
    series = []
    if old_p.exists():
        st, ls = from_trainer_state(old_p)
        if st:
            series.append(("original (86.4% fixed-string targets)", st, ls, "#B0413E"))
    if args.new_state.exists():
        st, ls = from_trainer_state(args.new_state)
        if st:
            series.append(("corrected (per-sample targets)", st, ls, "#2E86AB"))

    if series:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for label, st, ls, c in series:
            ax.plot(st, ls, lw=1.1, color=c, label=label)
            ax.annotate(f"{ls[-1]:.3f}", xy=(st[-1], ls[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=c, va="center")
        ax.set_xlabel("training step")
        ax.set_ylabel("training loss")
        ax.set_title("Stage 2 — before vs after the data fix")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25)
        ax.text(0.5, -0.22,
                "A higher floor on the corrected run is the expected result: the "
                "task is genuinely harder\nthan predicting a fixed string.",
                transform=ax.transAxes, ha="center", fontsize=7.5, color="0.35")
        fig.tight_layout()
        fig.savefig(OUT / "stage2_loss_before_after.png", dpi=160,
                    bbox_inches="tight")
        plt.close(fig)
        for label, st, ls, _ in series:
            print(f"stage2 {label}: {len(st)} steps, {ls[0]:.3f} -> {ls[-1]:.3f}")
    else:
        print("skip stage 2: no trainer_state found for either run")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
