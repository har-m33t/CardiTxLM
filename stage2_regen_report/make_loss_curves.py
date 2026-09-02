"""Phase 5a — loss curves for both stages.

Stage 1 is REPLOTTED from the existing 2026-08-13 logs. It was not retrained
(its data was never the problem), so this is a regeneration of the plot from
already-saved numbers, not a new run.

Stage 2 is plotted as a THREE-WAY comparison (Hypothesis B, Phase 4.4):

  1. original    — 86.4% of targets were one fixed string
  2. data-fix    — per-sample DE targets, four categories
  3. discrim     — data-fix plus the binary disease-vs-control category

Read the floors carefully, and read them against the right expectation:

  (1) -> (2): a HIGHER floor is evidence the fix worked. The original collapsed
  2.883 -> 0.166 in 155 steps because its targets were trivially predictable.
  Real per-sample content is a genuinely harder task; a corrected run that
  reproduced the old curve would have been the alarming outcome.

  (2) -> (3): the expectation INVERTS, and this is the trap in reading this
  plot. The added category's target is a one-bit label, which is much easier to
  fit than a gene list with effect sizes. So condition 3 should sit BELOW
  condition 2 purely from mixture arithmetic, and a lower floor here is NOT
  evidence of a better representation. It says nothing about Hypothesis B
  either way. Only the probe results in Phase 4.1-4.3 speak to that.

Because the three runs have different corpus sizes, absolute step is not
comparable across them. Both panels are drawn: absolute step (left) and
fraction of the epoch (right). Quote the right panel when comparing.

Inputs:
  runlogs/stage1_full.log                  step/loss lines from the Stage-1 run
  runlogs/stage2_trainer_state.json        log_history, ORIGINAL run
  runlogs/stage2_regen_trainer_state.json  log_history, data-fix run
  <--discrim-state>                        log_history, discriminative retrain
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
                    default=RUNLOGS / "stage2_regen_trainer_state.json",
                    help="trainer_state.json for the data-fix-only run")
    ap.add_argument("--discrim-state", type=Path,
                    default=RUNLOGS / "stage2_discrim_trainer_state.json",
                    help="trainer_state.json for the Hypothesis-B retrain")
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

    # --- Stage 2, three-way ----------------------------------------------
    # (label, path, colour). Missing conditions are skipped, not faked — the
    # plot must never imply a run that did not happen.
    conditions = [
        ("original (86.4% fixed-string targets)",
         RUNLOGS / "stage2_trainer_state.json", "#B0413E"),
        ("data fix (per-sample targets)",
         args.new_state, "#2E86AB"),
        ("data fix + discriminative (Hypothesis B)",
         args.discrim_state, "#3F8F5B"),
    ]

    series = []
    for label, path, colour in conditions:
        if not path.exists():
            print(f"skip: {path.name} not found")
            continue
        st, ls = from_trainer_state(path)
        if st:
            series.append((label, st, ls, colour))

    if series:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
        for ax, normalize in zip(axes, (False, True)):
            for label, st, ls, c in series:
                # Absolute step is NOT comparable across runs of different
                # corpus size; the right panel rescales each run to its own
                # epoch so the floors can be read against each other.
                x = [i / st[-1] for i in st] if normalize else st
                ax.plot(x, ls, lw=1.1, color=c, label=label)
                ax.annotate(f"{ls[-1]:.3f}", xy=(x[-1], ls[-1]),
                            xytext=(4, 0), textcoords="offset points",
                            fontsize=8, color=c, va="center")
            ax.set_ylabel("training loss")
            ax.grid(alpha=0.25)
            ax.set_xlabel("fraction of epoch" if normalize else "training step")
            ax.set_title("normalized — compare floors here" if normalize
                         else "absolute step (corpus sizes differ)")
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle("Stage 2 — three training conditions", y=1.0)
        fig.text(0.5, -0.06,
                 "Higher floor from (1) to (2) is the expected sign of the data fix. "
                 "A lower floor from (2) to (3) is\nmixture arithmetic — the added "
                 "binary target is easy to fit — and is NOT evidence about "
                 "Hypothesis B.",
                 ha="center", fontsize=7.5, color="0.35")
        fig.tight_layout()
        fig.savefig(OUT / "stage2_loss_three_way.png", dpi=160,
                    bbox_inches="tight")
        plt.close(fig)

        # Machine-readable floors, so the report cites numbers rather than a
        # reading of the picture.
        floors = {label: {"n_steps": len(st), "first": ls[0], "last": ls[-1],
                          "min": min(ls),
                          "final_10pct_mean": sum(ls[-max(1, len(ls)//10):])
                                              / max(1, len(ls)//10)}
                  for label, st, ls, _ in series}
        (OUT / "stage2_loss_floors.json").write_text(
            json.dumps(floors, indent=2) + "\n")
        for label, st, ls, _ in series:
            print(f"stage2 {label}: {len(st)} steps, {ls[0]:.3f} -> {ls[-1]:.3f}")
    else:
        print("skip stage 2: no trainer_state found for any run")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
