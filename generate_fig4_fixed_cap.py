"""
Generate Figure 4: Fixed-Cap Budget Forcing — Compute vs. Accuracy Pareto Frontier.

Reads results/budget_forcing_fixed_cap.json produced by budget_forcing_fixed_cap.py.
Replaces the old fig3_early_exit (broken "net savings" metric) with a correct
two-axis plot: compute saved (x) vs. accuracy retained (y).

Output: results/fig4_fixed_cap.pdf

Usage: python generate_fig4_fixed_cap.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")

C1    = "#2a78d6"   # blue  — frontier line
C2    = "#eb6834"   # orange — forced-only accuracy
C3    = "#1baf7a"   # green — oracle
GRAY  = "#52514e"
LIGHT_GRAY = "#e8e8e4"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        LIGHT_GRAY,
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
})

UNCAPPED_ACCURACY = 0.565   # from budget_sweep_extended.json


def main():
    with open(RESULTS_DIR / "budget_forcing_fixed_cap.json") as f:
        d = json.load(f)

    rows            = d["by_cap"]
    caps            = [r["cap"]               for r in rows]
    compute_saved   = [r["compute_saved_rate"] * 100 for r in rows]
    accuracy        = [r["accuracy"]           * 100 for r in rows]
    acc_forced      = [r["accuracy_forced"]    * 100 for r in rows]
    oracle_saved    = 42.5   # non-convergence rate = oracle ceiling

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # ── Pareto frontier: overall accuracy vs compute saved ──
    ax.plot(compute_saved, accuracy, marker="o", linewidth=2,
            color=C1, zorder=4, label="Accuracy (budget-forced)")

    # ── Forced-only accuracy (dashed) ──
    ax.plot(compute_saved, acc_forced, marker="s", linewidth=1.5,
            linestyle="--", color=C2, zorder=3, label="Accuracy (forced gens only)")

    # ── Uncapped accuracy reference ──
    ax.axhline(UNCAPPED_ACCURACY * 100, color=GRAY, linewidth=1.2,
               linestyle=":", zorder=2,
               label=f"Uncapped accuracy ({UNCAPPED_ACCURACY*100:.1f}%)")

    # ── Oracle compute ceiling ──
    ax.axvline(oracle_saved, color=C3, linewidth=1.2,
               linestyle="--", zorder=2,
               label=f"Oracle ceiling ({oracle_saved:.1f}% saved)")

    # ── Annotate each cap ──
    for cap, cs, acc in zip(caps, compute_saved, accuracy):
        label = f"C={cap//1000}k"
        offset = (4, 5) if cap != 4000 else (4, -12)
        ax.annotate(
            label,
            xy=(cs, acc),
            xytext=(cs + offset[0], acc + offset[1]),
            fontsize=8, color=C1,
            arrowprops=dict(arrowstyle="-", color=C1, lw=0.7),
        )

    ax.set_xlabel("Compute Saved vs. Uncapped (%)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(
        "Fixed-Cap Budget Forcing: Compute–Accuracy Frontier\n"
        "(DeepSeek-R1-Distill-Qwen-7B, AIME 1983–2024, $n$=200)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 75)
    ax.legend(fontsize=9, framealpha=0.9, loc="lower left")

    plt.tight_layout()
    out = RESULTS_DIR / "fig4_fixed_cap.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # ── Print table for paper ──
    print(f"\n{'Cap':>6}  {'Saved%':>7}  {'Acc%':>6}  {'AccForced%':>11}  {'NatConv':>8}  {'Forced':>7}")
    for r in rows:
        print(
            f"  {r['cap']:4d}  "
            f"{r['compute_saved_rate']*100:6.1f}%  "
            f"{r['accuracy']*100:5.1f}%  "
            f"{r['accuracy_forced']*100:10.1f}%  "
            f"{r['n_naturally_converged']:8d}  "
            f"{r['n_forced']:7d}"
        )
    print(f"\nUncapped: acc={UNCAPPED_ACCURACY*100:.1f}%  Oracle ceiling: {oracle_saved:.1f}%")


if __name__ == "__main__":
    main()
