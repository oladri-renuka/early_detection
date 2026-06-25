"""Regenerate plots from saved results JSON (no GPU needed)."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")

with open(RESULTS_DIR / "early_detection_results.json") as f:
    results = json.load(f)

# ── Per-checkpoint bar charts ─────────────────────────────────────────────────
for r in results:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    names = ["Behavioral\nBaseline", "Activation\nProbe", "Combined"]
    means = [r["behavioral_auc_mean"], r["activation_auc_mean"], r["combined_auc_mean"]]
    stds = [r["behavioral_auc_std"], r["activation_auc_std"], r["combined_auc_std"]]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    bars = ax.bar(names, means, yerr=stds, color=colors, capsize=6, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("AUC (5-fold CV)")
    ax.set_title(f"Probe Comparison at {r['checkpoint_pos']} Tokens")
    ax.set_ylim(0.3, 0.85)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"probe_comparison_cp{r['checkpoint_pos']}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved probe_comparison_cp{r['checkpoint_pos']}.png")

# ── Sweep plot ────────────────────────────────────────────────────────────────
positions = [r["checkpoint_pos"] for r in results]
act_aucs = [r["activation_auc_mean"] for r in results]
behav_aucs = [r["behavioral_auc_mean"] for r in results]
comb_aucs = [r["combined_auc_mean"] for r in results]
act_stds = [r["activation_auc_std"] for r in results]
behav_stds = [r["behavioral_auc_std"] for r in results]
comb_stds = [r["combined_auc_std"] for r in results]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.errorbar(positions, act_aucs, yerr=act_stds, marker="o", label="Activation Probe", capsize=4, linewidth=2)
ax.errorbar(positions, behav_aucs, yerr=behav_stds, marker="s", label="Behavioral Baseline", capsize=4, linewidth=2)
ax.errorbar(positions, comb_aucs, yerr=comb_stds, marker="^", label="Combined", capsize=4, linewidth=2)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random chance")
ax.set_xlabel("Checkpoint Position (tokens into thinking chain)")
ax.set_ylabel("AUC (5-fold CV)")
ax.set_title("Early Detection of Reasoning Non-Convergence\nAUC vs Checkpoint Position")
ax.legend()
ax.set_ylim(0.3, 0.85)
ax.set_xticks(positions)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "auc_vs_checkpoint.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved auc_vs_checkpoint.png")
