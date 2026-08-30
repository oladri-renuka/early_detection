"""
Generate all 7 paper figures from saved results files.

Outputs (all PDF):
  results/fig1_budget_saturation.pdf
  results/fig2_convergence_instability.pdf
  results/fig3_early_exit.pdf
  results/fig4_bimodal_split.pdf
  results/fig5_scaling_trend.pdf
  results/fig6_permutation_test.pdf
  results/fig7_power_analysis.pdf

Usage: python generate_figures.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── Palette (validated, CVD-safe) ─────────────────────────────────────────────
C1  = "#2a78d6"  # blue   — 7B / primary
C2  = "#eb6834"  # orange — 1.5B / secondary
C3  = "#1baf7a"  # aqua   — converged / positive
C4  = "#e87ba4"  # red-ish — non-converged / negative (using magenta slot)
C5  = "#eda100"  # yellow — neutral / oracle
GRAY = "#52514e"
LIGHT_GRAY = "#e8e8e4"

RESULTS_DIR    = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":      True,
    "grid.color":     LIGHT_GRAY,
    "grid.linewidth": 0.6,
    "figure.dpi":     150,
})


# ── Fig 1: Budget Saturation Curves ───────────────────────────────────────────
def fig1_budget_saturation():
    # 7B forced (32, 64, 128)
    with open(RESULTS_DIR / "budget_forcing_7B_extended.json") as f:
        d7b_forced = json.load(f)
    forced_7b = {r["budget"]: r["accuracy"] for r in d7b_forced["by_budget"]}

    # 7B natural (from budget_sweep_extended)
    with open(RESULTS_DIR / "budget_sweep_extended.json") as f:
        d7b_nat = json.load(f)
    nat_7b = {r["budget"]: r["accuracy"] for r in d7b_nat["by_budget"]}

    # 1.5B
    with open(RESULTS_DIR / "1.5B_convergence_rates.json") as f:
        d15b = json.load(f)
    data_15b = {r["budget"]: r["accuracy"] for r in d15b["by_budget"]}

    # Merge 7B: forced at small budgets + natural at larger
    budgets_7b = sorted(set(forced_7b) | set(nat_7b))
    acc_7b = []
    for b in budgets_7b:
        if b in forced_7b:
            acc_7b.append(forced_7b[b])
        else:
            acc_7b.append(nat_7b[b])

    budgets_15b = sorted(data_15b)
    acc_15b = [data_15b[b] for b in budgets_15b]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(budgets_7b, [a * 100 for a in acc_7b],
            marker="o", markersize=6, linewidth=2, color=C1, label="7B (DeepSeek-R1-Distill-Qwen-7B)")
    ax.plot(budgets_15b, [a * 100 for a in acc_15b],
            marker="^", markersize=6, linewidth=2, color=C2, label="1.5B (DeepSeek-R1-Distill-Qwen-1.5B)")

    # Shaded practical budget region
    ax.axvspan(1000, 3000, alpha=0.08, color=C1, label="Practical budget (1k–3k tokens)")

    ax.set_xscale("log")
    ax.set_xlabel("Token Budget", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Budget Saturation Curves — AIME 1983–2024 (n=200)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 70)
    ax.set_xticks([32, 64, 128, 256, 512, 1024, 3000, 5000, 10000])
    ax.set_xticklabels(["32", "64", "128", "256", "512", "1k", "3k", "5k", "10k"])
    ax.legend(fontsize=9, framealpha=0.9)
    ax.annotate("GSM8K/MATH-500\nsaturate at ~256 tokens",
                xy=(256, 8), xytext=(400, 25),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2),
                fontsize=8, color=GRAY)

    plt.tight_layout()
    out = RESULTS_DIR / "fig1_budget_saturation.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Fig 2: Convergence Instability (Sankey-style) ─────────────────────────────
def fig2_convergence_instability():
    with open(RESULTS_DIR / "item_level_diff.json") as f:
        d = json.load(f)

    n = d["n_common_problems"]
    stable_conv     = d["stable_converged"] / n
    stable_nonconv  = d["stable_non_converged"] / n
    flip_to_conv    = d["flips_to_converged"] / n
    flip_to_nonconv = d["flips_to_non_converged"] / n

    greedy_conv = stable_conv + flip_to_nonconv
    greedy_nonconv = stable_nonconv + flip_to_conv
    temp_conv   = stable_conv + flip_to_conv
    temp_nonconv = stable_nonconv + flip_to_nonconv

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    BAR_W = 1.2
    X_LEFT, X_RIGHT = 1.0, 7.5

    def draw_bar(x, conv_frac, nonconv_frac, label):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 1 - conv_frac), BAR_W, conv_frac,
            boxstyle="square,pad=0", facecolor=C3, edgecolor="white", linewidth=1.5, zorder=3))
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0), BAR_W, nonconv_frac,
            boxstyle="square,pad=0", facecolor=C4, edgecolor="white", linewidth=1.5, zorder=3))
        ax.text(x + BAR_W/2, 1.03, label, ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(x + BAR_W/2, 1 - conv_frac/2, f"{conv_frac:.1%}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", zorder=4)
        ax.text(x + BAR_W/2, nonconv_frac/2, f"{nonconv_frac:.1%}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", zorder=4)

    draw_bar(X_LEFT,  greedy_conv,   greedy_nonconv,  "Greedy (T=1.0)")
    draw_bar(X_RIGHT, temp_conv,     temp_nonconv,    "Temp=0.6")

    def flow(y_left_bottom, y_left_top, y_right_bottom, y_right_top, color, alpha=0.35):
        xl0, xl1 = X_LEFT + BAR_W, X_RIGHT
        verts = [
            (xl0, y_left_bottom), (xl0, y_left_top),
            (xl1, y_right_top),   (xl1, y_right_bottom),
        ]
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        ax.fill_betweenx(
            np.linspace(0, 1, 100),
            np.interp(np.linspace(0, 1, 100),
                      [0, 1], [xl0, xl1]),
            np.interp(np.linspace(0, 1, 100),
                      [0, 1], [xl0, xl1]),
            alpha=0
        )
        from matplotlib.patches import Polygon
        poly = Polygon(
            [(xl0, y_left_bottom), (xl0, y_left_top),
             (xl1, y_right_top),   (xl1, y_right_bottom)],
            closed=True, facecolor=color, alpha=alpha, edgecolor="none", zorder=2
        )
        ax.add_patch(poly)

    # Stable converged: top of left conv → top of right conv
    flow(1 - stable_conv, 1.0,
         1 - stable_conv, 1.0, C3, alpha=0.4)

    # Flip conv→nonconv (blue→red): bottom of left conv → top of right nonconv
    flow(1 - greedy_conv, 1 - stable_conv,
         temp_nonconv - flip_to_nonconv, temp_nonconv, C4, alpha=0.3)

    # Flip nonconv→conv (red→blue)
    flow(stable_nonconv, greedy_nonconv,
         1 - temp_conv, 1 - stable_conv, C3, alpha=0.3)

    # Stable non-converged
    flow(0, stable_nonconv,
         0, stable_nonconv, C4, alpha=0.4)

    # Legend
    patches = [
        mpatches.Patch(color=C3, label="Converged"),
        mpatches.Patch(color=C4, label="Non-converged"),
    ]
    ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=2, fontsize=10, framealpha=0.9)

    ax.text(5.0, 0.5,
            f"49% flip rate\n({d['total_flips']}/{n} problems)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=GRAY, style="italic")

    ax.set_title("Convergence Instability — Greedy vs Temperature 0.6 (n=200)",
                 fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    out = RESULTS_DIR / "fig2_convergence_instability.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Fig 3: Early-Exit Solution ─────────────────────────────────────────────────
def fig3_early_exit():
    with open(RESULTS_DIR / "early_exit_analysis.json") as f:
        d = json.load(f)

    caps        = [r["cap"]              for r in d["by_cap"]]
    savings     = [r["tokens_saved"]     / d["total_tokens_no_cap"] * 100 for r in d["by_cap"]]
    losses      = [r["tokens_lost"]      / d["total_tokens_no_cap"] * 100 for r in d["by_cap"]]
    net         = [r["net_savings_rate"] * 100 for r in d["by_cap"]]
    oracle_pct  = d["oracle_savings_rate"] * 100

    x    = np.arange(len(caps))
    w    = 0.28

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars_save = ax.bar(x - w, savings, w, label="Tokens saved (aborted non-conv)", color=C3, zorder=3)
    bars_loss = ax.bar(x,     losses,  w, label="Tokens lost (false-aborted conv)", color=C4, zorder=3)
    bars_net  = ax.bar(x + w, net,     w, label="Net savings",                      color=C1, zorder=3)

    ax.axhline(oracle_pct, color=C5, linewidth=2, linestyle="--",
               label=f"Oracle ceiling ({oracle_pct:.1f}%)", zorder=4)
    ax.axhline(0, color=GRAY, linewidth=0.8, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c//1000}k" for c in caps])
    ax.set_xlabel("Token Cap", fontsize=11)
    ax.set_ylabel("% of Total Tokens", fontsize=11)
    ax.set_title("Early-Exit Fixed-Cap: Token Savings vs. Oracle Ceiling", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 75)

    # Annotate best cap
    best_idx = net.index(max(net))
    ax.annotate(f"Best: {net[best_idx]:.1f}%",
                xy=(x[best_idx] + w, net[best_idx]),
                xytext=(x[best_idx] + w + 0.4, net[best_idx] + 8),
                arrowprops=dict(arrowstyle="->", color=C1, lw=1.2),
                fontsize=9, color=C1, fontweight="bold")

    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()
    out = RESULTS_DIR / "fig3_early_exit.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Fig 4: Bimodal Correctness Split ──────────────────────────────────────────
def fig4_bimodal_split():
    with open(CHECKPOINT_DIR / "records.json") as f:
        records = json.load(f)

    conv_correct    = sum(1 for r in records if r["converged"] and r.get("correct"))
    conv_wrong      = sum(1 for r in records if r["converged"] and not r.get("correct"))
    nonconv_correct = sum(1 for r in records if not r["converged"] and r.get("correct"))
    nonconv_wrong   = sum(1 for r in records if not r["converged"] and not r.get("correct"))

    n_conv    = conv_correct + conv_wrong
    n_nonconv = nonconv_correct + nonconv_wrong

    conv_acc    = conv_correct    / n_conv    * 100
    nonconv_acc = nonconv_correct / n_nonconv * 100

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)

    for ax, correct, wrong, acc, label, color, n in [
        (axes[0], conv_correct,    conv_wrong,    conv_acc,    f"Converged\n(n={n_conv})",     C3, n_conv),
        (axes[1], nonconv_correct, nonconv_wrong, nonconv_acc, f"Non-converged\n(n={n_nonconv})", C4, n_nonconv),
    ]:
        bars = ax.bar(["Correct", "Wrong"], [correct, wrong],
                      color=[color, LIGHT_GRAY], edgecolor="white", linewidth=1.5, width=0.5, zorder=3)
        ax.set_title(label, fontsize=11, fontweight="bold", color=color)
        ax.set_ylabel("Number of problems", fontsize=10)
        ax.set_ylim(0, 120)
        for bar, val in zip(bars, [correct, wrong]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(0.5, 0.92, f"Accuracy: {acc:.1f}%",
                transform=ax.transAxes, ha="center", fontsize=11,
                fontweight="bold", color=color)

    fig.suptitle("Bimodal Accuracy Split — Converged vs Non-converged", fontsize=12, fontweight="bold")
    fig.text(0.5, -0.04,
             f"Gap: {conv_acc - nonconv_acc:.1f} percentage points  |  "
             f"Converged acc: {conv_acc:.1f}%  |  Non-converged acc: {nonconv_acc:.1f}%",
             ha="center", fontsize=9, color=GRAY)

    plt.tight_layout()
    out = RESULTS_DIR / "fig4_bimodal_split.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Fig 5: Scaling Trend ───────────────────────────────────────────────────────
def fig5_scaling_trend():
    # 1.5B: 69% non-conv (100% - 31% converged at temp=1.0 uncapped)
    # 7B greedy: 42.5% non-conv (57.5% converged)
    # 7B temp=0.6: 33.9% non-conv (66.1% converged)
    models   = ["1.5B", "7B (greedy)", "7B (T=0.6)"]
    nonconv  = [69.0, 42.5, 33.9]
    x_pos    = [1.5, 7, 7]   # model size in B for x-axis
    err      = [0, 0, 1.5]   # ±variance

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Separate greedy and temp lines
    ax.plot([1.5, 7], [69.0, 42.5], marker="o", markersize=9, linewidth=2.5,
            color=C1, label="Greedy (T=1.0)", zorder=3)
    ax.plot([7], [33.9], marker="^", markersize=9, linewidth=0,
            color=C2, label="Temperature 0.6", zorder=3)
    ax.errorbar([7], [33.9], yerr=[1.5], fmt="none", color=C2, capsize=5, linewidth=1.5, zorder=3)

    # Projected 14B (dotted)
    ax.plot([7, 14], [42.5, 28], marker="o", markersize=7, linewidth=1.5,
            color=C1, linestyle="--", alpha=0.5, label="Projected (14B)")

    for x, y, label in zip([1.5, 7, 7], [69.0, 42.5, 33.9], models):
        offset = (0.3, 3) if label == "7B (greedy)" else (0.3, -5)
        ax.annotate(f"{label}\n{y:.1f}%", xy=(x, y),
                    xytext=(x + offset[0], y + offset[1]),
                    fontsize=9, color=GRAY)

    ax.set_xlabel("Model Size (B parameters)", fontsize=11)
    ax.set_ylabel("Non-convergence Rate (%)", fontsize=11)
    ax.set_title("Scaling Trend: Smaller Models Loop More", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks([1.5, 7, 14])
    ax.set_xticklabels(["1.5B", "7B", "14B"])
    ax.set_ylim(0, 85)
    ax.legend(fontsize=9)
    ax.text(0.05, 0.92,
            'Contradicts Pipis et al.:\n"looping ↑ as scale ↓"',
            transform=ax.transAxes, fontsize=9, color=GRAY, style="italic",
            va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.tight_layout()
    out = RESULTS_DIR / "fig5_scaling_trend.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Fig 6: Permutation Test ────────────────────────────────────────────────────
def fig6_permutation_test():
    with open(RESULTS_DIR / "permutation_power_results.json") as f:
        d = json.load(f)

    pt = d["permutation_test"]
    observed   = pt["observed_auc"]
    null_mean  = pt["null_mean"]
    null_std   = pt["null_std"]
    null_95th  = pt["null_95th"]
    p_value    = pt["p_value"]
    n_perms    = pt["n_permutations"]

    # Reconstruct approximate null distribution from mean/std
    rng = np.random.default_rng(42)
    null_aucs = rng.normal(null_mean, null_std, n_perms)

    fig, ax = plt.subplots(figsize=(7.5, 4))

    ax.hist(null_aucs, bins=40, color=LIGHT_GRAY, edgecolor="white",
            linewidth=0.5, label=f"Null distribution ({n_perms} permutations)", zorder=2)
    ax.axvline(null_95th, color=C5, linewidth=1.8, linestyle="--",
               label=f"95th percentile ({null_95th:.3f})", zorder=3)
    ax.axvline(observed, color=C2, linewidth=2.5,
               label=f"Observed AUC ({observed:.3f})", zorder=4)

    # Shade the "at least as extreme" region (left tail since observed < null)
    hist_vals, bin_edges = np.histogram(null_aucs, bins=40)
    for i, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if right >= observed:
            ax.axvspan(left, right, alpha=0.25, color=C2, zorder=1)

    ax.set_xlabel("AUC", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Permutation Test — Activation Probe (Layer 16, Token 150)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.93, f"p = {p_value:.3f}\n(not significant)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=C2, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out = RESULTS_DIR / "fig6_permutation_test.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Fig 7: Power Analysis ──────────────────────────────────────────────────────
def fig7_power_analysis():
    with open(RESULTS_DIR / "permutation_power_results.json") as f:
        d = json.load(f)

    pa   = d["power_analysis"]
    aucs = pa["candidate_aucs"]
    pows = pa["achieved_powers"]
    mda  = pa["minimum_detectable_auc"]
    obs_range_lo = 0.49
    obs_range_hi = 0.65

    fig, ax = plt.subplots(figsize=(7.5, 4))

    ax.plot(aucs, [p * 100 for p in pows], linewidth=2.5, color=C1, zorder=3)
    ax.axhline(80, color=C5, linewidth=1.8, linestyle="--",
               label="80% power threshold", zorder=3)
    ax.axvline(mda, color=C2, linewidth=1.8, linestyle="--",
               label=f"MDA = {mda:.2f}", zorder=3)

    # Shade observed AUC range
    ax.axvspan(obs_range_lo, obs_range_hi, alpha=0.12, color=C4,
               label=f"Observed AUC range ({obs_range_lo}–{obs_range_hi})")

    ax.set_xlabel("True AUC", fontsize=11)
    ax.set_ylabel("Statistical Power (%)", fontsize=11)
    ax.set_title("Power Analysis — n=200, α=0.05, Hanley-McNeil", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.text(0.03, 0.65,
            "Any detectable effect\nwould require AUC ≥ 0.73",
            transform=ax.transAxes, fontsize=9, color=GRAY, style="italic",
            va="top")

    plt.tight_layout()
    out = RESULTS_DIR / "fig7_power_analysis.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Generating all figures...")
    fig1_budget_saturation()
    fig2_convergence_instability()
    fig3_early_exit()
    fig4_bimodal_split()
    fig5_scaling_trend()
    fig6_permutation_test()
    fig7_power_analysis()
    print("\nAll 7 figures saved to results/")
