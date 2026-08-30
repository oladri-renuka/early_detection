"""
Extended budget sweep analysis.

Loads the original 200-problem run (budgets 256, 512, 1024, 10k) plus
extracts 32/64/128 if those activations were saved.
Reports accuracy, convergence rate, mean tokens per budget.
Identifies B* (saturation point: 95% of uncapped accuracy).

Usage: python budget_sweep_extended.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR    = Path("results")
CHECKPOINT_DIR = Path("checkpoints")
SATURATION     = 0.95  # B* defined as 95% of uncapped accuracy


def load_records():
    with open(CHECKPOINT_DIR / "records.json") as f:
        return json.load(f)


def accuracy_at_budget(records, budget):
    """
    At a given budget: converged → use answer; non-converged → wrong.
    Returns (accuracy, convergence_rate, mean_tokens).
    """
    correct   = 0
    converged = 0
    total_tok = 0

    for r in records:
        n_tok = r["n_tokens"]
        conv  = r["converged"]
        # Under this budget the model would have been cut off → non-converged
        if n_tok > budget:
            conv = False
        if conv and r.get("correct", False):
            correct += 1
        if conv:
            converged += 1
        total_tok += min(n_tok, budget)

    n = len(records)
    return correct / n, converged / n, total_tok / n


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    records = load_records()
    n = len(records)
    print(f"Loaded {n} records")

    # Uncapped (10k) baseline
    uncapped_acc  = sum(r.get("correct", False) for r in records) / n
    uncapped_conv = sum(r["converged"] for r in records) / n
    uncapped_tok  = sum(r["n_tokens"] for r in records) / n
    print(f"Uncapped: acc={uncapped_acc:.3f} conv={uncapped_conv:.3f} mean_tok={uncapped_tok:.0f}")

    # Budget grid — extend downward if possible
    budgets = [32, 64, 128, 256, 512, 1024, 10_000]

    rows = []
    for b in budgets:
        acc, conv, mean_tok = accuracy_at_budget(records, b)
        rows.append({
            "budget": b,
            "accuracy": acc,
            "convergence_rate": conv,
            "mean_tokens_generated": mean_tok,
        })
        print(f"  budget={b:6d}: acc={acc:.3f} conv={conv:.3f} mean_tok={mean_tok:.0f}")

    # B*: smallest budget where acc >= 95% of uncapped
    threshold = SATURATION * uncapped_acc
    b_star = None
    for row in rows:
        if row["accuracy"] >= threshold:
            b_star = row["budget"]
            break
    print(f"\nB* (saturation at {SATURATION:.0%} of uncapped acc={uncapped_acc:.3f}): {b_star}")

    # Plot
    plot_budgets = [r["budget"] for r in rows]
    plot_accs    = [r["accuracy"] for r in rows]
    plot_convs   = [r["convergence_rate"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.plot(plot_budgets, plot_accs,  marker="o", color="steelblue", label="Accuracy", linewidth=2)
    ax2.plot(plot_budgets, plot_convs, marker="s", color="coral",     label="Conv rate", linewidth=2, linestyle="--")

    ax1.axhline(threshold, color="steelblue", linestyle=":", alpha=0.6,
                label=f"95% of uncapped ({threshold:.3f})")
    if b_star:
        ax1.axvline(b_star, color="red", linestyle="--", alpha=0.7, label=f"B*={b_star}")

    ax1.set_xlabel("Token Budget")
    ax1.set_ylabel("Accuracy", color="steelblue")
    ax2.set_ylabel("Convergence Rate", color="coral")
    ax1.set_xscale("log")
    ax1.set_title("Budget Sweep — Accuracy & Convergence vs Token Budget")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "budget_sweep_extended.pdf", bbox_inches="tight")
    plt.close()

    out = {
        "n_problems": n,
        "uncapped_accuracy": uncapped_acc,
        "uncapped_convergence_rate": uncapped_conv,
        "saturation_threshold": SATURATION,
        "b_star": b_star,
        "by_budget": rows,
    }
    with open(RESULTS_DIR / "budget_sweep_extended.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {RESULTS_DIR}/budget_sweep_extended.json")


if __name__ == "__main__":
    main()
