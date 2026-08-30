"""
Early-exit fixed-cap analysis.

For each token cap in [4000, 5000, 5500, 6000, 7000]:
  - Non-converged generations aborted (tokens saved)
  - Converged generations falsely aborted (tokens lost)
  - Net token savings vs oracle ceiling

Usage: python early_exit_analysis.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")
CAPS = [4000, 5000, 5500, 6000, 7000]
MAX_TOKENS = 10_000
ORACLE_SAVINGS_RATE = 0.435  # 43.5% max savings (non-convergence rate)


def load_records():
    records_file = CHECKPOINT_DIR / "records.json"
    if not records_file.exists():
        raise FileNotFoundError(f"No records at {records_file}")
    with open(records_file) as f:
        return json.load(f)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    records = load_records()
    n = len(records)
    print(f"Loaded {n} records")

    # Baseline: total tokens if no cap (non-converged all hit 10k)
    total_tokens_no_cap = sum(r["total_tokens"] for r in records)
    # Oracle: perfect predictor aborts all non-converged at token 0
    oracle_savings = sum(r["total_tokens"] for r in records if not r["converged"])
    oracle_savings_rate = oracle_savings / total_tokens_no_cap
    print(f"Total tokens (no cap): {total_tokens_no_cap:,}")
    print(f"Oracle savings: {oracle_savings:,} ({oracle_savings_rate:.1%})")

    results = []
    for cap in CAPS:
        non_conv_aborted  = [r for r in records if not r["converged"] and r["total_tokens"] > cap]
        conv_aborted      = [r for r in records if r["converged"] and r["total_tokens"] > cap]
        non_conv_under    = [r for r in records if not r["converged"] and r["total_tokens"] <= cap]
        conv_under        = [r for r in records if r["converged"] and r["total_tokens"] <= cap]

        tokens_saved  = sum(MAX_TOKENS - cap for _ in non_conv_aborted)
        tokens_lost   = sum(r["total_tokens"] - cap for r in conv_aborted)
        net_savings   = tokens_saved - tokens_lost
        net_rate      = net_savings / total_tokens_no_cap

        print(
            f"cap={cap:5d}: non_conv_aborted={len(non_conv_aborted):3d} "
            f"conv_aborted={len(conv_aborted):3d} "
            f"tokens_saved={tokens_saved:7,} tokens_lost={tokens_lost:7,} "
            f"net={net_savings:7,} ({net_rate:.1%})"
        )

        results.append({
            "cap": cap,
            "n_non_conv_aborted": len(non_conv_aborted),
            "n_conv_false_abort": len(conv_aborted),
            "n_non_conv_under_cap": len(non_conv_under),
            "n_conv_under_cap": len(conv_under),
            "tokens_saved": tokens_saved,
            "tokens_lost": tokens_lost,
            "net_tokens_saved": net_savings,
            "net_savings_rate": net_rate,
        })

    # Plot
    caps       = [r["cap"] for r in results]
    net_rates  = [r["net_savings_rate"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(caps, [x * 100 for x in net_rates], marker="o", linewidth=2, label="Fixed-cap net savings")
    ax.axhline(oracle_savings_rate * 100, color="green", linestyle="--",
               label=f"Oracle ceiling ({oracle_savings_rate:.1%})")
    ax.set_xlabel("Token Cap")
    ax.set_ylabel("Net Tokens Saved (%)")
    ax.set_title("Early-Exit Fixed-Cap Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "early_exit_analysis.pdf", bbox_inches="tight")
    plt.close()

    out = {
        "n_problems": n,
        "total_tokens_no_cap": total_tokens_no_cap,
        "oracle_savings": oracle_savings,
        "oracle_savings_rate": oracle_savings_rate,
        "by_cap": results,
    }
    with open(RESULTS_DIR / "early_exit_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/early_exit_analysis.json")


if __name__ == "__main__":
    main()
