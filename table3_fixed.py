"""
Rebuild Table 3 with a better difficulty proxy.

Difficulty options (in order of preference):
  A) AIME problem number within year (1–15, harder = higher number)
  B) Year (older = easier, recent = harder) — weak fallback

Splits problems into Easy/Medium/Hard terciles by difficulty proxy.
Reports convergence rate and accuracy in each tercile.
Ensures arithmetic consistency with Table 2 totals.

Usage: python table3_fixed.py
"""

import json
from pathlib import Path

RESULTS_DIR    = Path("results")
CHECKPOINT_DIR = Path("checkpoints")


def load_records():
    with open(CHECKPOINT_DIR / "records.json") as f:
        return json.load(f)


def difficulty_score(record):
    """
    Primary proxy: AIME problem number (1–15).
    problem_id format: "YYYY-[I|II]-N" or "YYYY-N"
    Returns (problem_number, year) — sort by problem number first.
    """
    pid = str(record.get("problem_id", ""))
    parts = pid.replace("II", "2").replace("I", "1").split("-")
    try:
        prob_num = int(parts[-1])
    except (ValueError, IndexError):
        prob_num = 8  # middle fallback
    try:
        year = int(parts[0])
    except (ValueError, IndexError):
        year = 2000
    return prob_num, year


def tercile_stats(group, label):
    n = len(group)
    if n == 0:
        return {"label": label, "n": 0}
    n_conv    = sum(1 for r in group if r["converged"])
    n_correct = sum(1 for r in group if r.get("correct", False))
    n_conv_correct = sum(1 for r in group if r["converged"] and r.get("correct", False))
    n_nonconv_correct = sum(1 for r in group if not r["converged"] and r.get("correct", False))
    return {
        "label":                    label,
        "n":                        n,
        "n_converged":              n_conv,
        "convergence_rate":         n_conv / n,
        "n_correct":                n_correct,
        "accuracy":                 n_correct / n,
        "converged_accuracy":       n_conv_correct / n_conv if n_conv else None,
        "non_converged_accuracy":   n_nonconv_correct / (n - n_conv) if (n - n_conv) else None,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    records = load_records()
    n = len(records)
    print(f"Loaded {n} records")

    # Sort by difficulty proxy
    records_sorted = sorted(records, key=difficulty_score)

    # Split into terciles
    t1 = n // 3
    t2 = 2 * n // 3
    easy   = records_sorted[:t1]
    medium = records_sorted[t1:t2]
    hard   = records_sorted[t2:]

    easy_stats   = tercile_stats(easy,   "Easy")
    medium_stats = tercile_stats(medium, "Medium")
    hard_stats   = tercile_stats(hard,   "Hard")
    total_stats  = tercile_stats(records, "Total")

    print(f"\n{'Tercile':8} {'N':>4} {'Conv':>6} {'Acc':>6} {'ConvAcc':>8} {'NonConvAcc':>11}")
    for s in [easy_stats, medium_stats, hard_stats, total_stats]:
        conv_acc    = f"{s['converged_accuracy']:.3f}"    if s.get("converged_accuracy")    is not None else "N/A"
        nonconv_acc = f"{s['non_converged_accuracy']:.3f}" if s.get("non_converged_accuracy") is not None else "N/A"
        print(f"  {s['label']:8} {s['n']:4d} {s['convergence_rate']:6.3f} {s['accuracy']:6.3f} {conv_acc:>8} {nonconv_acc:>11}")

    # Consistency check with Table 2 totals
    total_conv    = sum(r["converged"] for r in records)
    total_correct = sum(r.get("correct", False) for r in records)
    print(f"\nConsistency check:")
    print(f"  Total converged: {total_conv}/{n} = {total_conv/n:.3f}")
    print(f"  Total correct:   {total_correct}/{n} = {total_correct/n:.3f}")
    print(f"  Sum of tercile N: {easy_stats['n']+medium_stats['n']+hard_stats['n']} (should be {n})")

    out = {
        "n_problems":       n,
        "difficulty_proxy": "AIME problem number within year (1-15)",
        "tercile_size":     t1,
        "easy":             easy_stats,
        "medium":           medium_stats,
        "hard":             hard_stats,
        "total":            total_stats,
        "consistency": {
            "total_converged": total_conv,
            "total_correct":   total_correct,
            "convergence_rate": total_conv / n,
            "accuracy":         total_correct / n,
        },
    }
    with open(RESULTS_DIR / "table3_fixed.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/table3_fixed.json")


if __name__ == "__main__":
    main()
