"""
Item-level convergence diff: greedy 7B (temp=1.0) vs temp=0.6.

Compares problem-by-problem convergence labels.
Reports flip rate and direction.

Usage: python item_level_diff.py
"""

import json
from pathlib import Path

RESULTS_DIR    = Path("results")
CHECKPOINT_DIR = Path("checkpoints")
TEMP_RESULTS   = Path("temp_robustness_results.json")


def load_greedy_labels():
    """Load convergence labels from the original greedy 7B run."""
    with open(CHECKPOINT_DIR / "records.json") as f:
        records = json.load(f)
    return {r["problem_id"]: r["converged"] for r in records}


def load_temp_labels():
    """
    Load convergence labels from temp_robustness_results.json.
    Aggregates across seeds: converged if majority of seeds converged.
    """
    with open(TEMP_RESULTS) as f:
        d = json.load(f)

    per_seed = d.get("per_seed", {}) or {k: v for k, v in d.items() if k.startswith("seed_")}

    # Build {problem_id: [converged_seed0, converged_seed1, ...]}
    problem_votes = {}
    for seed_data in per_seed.values():
        for r in seed_data["results"]:
            pid = r["problem_idx"]
            problem_votes.setdefault(pid, []).append(r["converged"])

    # Majority vote
    return {
        pid: (sum(votes) / len(votes)) >= 0.5
        for pid, votes in problem_votes.items()
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    greedy = load_greedy_labels()
    temp   = load_temp_labels()

    # Align on common problems
    common = set(greedy) & set(temp)
    print(f"Problems in greedy run:    {len(greedy)}")
    print(f"Problems in temp=0.6 run:  {len(temp)}")
    print(f"Common problems:           {len(common)}")

    flips_to_conv   = []  # greedy non-conv → temp conv
    flips_to_nonconv = []  # greedy conv → temp non-conv
    stable_conv     = []
    stable_nonconv  = []

    for pid in sorted(common):
        g = greedy[pid]
        t = temp[pid]
        if not g and t:
            flips_to_conv.append(pid)
        elif g and not t:
            flips_to_nonconv.append(pid)
        elif g and t:
            stable_conv.append(pid)
        else:
            stable_nonconv.append(pid)

    n = len(common)
    total_flips = len(flips_to_conv) + len(flips_to_nonconv)
    flip_rate   = total_flips / n

    print(f"\nStable converged:     {len(stable_conv):3d} ({len(stable_conv)/n:.1%})")
    print(f"Stable non-converged: {len(stable_nonconv):3d} ({len(stable_nonconv)/n:.1%})")
    print(f"Flip → converged:     {len(flips_to_conv):3d} ({len(flips_to_conv)/n:.1%})")
    print(f"Flip → non-converged: {len(flips_to_nonconv):3d} ({len(flips_to_nonconv)/n:.1%})")
    print(f"\nTotal flip rate:      {total_flips}/{n} = {flip_rate:.1%}")
    print(f'\nFinding: "Convergence is knife-edge for ~{flip_rate:.0%} of problems"')

    out = {
        "n_common_problems": n,
        "greedy_conv_rate":  sum(greedy[p] for p in common) / n,
        "temp06_conv_rate":  sum(temp[p]   for p in common) / n,
        "stable_converged":      len(stable_conv),
        "stable_non_converged":  len(stable_nonconv),
        "flips_to_converged":    len(flips_to_conv),
        "flips_to_non_converged": len(flips_to_nonconv),
        "total_flips":           total_flips,
        "flip_rate":             flip_rate,
        "flip_to_conv_ids":      flips_to_conv,
        "flip_to_nonconv_ids":   flips_to_nonconv,
    }
    with open(RESULTS_DIR / "item_level_diff.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/item_level_diff.json")


if __name__ == "__main__":
    main()
