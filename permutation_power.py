"""
Permutation test and power analysis for activation probes.

Permutation test: shuffles convergence labels at the generation level (n=200),
reruns the full CV pipeline, builds a null distribution over 1000 permutations,
and computes a proper p-value.

Power analysis: estimates the minimum detectable AUC at 80% power / α=0.05
given n=200 and observed class balance.

Usage:
  python permutation_power.py              # runs both
  python permutation_power.py --mode perm
  python permutation_power.py --mode power
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

_NETWORK_VOLUME = Path("/runpod-volume")
if _NETWORK_VOLUME.exists():
    CHECKPOINT_DIR = _NETWORK_VOLUME / "early_detection" / "checkpoints"
    RESULTS_DIR    = _NETWORK_VOLUME / "early_detection" / "results"
else:
    CHECKPOINT_DIR = Path("checkpoints")
    RESULTS_DIR    = Path("results")

RECORDS_FILE        = CHECKPOINT_DIR / "records.json"
LAYER_SWEEP_FILE    = CHECKPOINT_DIR / "layer_sweep.pt"
N_PERMUTATIONS      = 1000
TARGET_LAYER        = 16   # layer used in checkpoint sweep
TARGET_CP           = 150  # token position
ALPHA               = 0.05
POWER_TARGET        = 0.80
N_CV_SPLITS         = 5


# ── CV probe (same as analyze.py) ─────────────────────────────────────────────
def run_cv_probe(X, y, n_splits=N_CV_SPLITS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, random_state=42)
        clf.fit(X_train_s, y_train)
        y_prob = clf.predict_proba(X_test_s)[:, 1]
        aucs.append(roc_auc_score(y_test, y_prob))
    return np.array(aucs).mean() if aucs else 0.5


def load_activation_matrix():
    """Load activations for TARGET_LAYER at TARGET_CP into (X, y)."""
    if not RECORDS_FILE.exists():
        raise FileNotFoundError(f"No records at {RECORDS_FILE}. Run generate.py first.")
    with open(RECORDS_FILE) as f:
        records = json.load(f)
    y = np.array([1 if r["converged"] else 0 for r in records])

    # Try checkpoint_acts first, fall back to layer_sweep
    cp_file = CHECKPOINT_DIR / "checkpoint_acts.pt"
    ls_file  = LAYER_SWEEP_FILE

    if cp_file.exists():
        cp_acts = torch.load(cp_file, map_location="cpu", weights_only=True)
        pool = cp_acts.get(TARGET_CP, {})
        source = f"checkpoint_acts cp={TARGET_CP}"
    elif ls_file.exists():
        layer_sweep = torch.load(ls_file, map_location="cpu", weights_only=True)
        pool = {i: layer_sweep.get(i, {}).get(TARGET_LAYER) for i in range(len(records))}
        pool = {k: v for k, v in pool.items() if v is not None}
        source = f"layer_sweep layer={TARGET_LAYER}"
    else:
        raise FileNotFoundError("No activation files found.")

    valid_idx = sorted(k for k in pool if pool[k] is not None)
    X = np.stack([pool[i].squeeze(0).numpy() for i in valid_idx])
    y_sub = y[np.array(valid_idx)]
    print(f"Loaded {len(valid_idx)} samples from {source}")
    return X, y_sub


# ── Permutation test ───────────────────────────────────────────────────────────
def run_permutation_test(X, y, n_perms=N_PERMUTATIONS, rng_seed=0):
    print(f"\n{'='*60}")
    print(f"PERMUTATION TEST  (n_perms={n_perms}, layer={TARGET_LAYER}, cp={TARGET_CP})")
    print(f"{'='*60}")

    rng = np.random.default_rng(rng_seed)

    # Observed statistic
    observed_auc = run_cv_probe(X, y)
    print(f"  Observed AUC: {observed_auc:.4f}")

    # Null distribution
    null_aucs = []
    for i in range(n_perms):
        y_perm = rng.permutation(y)
        null_aucs.append(run_cv_probe(X, y_perm))
        if (i + 1) % 100 == 0:
            print(f"  Permutation {i+1}/{n_perms}  null_mean={np.mean(null_aucs):.4f}")

    null_aucs = np.array(null_aucs)
    p_value = (null_aucs >= observed_auc).mean()

    print(f"\n  Null distribution: mean={null_aucs.mean():.4f}  std={null_aucs.std():.4f}")
    print(f"  Observed AUC:      {observed_auc:.4f}")
    print(f"  p-value:           {p_value:.4f}  ({'significant' if p_value < ALPHA else 'not significant'} at α={ALPHA})")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null_aucs, bins=40, alpha=0.7, label=f"Null ({n_perms} permutations)")
    ax.axvline(observed_auc, color="red", linewidth=2, label=f"Observed AUC={observed_auc:.3f}")
    ax.axvline(np.percentile(null_aucs, 95), color="orange", linestyle="--",
               label=f"95th percentile={np.percentile(null_aucs, 95):.3f}")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test — Activation Probe (layer {TARGET_LAYER}, token {TARGET_CP})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "permutation_test.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {out}")

    return {
        "observed_auc": float(observed_auc),
        "p_value": float(p_value),
        "null_mean": float(null_aucs.mean()),
        "null_std": float(null_aucs.std()),
        "null_95th": float(np.percentile(null_aucs, 95)),
        "n_permutations": n_perms,
        "significant": bool(p_value < ALPHA),
    }


# ── Power analysis ─────────────────────────────────────────────────────────────
def run_power_analysis(y, alpha=ALPHA, power=POWER_TARGET, n_sim=2000, rng_seed=1):
    """
    Simulate: for a range of true AUC effect sizes, how often does the
    5-fold CV probe reject the null at level alpha?
    Reports the minimum detectable AUC at the target power.
    """
    print(f"\n{'='*60}")
    print(f"POWER ANALYSIS  (α={alpha}, target_power={power}, n_sim={n_sim})")
    print(f"{'='*60}")

    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    print(f"  n={n}, n_converged={n_pos}, n_non_converged={n_neg}, balance={n_pos/n:.3f}")

    rng = np.random.default_rng(rng_seed)

    # Candidate effect sizes (true AUC)
    candidate_aucs = np.arange(0.50, 0.85, 0.02)
    empirical_powers = []

    # Critical value: AUC that beats chance at level alpha under the null.
    # We approximate this via a normal approximation on the fold-level AUCs.
    # Simpler: for each true AUC, simulate n_sim datasets and check rejection rate.

    for true_auc in candidate_aucs:
        rejections = 0
        for _ in range(n_sim):
            # Simulate scores that yield approximately true_auc:
            # positives drawn from N(true_auc, 0.15), negatives from N(0, 0.15)
            scores_pos = rng.normal(loc=true_auc, scale=0.15, size=n_pos)
            scores_neg = rng.normal(loc=0.0,      scale=0.15, size=n_neg)
            scores = np.concatenate([scores_pos, scores_neg])
            labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

            # Shuffle together
            idx = rng.permutation(n)
            scores, labels = scores[idx], labels[idx]

            # 5-fold CV AUC
            skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True,
                                  random_state=int(rng.integers(1e6)))
            fold_aucs = []
            for tr, te in skf.split(scores.reshape(-1, 1), labels):
                if len(np.unique(labels[te])) < 2:
                    continue
                fold_aucs.append(roc_auc_score(labels[te], scores[tr].mean() + scores[te]))

            if not fold_aucs:
                continue

            # One-sided t-test: H0: mean AUC = 0.5
            fold_aucs = np.array(fold_aucs)
            t, p = stats.ttest_1samp(fold_aucs, 0.5)
            if t > 0 and p / 2 < alpha:
                rejections += 1

        emp_power = rejections / n_sim
        empirical_powers.append(emp_power)
        print(f"  True AUC={true_auc:.2f}  empirical power={emp_power:.3f}")

    empirical_powers = np.array(empirical_powers)

    # Find minimum detectable AUC
    detectable = candidate_aucs[empirical_powers >= power]
    mda = float(detectable[0]) if len(detectable) > 0 else float("nan")
    print(f"\n  Minimum detectable AUC at {power:.0%} power: {mda:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(candidate_aucs, empirical_powers, marker="o", linewidth=2)
    ax.axhline(power, color="orange", linestyle="--", label=f"Target power={power:.0%}")
    ax.axhline(alpha, color="gray",   linestyle=":",  label=f"α={alpha}")
    if not np.isnan(mda):
        ax.axvline(mda, color="red", linestyle="--", label=f"MDA={mda:.2f}")
    ax.set_xlabel("True AUC")
    ax.set_ylabel("Empirical Power")
    ax.set_title(f"Power Analysis — n={n}, balance={n_pos/n:.2f}, α={alpha}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "power_analysis.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {out}")

    return {
        "n": n,
        "n_converged": n_pos,
        "class_balance": float(n_pos / n),
        "alpha": alpha,
        "target_power": power,
        "minimum_detectable_auc": mda,
        "candidate_aucs": candidate_aucs.tolist(),
        "empirical_powers": empirical_powers.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["perm", "power", "all"], default="all")
    parser.add_argument("--n_perms", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    with open(RECORDS_FILE) as f:
        records = json.load(f)
    y = np.array([1 if r["converged"] else 0 for r in records])

    output = {}

    if args.mode in ("perm", "all"):
        X, y_sub = load_activation_matrix()
        output["permutation_test"] = run_permutation_test(X, y_sub, n_perms=args.n_perms)

    if args.mode in ("power", "all"):
        output["power_analysis"] = run_power_analysis(y)

    out_path = RESULTS_DIR / "permutation_power_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
