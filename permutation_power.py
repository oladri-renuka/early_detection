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
def hanley_mcneil_var(auc, n_pos, n_neg):
    """
    Hanley & McNeil (1982) variance of the AUC estimator.
    V(AUC) = (auc*(1-auc) + (n_pos-1)*Q1 + (n_neg-1)*Q2) / (n_pos*n_neg)
    where Q1 = auc/(2-auc), Q2 = 2*auc^2/(1+auc).
    """
    Q1 = auc / (2 - auc)
    Q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * Q1 + (n_neg - 1) * Q2) / (n_pos * n_neg)
    return var


def run_power_analysis(y, alpha=ALPHA, power=POWER_TARGET):
    """
    Closed-form power analysis using the Hanley-McNeil (1982) variance formula.
    For a one-sided z-test of H0: AUC = 0.5 vs H1: AUC > 0.5.
    """
    print(f"\n{'='*60}")
    print(f"POWER ANALYSIS  (α={alpha}, target_power={power}, Hanley-McNeil)")
    print(f"{'='*60}")

    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    print(f"  n={n}, n_converged={n_pos}, n_non_converged={n_neg}, balance={n_pos/n:.3f}")

    z_alpha = stats.norm.ppf(1 - alpha)       # one-sided critical value
    z_power = stats.norm.ppf(power)

    candidate_aucs = np.arange(0.51, 0.85, 0.01)
    achieved_powers = []

    for true_auc in candidate_aucs:
        # Variance under H1 (true AUC)
        var_h1  = hanley_mcneil_var(true_auc, n_pos, n_neg)
        se_h1   = np.sqrt(var_h1)
        # Variance under H0 (AUC = 0.5)
        var_h0  = hanley_mcneil_var(0.5, n_pos, n_neg)
        se_h0   = np.sqrt(var_h0)
        # Power: P(Z > z_alpha | true AUC)
        z = (true_auc - 0.5 - z_alpha * se_h0) / se_h1
        pwr = float(stats.norm.cdf(z))
        achieved_powers.append(pwr)

    achieved_powers = np.array(achieved_powers)

    # Minimum detectable AUC at target power
    detectable = candidate_aucs[achieved_powers >= power]
    mda = float(detectable[0]) if len(detectable) > 0 else float("nan")

    # Also compute analytically
    var_h0 = hanley_mcneil_var(0.5, n_pos, n_neg)
    se_h0  = np.sqrt(var_h0)
    # Solve: (mda - 0.5) / se_h1 = z_alpha + z_power  (approx with se_h1 ≈ se_h0)
    mda_approx = 0.5 + (z_alpha + z_power) * se_h0

    print(f"  Minimum detectable AUC at {power:.0%} power: {mda:.3f}")
    print(f"  (analytic approximation: {mda_approx:.3f})")

    # Table of key AUC thresholds
    for auc_check in [0.55, 0.60, 0.62, 0.65, 0.70, 0.75]:
        idx = np.argmin(np.abs(candidate_aucs - auc_check))
        print(f"  AUC={auc_check:.2f}  power={achieved_powers[idx]:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(candidate_aucs, achieved_powers, linewidth=2, label="Power curve")
    ax.axhline(power, color="orange", linestyle="--", label=f"Target power={power:.0%}")
    ax.axhline(alpha, color="gray",   linestyle=":",  label=f"α={alpha}")
    if not np.isnan(mda):
        ax.axvline(mda, color="red", linestyle="--", label=f"MDA={mda:.2f}")
    ax.set_xlabel("True AUC")
    ax.set_ylabel("Power")
    ax.set_title(f"Power Analysis (Hanley-McNeil) — n={n}, balance={n_pos/n:.2f}, α={alpha}")
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
        "mda_analytic_approx": float(mda_approx),
        "candidate_aucs": candidate_aucs.tolist(),
        "achieved_powers": achieved_powers.tolist(),
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
        output["power_analysis"] = run_power_analysis(y, alpha=ALPHA, power=POWER_TARGET)

    out_path = RESULTS_DIR / "permutation_power_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
