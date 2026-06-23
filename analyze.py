"""
Analysis script — trains probes, compares activation vs behavioral baseline,
produces all results and plots. Runs in ~2 minutes on CPU, no GPU needed.

Usage:
  python analyze.py                          # analyze checkpoint at 150 tokens
  python analyze.py --checkpoints 100,150,200,300   # analyze sweep
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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


def load_data(checkpoint_pos):
    json_path = CHECKPOINT_DIR / f"generation_cp{checkpoint_pos}.json"
    act_path = CHECKPOINT_DIR / f"activations_cp{checkpoint_pos}.pt"

    if not json_path.exists():
        raise FileNotFoundError(f"No generation data at {json_path}")

    with open(json_path) as f:
        records = json.load(f)

    activations = {}
    if act_path.exists():
        activations = torch.load(act_path, map_location="cpu", weights_only=True)

    return records, activations


def sanity_check(records):
    """Step 2: Compare against original project's rates."""
    n = len(records)
    n_conv = sum(1 for r in records if r["converged"])
    n_correct = sum(1 for r in records if r["correct"])
    conv_rate = n_conv / n
    acc = n_correct / n

    conv_correct = sum(1 for r in records if r["converged"] and r["correct"])
    n_nonconv = n - n_conv
    nonconv_correct = sum(1 for r in records if not r["converged"] and r["correct"])

    print("=" * 60)
    print("SANITY CHECK vs Original Project")
    print("=" * 60)
    print(f"  N = {n}")
    print(f"  Convergence rate:  {conv_rate:.1%}  (original: ~56.5%)")
    print(f"  Overall accuracy:  {acc:.1%}")
    if n_conv > 0:
        print(f"  Converged acc:     {conv_correct / n_conv:.1%}  (original: ~96.5%)")
    if n_nonconv > 0:
        print(f"  Non-conv acc:      {nonconv_correct / n_nonconv:.1%}  (original: ~11.5%)")

    # Flag large deviations
    if abs(conv_rate - 0.565) > 0.15:
        print(f"\n  ⚠ WARNING: Convergence rate {conv_rate:.1%} deviates significantly from ~56.5%")
        print("    This may indicate a problem with the generation setup.")
        print("    Check: chat template, thinking=True init, model version.")
    else:
        print(f"\n  ✓ Convergence rate is within expected range")

    avg_tokens_conv = np.mean([r["total_tokens"] for r in records if r["converged"]]) if n_conv else 0
    avg_tokens_nonconv = np.mean([r["total_tokens"] for r in records if not r["converged"]]) if n_nonconv else 0
    print(f"\n  Avg tokens (converged):     {avg_tokens_conv:.0f}  (original: ~4100)")
    print(f"  Avg tokens (non-converged): {avg_tokens_nonconv:.0f}  (original: ~10000)")

    return conv_rate


def prepare_features(records, activations, checkpoint_pos):
    """
    Prepare feature matrices for both probes.
    Returns: X_act, X_behav, y, valid_mask
    """
    n = len(records)
    labels = np.array([1 if r["converged"] else 0 for r in records])

    # Behavioral features
    behav_keys = [
        "entropy_mean", "entropy_max", "entropy_std", "entropy_trend",
        "repetition_bigram", "repetition_trigram", "token_count",
    ]
    X_behav = np.zeros((n, len(behav_keys)))
    for i, r in enumerate(records):
        bf = r["behavioral_features"]
        X_behav[i] = [bf[k] for k in behav_keys]

    # Activation features — mean-pooled and last-position
    # Since we capture at the checkpoint position, we have a [1, hidden_dim] vector
    valid_act = []
    act_vectors = []
    for i in range(n):
        act = activations.get(i)
        if act is not None:
            valid_act.append(i)
            # act shape: [1, hidden_dim]
            act_vectors.append(act.squeeze(0).numpy())

    if not act_vectors:
        return None, X_behav, labels, None

    X_act_full = np.stack(act_vectors)  # [n_valid, hidden_dim]

    return X_act_full, X_behav, labels, np.array(valid_act)


def run_cv_probe(X, y, name, n_splits=5):
    """
    Run stratified k-fold CV with logistic regression.
    Returns per-fold AUCs, accuracies.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aucs = []
    accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Check that both classes present in train and test
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  {name} fold {fold}: skipped (single class in fold)")
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            max_iter=2000, solver="lbfgs", C=1.0, random_state=42
        )
        clf.fit(X_train_s, y_train)

        y_prob = clf.predict_proba(X_test_s)[:, 1]
        y_pred = clf.predict(X_test_s)

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        aucs.append(auc)
        accs.append(acc)

    return np.array(aucs), np.array(accs)


def analyze_checkpoint(checkpoint_pos):
    """Full analysis for one checkpoint position."""
    print(f"\n{'=' * 60}")
    print(f"ANALYSIS: Checkpoint at {checkpoint_pos} tokens")
    print(f"{'=' * 60}")

    records, activations = load_data(checkpoint_pos)
    print(f"Loaded {len(records)} records, {sum(1 for v in activations.values() if v is not None)} activations")

    # Sanity check
    conv_rate = sanity_check(records)

    # Prepare features
    X_act, X_behav, y, valid_idx = prepare_features(records, activations, checkpoint_pos)

    n_pos = np.sum(y)
    n_neg = len(y) - n_pos
    print(f"\nLabel distribution: converged={n_pos}, non-converged={n_neg}")

    # ── Behavioral baseline probe ─────────────────────────────────────────
    print(f"\n--- Behavioral Baseline Probe (all {len(y)} samples) ---")
    behav_aucs, behav_accs = run_cv_probe(X_behav, y, "Behavioral")
    print(f"  AUC:      {behav_aucs.mean():.3f} ± {behav_aucs.std():.3f}  (per-fold: {behav_aucs.round(3)})")
    print(f"  Accuracy: {behav_accs.mean():.3f} ± {behav_accs.std():.3f}")

    # ── Activation probe ─────────────────────────────────────────────────
    act_result = None
    if X_act is not None and len(valid_idx) > 20:
        y_act = y[valid_idx]
        X_behav_act = X_behav[valid_idx]

        print(f"\n--- Activation Probe ({len(valid_idx)} samples with activations) ---")
        act_aucs, act_accs = run_cv_probe(X_act, y_act, "Activation")
        print(f"  AUC:      {act_aucs.mean():.3f} ± {act_aucs.std():.3f}  (per-fold: {act_aucs.round(3)})")
        print(f"  Accuracy: {act_accs.mean():.3f} ± {act_accs.std():.3f}")

        # ── Combined probe (activation + behavioral) ─────────────────────
        print(f"\n--- Combined Probe (activation + behavioral) ---")
        X_combined = np.hstack([X_act, X_behav_act])
        comb_aucs, comb_accs = run_cv_probe(X_combined, y_act, "Combined")
        print(f"  AUC:      {comb_aucs.mean():.3f} ± {comb_aucs.std():.3f}  (per-fold: {comb_aucs.round(3)})")
        print(f"  Accuracy: {comb_accs.mean():.3f} ± {comb_accs.std():.3f}")

        # ── Behavioral on same subset (fair comparison) ──────────────────
        print(f"\n--- Behavioral on same {len(valid_idx)} samples (fair comparison) ---")
        behav_sub_aucs, behav_sub_accs = run_cv_probe(X_behav_act, y_act, "Behavioral-subset")
        print(f"  AUC:      {behav_sub_aucs.mean():.3f} ± {behav_sub_aucs.std():.3f}")
        print(f"  Accuracy: {behav_sub_accs.mean():.3f} ± {behav_sub_accs.std():.3f}")

        # ── Statistical comparison ───────────────────────────────────────
        print(f"\n--- Comparison ---")
        delta_auc = act_aucs.mean() - behav_sub_aucs.mean()
        if len(act_aucs) == len(behav_sub_aucs) and len(act_aucs) > 1:
            t_stat, p_value = stats.ttest_rel(act_aucs, behav_sub_aucs)
            print(f"  Activation - Behavioral AUC: {delta_auc:+.3f}")
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.3f}")
        else:
            p_value = None
            print(f"  Activation - Behavioral AUC: {delta_auc:+.3f}")

        if delta_auc > 0.03 and (p_value is None or p_value < 0.05):
            verdict = "ACTIVATION PROBE BEATS BASELINE"
        elif delta_auc < -0.03 and (p_value is None or p_value < 0.05):
            verdict = "ACTIVATION PROBE UNDERPERFORMS BASELINE (unexpected — check probe setup)"
        else:
            verdict = "ACTIVATION PROBE ROUGHLY TIES BASELINE"
        print(f"\n  VERDICT: {verdict}")

        act_result = {
            "checkpoint_pos": checkpoint_pos,
            "n_samples": len(records),
            "n_with_activations": len(valid_idx),
            "convergence_rate": float(conv_rate),
            "behavioral_auc_mean": float(behav_sub_aucs.mean()),
            "behavioral_auc_std": float(behav_sub_aucs.std()),
            "behavioral_auc_folds": behav_sub_aucs.tolist(),
            "activation_auc_mean": float(act_aucs.mean()),
            "activation_auc_std": float(act_aucs.std()),
            "activation_auc_folds": act_aucs.tolist(),
            "combined_auc_mean": float(comb_aucs.mean()),
            "combined_auc_std": float(comb_aucs.std()),
            "combined_auc_folds": comb_aucs.tolist(),
            "delta_auc": float(delta_auc),
            "p_value": float(p_value) if p_value is not None else None,
            "verdict": verdict,
        }
    else:
        print("\n  ⚠ Not enough activations for activation probe")
        act_result = {
            "checkpoint_pos": checkpoint_pos,
            "n_samples": len(records),
            "n_with_activations": len(valid_idx) if valid_idx is not None else 0,
            "error": "insufficient activations",
        }

    return act_result


def plot_sweep(results):
    """Plot AUC vs checkpoint position for the sweep."""
    positions = []
    act_aucs = []
    behav_aucs = []
    comb_aucs = []
    act_stds = []
    behav_stds = []
    comb_stds = []

    for r in results:
        if "error" in r:
            continue
        positions.append(r["checkpoint_pos"])
        act_aucs.append(r["activation_auc_mean"])
        behav_aucs.append(r["behavioral_auc_mean"])
        comb_aucs.append(r["combined_auc_mean"])
        act_stds.append(r["activation_auc_std"])
        behav_stds.append(r["behavioral_auc_std"])
        comb_stds.append(r["combined_auc_std"])

    if len(positions) < 2:
        print("Not enough valid checkpoints for sweep plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.errorbar(positions, act_aucs, yerr=act_stds, marker="o", label="Activation Probe", capsize=4)
    ax.errorbar(positions, behav_aucs, yerr=behav_stds, marker="s", label="Behavioral Baseline", capsize=4)
    ax.errorbar(positions, comb_aucs, yerr=comb_stds, marker="^", label="Combined", capsize=4)

    ax.set_xlabel("Checkpoint Position (tokens into thinking chain)")
    ax.set_ylabel("AUC (5-fold CV)")
    ax.set_title("Early Detection of Reasoning Non-Convergence")
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "auc_vs_checkpoint.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSweep plot saved to {out_path}")


def plot_single(result):
    """Bar chart comparing probes for a single checkpoint."""
    if "error" in result:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    names = ["Behavioral\nBaseline", "Activation\nProbe", "Combined"]
    means = [result["behavioral_auc_mean"], result["activation_auc_mean"], result["combined_auc_mean"]]
    stds = [result["behavioral_auc_std"], result["activation_auc_std"], result["combined_auc_std"]]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    bars = ax.bar(names, means, yerr=stds, color=colors, capsize=6, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("AUC (5-fold CV)")
    ax.set_title(f"Probe Comparison at {result['checkpoint_pos']} Tokens")
    ax.set_ylim(0.4, 1.05)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out_path = RESULTS_DIR / f"probe_comparison_cp{result['checkpoint_pos']}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="150",
        help="Comma-separated checkpoint positions to analyze",
    )
    args = parser.parse_args()
    checkpoint_positions = [int(x.strip()) for x in args.checkpoints.split(",")]

    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = []
    for cp in checkpoint_positions:
        try:
            result = analyze_checkpoint(cp)
            all_results.append(result)
            plot_single(result)
        except FileNotFoundError as e:
            print(f"\n⚠ Skipping checkpoint {cp}: {e}")

    if len(all_results) > 1:
        plot_sweep(all_results)

    # Save final results
    out_path = RESULTS_DIR / "early_detection_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal results saved to {out_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for r in all_results:
        if "error" in r:
            print(f"  Checkpoint {r['checkpoint_pos']}: {r['error']}")
        else:
            print(f"  Checkpoint {r['checkpoint_pos']}:")
            print(f"    Behavioral AUC: {r['behavioral_auc_mean']:.3f} ± {r['behavioral_auc_std']:.3f}")
            print(f"    Activation AUC: {r['activation_auc_mean']:.3f} ± {r['activation_auc_std']:.3f}")
            print(f"    Combined AUC:   {r['combined_auc_mean']:.3f} ± {r['combined_auc_std']:.3f}")
            print(f"    Delta (Act-Behav): {r['delta_auc']:+.3f}")
            if r["p_value"] is not None:
                print(f"    p-value: {r['p_value']:.3f}")
            print(f"    VERDICT: {r['verdict']}")


if __name__ == "__main__":
    main()
