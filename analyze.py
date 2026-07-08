"""
Analysis script — two modes:

  1. Layer sweep: probe all 28 layers at token 150, plot AUC vs layer depth
  2. Checkpoint sweep: probe layer 16 at {50,75,100,125,150,175,200,250,300}

Both use 5-fold stratified CV with logistic regression.
AUC is reported as max(auc, 1-auc) to correct for label-flip.

Usage:
  python analyze.py              # runs both modes
  python analyze.py --mode layer
  python analyze.py --mode checkpoint
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

_NETWORK_VOLUME = Path("/runpod-volume")
if _NETWORK_VOLUME.exists():
    CHECKPOINT_DIR = _NETWORK_VOLUME / "early_detection" / "checkpoints"
    RESULTS_DIR    = _NETWORK_VOLUME / "early_detection" / "results"
    print(f"[INFO] Using network volume: {_NETWORK_VOLUME}")
else:
    CHECKPOINT_DIR = Path("checkpoints")
    RESULTS_DIR    = Path("results")

RECORDS_FILE        = CHECKPOINT_DIR / "records.json"
LAYER_SWEEP_FILE    = CHECKPOINT_DIR / "layer_sweep.pt"
CHECKPOINT_ACT_FILE = CHECKPOINT_DIR / "checkpoint_acts.pt"
CHECKPOINT_POSITIONS = [50, 75, 100, 125, 150, 175, 200, 250, 300]
LAYER_SWEEP_CHECKPOINT = 150


# ── Core CV probe ─────────────────────────────────────────────────────────────
def run_cv_probe(X, y, n_splits=5):
    """
    5-fold stratified CV with logistic regression.
    Returns per-fold AUCs (corrected: max(auc, 1-auc)).
    """
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
        auc = roc_auc_score(y_test, y_prob)
        auc = max(auc, 1 - auc)   # correct for label-flip
        aucs.append(auc)

    return np.array(aucs)


# ── Load data ─────────────────────────────────────────────────────────────────
def load_all():
    if not RECORDS_FILE.exists():
        raise FileNotFoundError(f"No records found at {RECORDS_FILE}. Run generate.py first.")

    with open(RECORDS_FILE) as f:
        records = json.load(f)

    layer_sweep    = torch.load(LAYER_SWEEP_FILE,    map_location="cpu", weights_only=True) if LAYER_SWEEP_FILE.exists() else {}
    checkpoint_acts = torch.load(CHECKPOINT_ACT_FILE, map_location="cpu", weights_only=True) if CHECKPOINT_ACT_FILE.exists() else {}

    y = np.array([1 if r["converged"] else 0 for r in records])
    return records, layer_sweep, checkpoint_acts, y


# ── Behavioral features ───────────────────────────────────────────────────────
def get_behavioral_X(records, checkpoint_pos):
    keys = ["entropy_mean", "entropy_max", "entropy_std", "entropy_trend",
            "repetition_bigram", "repetition_trigram", "token_count"]
    X = np.zeros((len(records), len(keys)))
    for i, r in enumerate(records):
        bf = r["behavioral_features"].get(str(checkpoint_pos)) or r["behavioral_features"].get(checkpoint_pos, {})
        X[i] = [bf.get(k, 0.0) for k in keys]
    return X


# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check(records):
    n = len(records)
    n_conv    = sum(1 for r in records if r["converged"])
    n_correct = sum(1 for r in records if r["correct"])
    conv_correct    = sum(1 for r in records if r["converged"] and r["correct"])
    nonconv_correct = sum(1 for r in records if not r["converged"] and r["correct"])
    n_nonconv = n - n_conv

    print("=" * 60)
    print("SANITY CHECK vs Original Project")
    print("=" * 60)
    print(f"  N = {n}")
    print(f"  Convergence rate:  {n_conv/n:.1%}  (original: ~56.5%)")
    print(f"  Overall accuracy:  {n_correct/n:.1%}")
    if n_conv:    print(f"  Converged acc:     {conv_correct/n_conv:.1%}  (original: ~96.5%)")
    if n_nonconv: print(f"  Non-conv acc:      {nonconv_correct/n_nonconv:.1%}  (original: ~11.5%)")

    if abs(n_conv/n - 0.565) > 0.15:
        print(f"\n  ⚠ WARNING: Convergence rate deviates significantly from ~56.5%")
    else:
        print(f"\n  ✓ Convergence rate within expected range")


# ── MODE 1: Layer sweep ───────────────────────────────────────────────────────
def run_layer_sweep(records, layer_sweep, y):
    print("\n" + "=" * 60)
    print(f"LAYER SWEEP — all layers at token {LAYER_SWEEP_CHECKPOINT}")
    print("=" * 60)

    n_layers = max(
        max(acts.keys()) for acts in layer_sweep.values() if acts
    ) + 1
    print(f"  Detected {n_layers} layers")
    print(f"  Samples with activations: {len(layer_sweep)}")

    layer_aucs = {}  # {layer_idx: array of per-fold AUCs}

    for layer_idx in range(n_layers):
        # Collect activation vectors for samples that have this layer
        valid_idx = []
        vecs = []
        for sample_idx in range(len(records)):
            act = layer_sweep.get(sample_idx, {}).get(layer_idx)
            if act is not None:
                valid_idx.append(sample_idx)
                vecs.append(act.squeeze(0).numpy())

        if len(vecs) < 20:
            print(f"  Layer {layer_idx:2d}: skipped (only {len(vecs)} samples)")
            continue

        X = np.stack(vecs)
        y_sub = y[np.array(valid_idx)]

        aucs = run_cv_probe(X, y_sub)
        layer_aucs[layer_idx] = aucs
        print(f"  Layer {layer_idx:2d}: AUC {aucs.mean():.3f} ± {aucs.std():.3f}")

    if not layer_aucs:
        print("  No layers had enough samples.")
        return None

    best_layer = max(layer_aucs, key=lambda k: layer_aucs[k].mean())
    best_auc   = layer_aucs[best_layer].mean()
    print(f"\n  Best layer: {best_layer}  (AUC {best_auc:.3f})")

    # Plot
    layers  = sorted(layer_aucs.keys())
    means   = [layer_aucs[l].mean() for l in layers]
    stds    = [layer_aucs[l].std()  for l in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(layers, means, yerr=stds, marker="o", linewidth=1.5, capsize=3, markersize=4)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.axvline(best_layer, color="red", linestyle=":", alpha=0.6, label=f"Best layer ({best_layer})")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel(f"AUC (5-fold CV) at token {LAYER_SWEEP_CHECKPOINT}")
    ax.set_title("Layer Sweep: Which Layers Encode Convergence?")
    ax.set_ylim(0.4, 0.9)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "layer_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to {out}")

    result = {
        "mode": "layer_sweep",
        "checkpoint_pos": LAYER_SWEEP_CHECKPOINT,
        "best_layer": best_layer,
        "best_auc_mean": float(best_auc),
        "best_auc_std": float(layer_aucs[best_layer].std()),
        "all_layers": {
            str(l): {
                "auc_mean": float(layer_aucs[l].mean()),
                "auc_std":  float(layer_aucs[l].std()),
                "auc_folds": layer_aucs[l].tolist(),
            }
            for l in layers
        },
    }
    return result


# ── MODE 2: Checkpoint sweep ──────────────────────────────────────────────────
def run_checkpoint_sweep(records, checkpoint_acts, y):
    print("\n" + "=" * 60)
    print(f"CHECKPOINT SWEEP — layer ~60% depth at {CHECKPOINT_POSITIONS} tokens")
    print("=" * 60)

    all_results = []

    for cp in CHECKPOINT_POSITIONS:
        cp_acts = checkpoint_acts.get(cp, {})

        # Activation features
        valid_idx = [i for i in range(len(records)) if cp_acts.get(i) is not None]
        if len(valid_idx) < 20:
            print(f"  cp={cp:3d}: skipped (only {len(valid_idx)} activations)")
            continue

        vecs  = [cp_acts[i].squeeze(0).numpy() for i in valid_idx]
        X_act = np.stack(vecs)
        y_act = y[np.array(valid_idx)]

        # Behavioral features (same subset)
        X_behav_all = get_behavioral_X(records, cp)
        X_behav = X_behav_all[np.array(valid_idx)]

        # Run probes
        act_aucs   = run_cv_probe(X_act,   y_act)
        behav_aucs = run_cv_probe(X_behav, y_act)
        comb_aucs  = run_cv_probe(np.hstack([X_act, X_behav]), y_act)

        # Paired t-test
        if len(act_aucs) == len(behav_aucs) and len(act_aucs) > 1:
            t_stat, p_val = stats.ttest_rel(act_aucs, behav_aucs)
        else:
            p_val = None

        delta = act_aucs.mean() - behav_aucs.mean()

        if delta > 0.03 and (p_val is None or p_val < 0.05):
            verdict = "ACTIVATION BEATS BASELINE"
        elif delta < -0.03 and (p_val is None or p_val < 0.05):
            verdict = "BASELINE BEATS ACTIVATION"
        else:
            verdict = "ROUGHLY TIED"

        print(
            f"  cp={cp:3d}: act={act_aucs.mean():.3f}±{act_aucs.std():.3f} | "
            f"behav={behav_aucs.mean():.3f}±{behav_aucs.std():.3f} | "
            f"Δ={delta:+.3f} | p={p_val:.3f if p_val else 'N/A'} | {verdict}"
        )

        all_results.append({
            "checkpoint_pos": cp,
            "n_samples": len(valid_idx),
            "activation_auc_mean":  float(act_aucs.mean()),
            "activation_auc_std":   float(act_aucs.std()),
            "activation_auc_folds": act_aucs.tolist(),
            "behavioral_auc_mean":  float(behav_aucs.mean()),
            "behavioral_auc_std":   float(behav_aucs.std()),
            "behavioral_auc_folds": behav_aucs.tolist(),
            "combined_auc_mean":    float(comb_aucs.mean()),
            "combined_auc_std":     float(comb_aucs.std()),
            "delta_auc":            float(delta),
            "p_value":              float(p_val) if p_val is not None else None,
            "verdict":              verdict,
        })

    if not all_results:
        print("  No checkpoint positions had enough data.")
        return None

    # ── Plot: activation vs behavioral across checkpoints ─────────────────────
    positions   = [r["checkpoint_pos"]        for r in all_results]
    act_means   = [r["activation_auc_mean"]   for r in all_results]
    behav_means = [r["behavioral_auc_mean"]   for r in all_results]
    comb_means  = [r["combined_auc_mean"]     for r in all_results]
    act_stds    = [r["activation_auc_std"]    for r in all_results]
    behav_stds  = [r["behavioral_auc_std"]    for r in all_results]
    comb_stds   = [r["combined_auc_std"]      for r in all_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(positions, act_means,   yerr=act_stds,   marker="o", label="Activation Probe",       capsize=4, linewidth=2)
    ax.errorbar(positions, behav_means, yerr=behav_stds, marker="s", label="Behavioral Baseline",    capsize=4, linewidth=2)
    ax.errorbar(positions, comb_means,  yerr=comb_stds,  marker="^", label="Combined",               capsize=4, linewidth=2, linestyle="--")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Checkpoint Position (tokens into thinking chain)")
    ax.set_ylabel("AUC (5-fold CV, corrected)")
    ax.set_title("Early Detection of Non-Convergence — Checkpoint Sweep")
    ax.set_xticks(positions)
    ax.legend()
    ax.set_ylim(0.4, 0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "checkpoint_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved to {out}")

    return {"mode": "checkpoint_sweep", "results": all_results}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["layer", "checkpoint", "all"], default="all")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    records, layer_sweep, checkpoint_acts, y = load_all()
    print(f"Loaded {len(records)} records")
    sanity_check(records)

    n_pos = np.sum(y)
    print(f"\nLabel distribution: converged={n_pos}, non-converged={len(y)-n_pos}")

    all_output = {}

    if args.mode in ("layer", "all"):
        result = run_layer_sweep(records, layer_sweep, y)
        if result:
            all_output["layer_sweep"] = result

    if args.mode in ("checkpoint", "all"):
        result = run_checkpoint_sweep(records, checkpoint_acts, y)
        if result:
            all_output["checkpoint_sweep"] = result

    # Save
    out_path = RESULTS_DIR / "early_detection_results.json"
    with open(out_path, "w") as f:
        json.dump(all_output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Final summary
    if "layer_sweep" in all_output:
        ls = all_output["layer_sweep"]
        print(f"\nBest layer: {ls['best_layer']}  AUC: {ls['best_auc_mean']:.3f} ± {ls['best_auc_std']:.3f}")

    if "checkpoint_sweep" in all_output:
        print("\nCheckpoint sweep summary:")
        for r in all_output["checkpoint_sweep"]["results"]:
            print(f"  {r['checkpoint_pos']:3d} tokens: act={r['activation_auc_mean']:.3f} behav={r['behavioral_auc_mean']:.3f}  {r['verdict']}")


if __name__ == "__main__":
    main()
