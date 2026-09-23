"""
CPU recomputation: token sweep + layer sweep + power analysis
using the same full-sample permutation null as the primary evaluation.

Changes vs original token_sweep_fixed_c1.py:
  OLD null: 1000x StratifiedShuffleSplit 80/20 (null SD ~0.10)
  NEW null: 1000x full-label permutation + 5-fold CV (null SD ~0.060)

This matches permutation_power.py exactly, so p-values are comparable
to the primary result (Figure 6).

Outputs:
  token_sweep_fullnull.json   → replaces token_sweep_fixed_c1.json values
  layer_sweep_fullnull.json   → replaces layer_sweep.pt table values
  power_recheck.json          → MDA using permutation null, not Hanley-McNeil

Runtime: ~20-40 min on CPU (no GPU needed).

Usage:
  python sweep_rerun_fullnull.py           # all three
  python sweep_rerun_fullnull.py --token   # token sweep only
  python sweep_rerun_fullnull.py --layer   # layer sweep only
  python sweep_rerun_fullnull.py --power   # power recheck only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
ACTS_PATH        = BASE / "checkpoints" / "checkpoint_acts.pt"
LAYER_SWEEP_PATH = BASE / "checkpoints" / "layer_sweep.pt"
RECORDS_PATH     = BASE / "checkpoints" / "records.json"
OUT_DIR          = Path("/Users/renukaoladri/Downloads/early_detection_25")

TOKEN_POSITIONS = [50, 75, 100, 125, 150, 175, 200, 250, 300]
LAYERS          = [0, 5, 10, 15, 20, 25, 27]
TOKEN_FOR_LAYER = 150   # layer sweep is at token 150

N_PERMS   = 1000
N_FOLDS   = 5
C_FIXED   = 1.0
SEED      = 42
ALPHA     = 0.05
BONF_THR  = ALPHA / len(TOKEN_POSITIONS)   # 0.0056 for 9 positions


# ── Labels ─────────────────────────────────────────────────────────────────────
def load_labels():
    records = sorted(json.load(open(RECORDS_PATH)), key=lambda r: r["idx"])
    y = np.array([int(r["converged"]) for r in records])
    assert len(y) == 200 and y.sum() == 115, f"Expected 115/200, got {y.sum()}"
    print(f"Labels: {int(y.sum())}/200 converged (57.5%, seed-42 Run A)")
    return y


# ── CV AUC (5-fold, same as permutation_power.py) ─────────────────────────────
def cv_auc(X, y, seed=SEED):
    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf  = LogisticRegression(C=C_FIXED, solver="liblinear", max_iter=1000,
                                  random_state=seed)
        clf.fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs))


# ── Full-sample permutation null (matches permutation_power.py) ────────────────
def permutation_p(X, y, obs_auc, rng):
    """
    Permute all 200 labels, run 5-fold CV, repeat N_PERMS times.
    This is the same null as the primary evaluation (null SD ≈ 0.060).
    """
    null = []
    for i in range(N_PERMS):
        yp = rng.permutation(y)
        null.append(cv_auc(X, yp))
        if (i + 1) % 200 == 0:
            print(f"      perm {i+1}/{N_PERMS}  null_mean={np.mean(null):.4f}", flush=True)
    null = np.array(null)
    p    = float((null >= obs_auc).mean())
    return p, null


def make_X(raw, dim=3584):
    X = np.zeros((200, dim), dtype=np.float32)
    for idx, t in raw.items():
        X[idx] = t.squeeze().numpy()
    return X


# ── Token sweep ────────────────────────────────────────────────────────────────
def run_token_sweep(y):
    print("\n" + "="*60)
    print("TOKEN SWEEP — full-sample permutation null")
    print("="*60)

    print(f"Loading {ACTS_PATH.name}...")
    t0 = time.time()
    acts = torch.load(ACTS_PATH, map_location="cpu", weights_only=False)
    print(f"Loaded in {time.time()-t0:.1f}s  positions: {sorted(acts.keys())}")

    rng = np.random.default_rng(SEED)
    results = []

    print(f"\n{'Pos':>4}  {'AUC':>6}  {'p':>6}  {'null_SD':>8}  {'Bonf?':>8}")
    print("-" * 45)

    for pos in TOKEN_POSITIONS:
        print(f"\n--- Token {pos} ---", flush=True)
        t0 = time.time()
        X  = make_X(acts[pos])

        auc = cv_auc(X, y)
        print(f"  Observed AUC = {auc:.4f}", flush=True)
        print(f"  Running {N_PERMS} permutations...", flush=True)

        p, null = permutation_p(X, y, auc, rng)
        elapsed = time.time() - t0

        bonf = "SURVIVES" if p < BONF_THR else ""
        print(f"{pos:4d}  {auc:.3f}  {p:.3f}  {null.std():.5f}  {bonf}  ({elapsed:.0f}s)")

        results.append({
            "token_pos":  pos,
            "auc_mean":   round(auc, 6),
            "p_value":    round(p, 6),
            "null_mean":  round(float(null.mean()), 6),
            "null_std":   round(float(null.std()), 6),
            "n_perms":    N_PERMS,
            "bonferroni_survives": p < BONF_THR,
        })

    out = OUT_DIR / "token_sweep_fullnull.json"
    json.dump({
        "description": "Token sweep, layer-20 activations, fixed C=1.0, liblinear, "
                       "full-sample permutation null (matches primary evaluation)",
        "null_method": "Full-label permutation + 5-fold CV (same as permutation_power.py)",
        "null_note":   "null SD ~0.060, comparable to Figure 6 null",
        "bonferroni_threshold": BONF_THR,
        "results": results,
    }, open(out, "w"), indent=2)
    print(f"\nSaved → {out}")
    return results


# ── Layer sweep ────────────────────────────────────────────────────────────────
def run_layer_sweep(y):
    print("\n" + "="*60)
    print("LAYER SWEEP — full-sample permutation null")
    print("="*60)

    print(f"Loading {LAYER_SWEEP_PATH.name}...")
    t0 = time.time()
    ls = torch.load(LAYER_SWEEP_PATH, map_location="cpu", weights_only=False)
    print(f"Loaded in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(SEED + 1)
    results = []

    print(f"\n{'Layer':>6}  {'AUC':>6}  {'p':>6}  {'null_SD':>8}")
    print("-" * 35)

    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---", flush=True)
        t0 = time.time()
        # layer_sweep.pt: {problem_idx: {layer_idx: tensor(1, 3584)}}
        raw = {idx: ls[idx][layer] for idx in sorted(ls.keys()) if layer in ls[idx]}
        X   = make_X(raw)

        auc = cv_auc(X, y)
        print(f"  Observed AUC = {auc:.4f}", flush=True)
        print(f"  Running {N_PERMS} permutations...", flush=True)

        p, null = permutation_p(X, y, auc, rng)
        elapsed = time.time() - t0

        print(f"{layer:6d}  {auc:.3f}  {p:.3f}  {null.std():.5f}  ({elapsed:.0f}s)")
        results.append({
            "layer":    layer,
            "auc":      round(auc, 6),
            "p_value":  round(p, 6),
            "null_mean": round(float(null.mean()), 6),
            "null_std":  round(float(null.std()), 6),
            "n_perms":   N_PERMS,
        })

    out = OUT_DIR / "layer_sweep_fullnull.json"
    json.dump({
        "description": "Layer sweep at token 150, fixed C=1.0, liblinear, "
                       "full-sample permutation null",
        "null_method": "Full-label permutation + 5-fold CV",
        "results": results,
    }, open(out, "w"), indent=2)
    print(f"\nSaved → {out}")
    return results


# ── Power recheck ──────────────────────────────────────────────────────────────
def run_power_recheck(y, null_95th=0.5950, null_std=0.0599):
    """
    Abdi's fix: use the empirical permutation null (95th pct, SD) for power,
    not Hanley-McNeil. Critical threshold = null_95th.
    MDA at 80% power ≈ null_95th + 0.84 * null_std.
    """
    print("\n" + "="*60)
    print("POWER RECHECK — permutation null, not Hanley-McNeil")
    print("="*60)
    print(f"Using null from Figure 6: 95th pct={null_95th:.4f}, SD={null_std:.4f}")

    # Analytic approximation (Abdi's formula)
    z_power = 0.8416   # z for 80%
    mda_approx = null_95th + z_power * null_std
    print(f"\nAnalytic MDA (80% power) = {null_95th:.4f} + 0.8416 × {null_std:.4f} = {mda_approx:.4f}")

    # Simulation: for each candidate true AUC, how often does a draw from
    # N(true_auc, null_std) exceed null_95th?  (assumes symmetric, same SD)
    rng = np.random.default_rng(SEED)
    candidate_aucs = np.arange(0.51, 0.85, 0.005)
    sim_powers = []

    for true_auc in candidate_aucs:
        # Draw observed AUC from N(true_auc, null_std)
        obs = rng.normal(loc=true_auc, scale=null_std, size=10000)
        power = float((obs > null_95th).mean())
        sim_powers.append(power)

    sim_powers = np.array(sim_powers)
    detectable = candidate_aucs[sim_powers >= 0.80]
    mda_sim = float(detectable[0]) if len(detectable) > 0 else float("nan")

    print(f"Simulation MDA (80% power) = {mda_sim:.4f}")
    print("\nPower at specific AUCs:")
    for check in [0.55, 0.60, 0.62, 0.65, 0.70, 0.73, 0.75]:
        idx = np.argmin(np.abs(candidate_aucs - check))
        print(f"  AUC={check:.2f}  power={sim_powers[idx]:.3f}")

    out = OUT_DIR / "power_recheck.json"
    result = {
        "null_95th":    null_95th,
        "null_std":     null_std,
        "mda_analytic": round(mda_approx, 4),
        "mda_sim":      round(mda_sim, 4),
        "method":       "permutation null (not Hanley-McNeil)",
        "note":         "Hanley-McNeil gives MDA=0.73; permutation null gives MDA~0.65",
        "candidate_aucs": candidate_aucs.tolist(),
        "sim_powers":     sim_powers.tolist(),
    }
    json.dump(result, open(out, "w"), indent=2)
    print(f"\nSaved → {out}")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", action="store_true")
    parser.add_argument("--layer", action="store_true")
    parser.add_argument("--power", action="store_true")
    args = parser.parse_args()
    run_all = not (args.token or args.layer or args.power)

    y = load_labels()

    if run_all or args.token:
        run_token_sweep(y)

    if run_all or args.layer:
        run_layer_sweep(y)

    if run_all or args.power:
        run_power_recheck(y)

    print("\nDone. Update paper.tex once you see the new p-values.")
    print("Key: if token 125 or 175 now have p<0.05 under this null,")
    print("remove 'all p>0.08' and 'uniformly null' from Introduction/Contributions.")


if __name__ == "__main__":
    main()
