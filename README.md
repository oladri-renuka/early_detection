# Mechanistic Early-Detection of Reasoning Non-Convergence

Can early activations predict a stuck generation before it behaviorally fails?

## Key Finding

A logistic regression probe trained on **internal activations** at 150 tokens into the thinking chain predicts eventual convergence/non-convergence with **AUC 0.612** (p=0.001), significantly outperforming a behavioral-only baseline (AUC 0.445) that uses only surface signals (entropy, repetition) available at the same checkpoint.

This demonstrates that the model's internal representations carry early predictive signal about reasoning non-convergence that is invisible from text/logit statistics alone.

## Results

| Checkpoint | Activation AUC | Behavioral AUC | Delta | p-value | Verdict |
|:---:|:---:|:---:|:---:|:---:|:---|
| 100 tokens | 0.561 ± 0.033 | 0.419 ± 0.047 | +0.143 | 0.018 | Activation beats baseline |
| **150 tokens** | **0.612 ± 0.039** | **0.445 ± 0.052** | **+0.167** | **0.001** | **Activation beats baseline** |
| 200 tokens | 0.562 ± 0.117 | 0.513 ± 0.050 | +0.049 | 0.512 | Roughly ties |
| 300 tokens | 0.619 ± 0.065 | 0.555 ± 0.052 | +0.064 | 0.102 | Roughly ties |

![AUC vs Checkpoint Position](results/auc_vs_checkpoint.png)

## Generation Summary

- **Model**: DeepSeek-R1-Distill-Qwen-7B (fp16)
- **Dataset**: 200 AIME problems (gneubig/aime-1983-2024, seed=42)
- **Convergence rate**: 62.0% (original project: 56.5%)
- **Converged accuracy**: 90.3% | **Non-converged accuracy**: 6.6%
- **Hook layer**: 16/28 (60% depth)
- **GPU**: RTX PRO 4000 (24GB VRAM)

## Extends

[token-efficiency-math-reasoning](https://github.com/oladri-renuka/token-efficiency-math-reasoning) — the original project that discovered the bimodal convergence split.

## Setup (RunPod)

```bash
bash setup_runpod.sh
source venv/bin/activate
```

## Run

```bash
# 1. Verify hooks work (< 1 min)
python verify_hooks.py

# 2. Generation run (~8-12 hours, run in tmux)
python generate.py

# 3. Train probes + produce results (~2 min)
python analyze.py

# 4. (Optional) Sweep checkpoint positions
python generate.py --checkpoints 100,150,200,300
python analyze.py --checkpoints 100,150,200,300
```
