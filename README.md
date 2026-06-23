# Mechanistic Early-Detection of Reasoning Non-Convergence

Can early activations predict a stuck generation before it behaviorally fails?

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
