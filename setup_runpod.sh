#!/bin/bash
set -e

echo "=== Setting up Early Detection Project on RunPod ==="

apt-get update && apt-get install -y python3-pip python3-venv tmux git

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Create persistent directories on network volume if available
if [ -d "/runpod-volume" ]; then
    mkdir -p /runpod-volume/early_detection/checkpoints
    mkdir -p /runpod-volume/early_detection/results
    echo ""
    echo ">>> Network volume detected at /runpod-volume"
    echo ">>> Checkpoints will be saved to /runpod-volume/early_detection/checkpoints/"
    echo ">>> These PERSIST even if the pod is terminated."
    echo ">>> To resume: spin up a new pod with the SAME network volume attached,"
    echo ">>> clone the repo, and re-run generate.py — it will resume automatically."
else
    echo ""
    echo ">>> WARNING: No network volume found at /runpod-volume"
    echo ">>> Checkpoints will be saved locally and LOST if the pod is terminated."
    echo ">>> Attach a network volume before running generate.py."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run inside tmux:"
echo "  tmux new -s early"
echo "  source venv/bin/activate"
echo ""
echo "Step 1 — Verify hooks work (< 1 min, catches all config issues):"
echo "  python verify_hooks.py"
echo ""
echo "Step 2 — Run generation (~8-12 hours, checkpoints every sample):"
echo "  python generate.py"
echo ""
echo "Step 3 — Train probes and produce final results (~2 min):"
echo "  python analyze.py"
echo ""
echo "Step 4 (optional) — Sweep checkpoint positions (~24-36 hours):"
echo "  python generate.py --checkpoints 100,150,200,300"
echo "  python analyze.py --checkpoints 100,150,200,300"
