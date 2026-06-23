#!/bin/bash
set -e

echo "=== Setting up Early Detection Project on RunPod ==="

apt-get update && apt-get install -y python3-pip python3-venv tmux git

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

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
