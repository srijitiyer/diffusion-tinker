#!/bin/bash
# =============================================================
# RunPod GPU Validation for diffusion-tinker
# =============================================================
# GPU:  1x A100 80GB ($1.64/hr community cloud)
# Time: ~60-90 minutes
# Cost: ~$2-3
#
# Prerequisites:
#   1. Accept SD3.5 license: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
#   2. Have your HF token ready
#
# Usage:
#   export HF_TOKEN=hf_xxxxx
#   bash scripts/runpod_validate.sh
# =============================================================
set -euo pipefail

echo "============================================="
echo "diffusion-tinker GPU validation"
echo "============================================="

# Check HF token
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Set HF_TOKEN environment variable first."
    echo "  export HF_TOKEN=hf_xxxxx"
    exit 1
fi

# Set cache dirs to persistent workspace storage
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
mkdir -p "$HF_HOME" "$TORCH_HOME"

# HF login
echo ""
echo "Logging into HuggingFace..."
huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true

# Clone and install
echo ""
echo "Setting up diffusion-tinker..."
cd /workspace
if [ ! -d "diffusion-tinker" ]; then
    git clone https://github.com/srijitiyer/diffusion-tinker.git
fi
cd diffusion-tinker
git pull --quiet
pip install -e . --quiet 2>&1 | tail -3

# GPU check
echo ""
python -c "
import torch
assert torch.cuda.is_available(), 'No GPU'
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f'GPU: {name} ({vram:.0f} GB)')
assert vram >= 40, f'Need >= 40GB, got {vram:.0f}'
print('GPU check: PASSED')
"

# Run validation
echo ""
echo "Starting validation (estimated 60-90 minutes)..."
echo "============================================="
python scripts/runpod_validate.py 2>&1 | tee /workspace/validation_results/training.log

# Package results
echo ""
echo "Packaging results..."
cd /workspace
tar -czf validation_results.tar.gz validation_results/
echo "Download with: runpodctl receive (or scp)"
echo "Archive: /workspace/validation_results.tar.gz"
echo ""
echo "Done."
