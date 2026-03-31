#!/bin/bash
# =============================================================
# OCR Reproduction: DDRL on SD3.5-Medium
# =============================================================
# Reproduces OCR accuracy results from DDRL paper (arXiv:2512.04332)
# GPU:  1x A100 80GB
# Time: ~5-7 hours (150 epochs)
# Cost: ~$7-8 on RunPod A100 PCIe
#
# Usage:
#   export HF_TOKEN=hf_xxxxx
#   bash scripts/ocr_reproduce.sh
# =============================================================
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Set HF_TOKEN first."
    exit 1
fi

export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true

cd /workspace
if [ ! -d "diffusion-tinker" ]; then
    git clone https://github.com/srijitiyer/diffusion-tinker.git
fi
cd diffusion-tinker
git pull --quiet

# Install with OCR dependencies
pip install -e ".[ocr]" --quiet 2>&1 | tail -3

# Upgrade torch if needed (RunPod 2.4 template needs 2.5+ for diffusers)
python -c "import torch; v=torch.__version__; print(f'PyTorch {v}')" 2>&1
python -c "from diffusers import StableDiffusion3Pipeline; print('diffusers: OK')" 2>&1 || {
    echo "Upgrading PyTorch..."
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --quiet 2>&1 | tail -3
}

echo ""
echo "Starting OCR reproduction..."
python scripts/ocr_reproduce.py 2>&1 | tee /workspace/ocr_results/training.log

echo ""
echo "Results in /workspace/ocr_results/"
echo "Download: tar -czf /workspace/ocr_results.tar.gz -C /workspace ocr_results/"
