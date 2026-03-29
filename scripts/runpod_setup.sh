#!/bin/bash
# RunPod setup script for diffusion-tinker
# Use with: RunPod > Deploy > PyTorch 2.4+ template > 1x A100 80GB
# Then SSH in and run: bash runpod_setup.sh

set -e

echo "=== Setting up diffusion-tinker on RunPod ==="

# 1. Clone and install
cd /workspace
git clone https://github.com/srijitiyer/diffusion-tinker.git || true
cd diffusion-tinker

pip install -e ".[rewards]" --quiet
pip install wandb --quiet

# 2. Login to HuggingFace (SD3.5 requires acceptance of license)
echo ""
echo "You need HF access to stabilityai/stable-diffusion-3.5-medium"
echo "Accept the license at: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"
echo ""
huggingface-cli login

# 3. Quick sanity check
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
from diffusion_tinker import DDRLTrainer, DDRLConfig
print('diffusion-tinker imported OK')
"

# 4. Run the demo
echo ""
echo "=== Starting DDRL training ==="
python examples/ddrl_aesthetic.py

echo "=== Done. Check ./ddrl_aesthetic_output/ for results ==="
