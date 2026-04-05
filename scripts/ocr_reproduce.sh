#!/bin/bash
# OCR Reproduction - DDRL on SD3.5-Medium
# One command, zero manual steps.
# Usage: export HF_TOKEN=xxx && bash scripts/ocr_reproduce.sh
set -euo pipefail

[ -z "${HF_TOKEN:-}" ] && echo "Set HF_TOKEN first" && exit 1

export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
mkdir -p $HF_HOME $TORCH_HOME /workspace/ocr_results

huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true

cd /workspace
[ ! -d diffusion-tinker ] && git clone https://github.com/srijitiyer/diffusion-tinker.git
cd diffusion-tinker
git pull --quiet
pip install -e . --quiet 2>&1 | tail -1

# Fix torch version for diffusers compat (RunPod template ships 2.4, need 2.5+)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 --quiet 2>&1 | tail -1

# Install easyocr WITHOUT letting it touch torch
pip install easyocr --no-deps --quiet 2>&1 | tail -1
pip install pyclipper shapely python-bidi scikit-image opencv-python-headless --quiet 2>&1 | tail -1

# Force numpy compat
pip install --force-reinstall numpy --quiet 2>&1 | tail -1

# Verify everything works BEFORE starting the expensive training
python -c "
import torch; assert '2.5' in torch.__version__, f'Wrong torch: {torch.__version__}'
from diffusers import StableDiffusion3Pipeline
import easyocr
from diffusion_tinker import DDRLTrainer, DDRLConfig
print(f'All OK: torch={torch.__version__}, CUDA={torch.cuda.is_available()}')
"

echo "Starting OCR reproduction..."
python scripts/ocr_reproduce.py 2>&1 | tee /workspace/ocr_results/training.log
