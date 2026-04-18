#!/bin/bash
#SBATCH --job-name=difftinker-diag
#SBATCH --account=atlas
#SBATCH --partition=atlas
#SBATCH --nodelist=atlas24
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/atlas/u/srijit/diffusion-tinker/logs/diag-%j.out
#SBATCH --error=/atlas/u/srijit/diffusion-tinker/logs/diag-%j.err

source /atlas/u/srijit/venv/bin/activate
source /atlas/u/srijit/.env  # exports HF_TOKEN and HUGGING_FACE_HUB_TOKEN
export HF_HOME=/atlas/u/srijit/.cache/huggingface
export TORCH_HOME=/atlas/u/srijit/.cache/torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OUTPUT_DIR=/atlas/u/srijit/diffusion-tinker/output/diagnose_ocr

cd /atlas/u/srijit/diffusion-tinker

echo "=== OCR diagnostic: baseline model at different sampling settings ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/diagnose_ocr.py
echo "=== DIAGNOSTIC COMPLETE ==="
