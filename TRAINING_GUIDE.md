# Training Guide - Lessons Learned & Requirements

Everything needed to run diffusion-tinker training on a GPU cluster without debugging dependency hell.

---

## 1. Dependency Hell - Every Issue We Hit on RunPod

### Issue 1: PyTorch version incompatibility with diffusers
**Symptom:** `ValueError: infer_schema(func): Parameter q has unsupported type torch.Tensor`
**Cause:** diffusers 0.37+ uses `torch._custom_op` APIs that require PyTorch >= 2.5
**Fix:** Must install torch >= 2.5.1. RunPod template shipped 2.4.1.

### Issue 2: EasyOCR downgrades torch
**Symptom:** After `pip install easyocr`, torch goes back to 2.4.1
**Cause:** easyocr depends on `torch` without version pin, pip resolves to older cached version
**Fix:** Install easyocr with `--no-deps`, then install its deps separately

### Issue 3: PaddleOCR API changes
**Symptom:** `ValueError: Unknown argument: show_log` or `use_angle_cls`
**Cause:** PaddleOCR 3.x removed these params. Also PaddlePaddle has circular import bugs.
**Fix:** Don't use PaddleOCR at all. Use EasyOCR instead. Library already supports both.

### Issue 4: numpy ABI mismatch
**Symptom:** `RuntimeError: module compiled against ABI version 0x1000009 but this version of numpy is 0x2000000`
**Cause:** opencv-python compiled against old numpy, then numpy was upgraded
**Fix:** `pip install --force-reinstall numpy opencv-python-headless`

### Issue 5: CLIP get_image_features return type
**Symptom:** `TypeError: linalg_vector_norm(): argument 'input' must be Tensor, not BaseModelOutputWithPooling`
**Cause:** Newer transformers changed `get_image_features` to return an object, not tensor
**Fix:** Use `model.vision_model() + model.visual_projection()` directly instead

### Issue 6: PyTorch total_mem vs total_memory
**Symptom:** `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'`
**Cause:** PyTorch 2.4 uses `total_mem`, 2.5+ uses `total_memory`
**Fix:** Use `total_memory` (the newer name)

### Issue 7: Disk space
**Symptom:** `RuntimeError: No space left on device`
**Cause:** Model weights (~17GB) downloaded to container disk (20GB) instead of volume
**Fix:** Set HF_HOME, TORCH_HOME to the persistent volume before downloading

---

## 2. Exact Dependency Versions That Work

Tested on A100 80GB PCIe, Ubuntu 22.04, CUDA 12.4:

```
torch==2.5.1+cu124
torchvision==0.20.1+cu124
diffusers==0.37.1
transformers==4.51.3
peft==0.15.2
easyocr==1.7.2 (installed with --no-deps)
numpy==2.4.4
Pillow==12.2.0
safetensors==0.5.3
huggingface-hub==0.30.2
```

---

## 3. Environment Setup Script (Bulletproof)

This is the exact sequence that works. Do NOT change the order.

```bash
# Step 1: Set cache dirs FIRST (before any downloads)
export HF_HOME=/path/to/large/storage/.cache/huggingface
export TORCH_HOME=/path/to/large/storage/.cache/torch
export HF_TOKEN=your_token_here
mkdir -p $HF_HOME $TORCH_HOME

# Step 2: Clone and install base package
git clone https://github.com/srijitiyer/diffusion-tinker.git
cd diffusion-tinker
pip install -e . --quiet

# Step 3: Ensure correct torch version (MUST be 2.5+)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 --quiet

# Step 4: Install easyocr WITHOUT touching torch
pip install easyocr --no-deps --quiet
pip install pyclipper shapely python-bidi scikit-image opencv-python-headless --quiet

# Step 5: Fix numpy ABI
pip install --force-reinstall numpy --quiet

# Step 6: VERIFY before running anything expensive
python -c "
import torch
assert torch.cuda.is_available(), 'No GPU'
assert '2.5' in torch.__version__, f'Wrong torch: {torch.__version__}'
from diffusers import StableDiffusion3Pipeline
import easyocr
from diffusion_tinker import DDRLTrainer, DDRLConfig
print(f'ALL OK: torch={torch.__version__}, GPU={torch.cuda.get_device_name(0)}')
"
```

If the verification step fails, DO NOT proceed. Fix the issue first.

---

## 4. Model Requirements

### SD3.5-Medium
- HuggingFace ID: `stabilityai/stable-diffusion-3.5-medium`
- License: Must accept at https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
- Download size: ~17GB (transformer 5GB + 3 text encoders 12GB + VAE 300MB)
- VRAM usage with LoRA + gradient checkpointing: ~30GB
- Minimum GPU: A6000 (48GB) or A100 (40/80GB). A4000 (16GB) is too small.

### Storage
- Model cache: ~20GB
- EasyOCR models: ~100MB (downloads on first use)
- Training outputs (checkpoints, images): ~2GB per run
- Total: need ~25GB free storage minimum

---

## 5. Training Results So Far

### Aesthetic reward (30 epochs, 6 prompts, DDRL)
- Baseline: 6.04
- Final: 5.51
- Delta: -0.52 (reward went DOWN)
- Time: 52 minutes on A100 80GB
- VRAM peak: 29.3GB

### Why reward went down
The default DDRL hyperparameters were tuned for OCR reward in the paper:
- `data_beta=0.01` - may be wrong for aesthetic
- `clip_range=1e-4` - very tight clipping
- `use_monotonic_transform=True` - the -exp(-x) transform compresses advantages aggressively
- `learning_rate=1e-4` - may be too high for aesthetic

### What we haven't tried yet
- FlowGRPO trainer (different loss, no data_beta, different clip_range=0.2)
- Lower learning rate (1e-5)
- Disabling monotonic transform
- More epochs (100+)
- OCR reward (confirmed by Haotian as the right benchmark, but too slow with CPU-based EasyOCR)

---

## 6. OCR Reward Speed Issue

Each epoch with OCR reward:
- GPU phase: generate 80 images (~30 seconds)
- CPU phase: run EasyOCR on 80 images (~2 min, sequential, CPU-bound)
- Training step: ~2 min
- Total: ~5-10 min per epoch

With 150 epochs: 12-25 hours on a single GPU.

The DDRL paper used 24 GPUs with a separate async reward server. Options to speed up:
1. Run EasyOCR on GPU (`gpu=True`) - faster but takes VRAM
2. Reduce prompts from 20 to 8
3. Reduce num_samples_per_prompt from 4 to 2
4. Use fewer epochs (50 instead of 150)

---

## 7. Stanford SC Cluster Specifics

### Access
- SSH: `ssh srijit@scdt.stanford.edu` then `ssh sc`
- Account: `atlas` (Ermon group)
- Partition: `atlas`
- Home dir: `/atlas/u/srijit/` (group storage, ~TB scale)
- AFS home: `/afs/cs.stanford.edu/u/srijit/` (only 20MB, don't use)

### Available GPUs in atlas partition
| Nodes | GPUs | VRAM | Status |
|-------|------|------|--------|
| atlas29 | A6000 Ada x7 | 48GB | Best for SD3.5 |
| atlas[23-28] | A4000 x10 | 16GB | Too small for SD3.5 |
| atlas24 | A5000 x10 | 24GB | Marginal for SD3.5 |
| atlas[11-18] | Titan Xp x3-4 | 12GB | Too small |
| atlas[19-22] | 2080Ti x8 | 11GB | Too small |

### Important rules
- DO NOT run Claude/Cursor on sc/scdt (banner warning)
- DO NOT run heavy processes on the login node
- ALL training must go through SLURM (`sbatch`)
- Data transfer on scdt, not sc

### SLURM job template
```bash
#!/bin/bash
#SBATCH --job-name=diffusion-tinker
#SBATCH --partition=atlas
#SBATCH --nodelist=atlas29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/atlas/u/srijit/diffusion-tinker/logs/%j.out

# Environment
export HF_HOME=/atlas/u/srijit/.cache/huggingface
export TORCH_HOME=/atlas/u/srijit/.cache/torch
export HF_TOKEN=your_token

cd /atlas/u/srijit/diffusion-tinker

# Run training
python scripts/your_training_script.py
```

---

## 8. Pre-Flight Checklist (Before ANY Training Run)

1. [ ] HF_TOKEN set and SD3.5 license accepted
2. [ ] HF_HOME and TORCH_HOME point to large storage (not AFS)
3. [ ] torch >= 2.5.1 installed and verified
4. [ ] `from diffusers import StableDiffusion3Pipeline` works
5. [ ] `import easyocr` works (if using OCR reward)
6. [ ] `from diffusion_tinker import DDRLTrainer` works
7. [ ] GPU has >= 30GB VRAM free
8. [ ] Storage has >= 25GB free
9. [ ] Output directory exists and is writable
10. [ ] For SLURM: job script tested with a 5-minute test run first
