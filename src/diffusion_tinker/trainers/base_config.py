from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseDiffusionConfig:
    """Base configuration for all diffusion RL trainers."""

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64

    # Optimization
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-4

    # Training schedule
    num_epochs: int = 100
    save_every: int = 20
    eval_every: int = 10
    log_every: int = 1

    # Sampling / rollout
    num_samples_per_prompt: int = 4
    num_inference_steps: int = 28
    num_eval_inference_steps: int = 28
    guidance_scale: float = 7.0
    noise_level: float = 0.1
    resolution: int = 512

    # RL
    clip_range: float = 0.2
    adv_clip_max: float = 5.0
    # Train only on the first timestep_fraction of denoising steps (the higher-
    # sigma ones). The last low-sigma steps have std_dev_t -> 0, where the SDE
    # log-prob is ill-conditioned and the importance ratio is unreliable;
    # FlowGRPO drops them with timestep_fraction=0.99. 1.0 = train on all steps.
    timestep_fraction: float = 1.0
    # Normalize advantages by the global batch std instead of per-prompt std.
    # Helps low-variance rewards (e.g. aesthetic) where a tiny per-prompt std
    # otherwise amplifies sampling noise. Matches FlowGRPO's global_std option.
    advantage_global_std: bool = False

    # Model
    model_type: str = "auto"  # "auto", "sd3", or "flux"

    # Memory
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    # Offload the (frozen) text encoders to CPU during the gradient-tracked
    # training step, keeping them on GPU only for sampling/eval. Frees ~10GB
    # (T5-XXL) so more samples/larger replay batches fit on a 24GB card, at the
    # cost of a per-epoch CPU<->GPU transfer. Off by default.
    offload_text_encoders: bool = False

    # Output
    output_dir: str = "./output"
    seed: int = 42
    save_best: bool = True
    early_stop_patience: int = 0  # 0 = disabled, N = stop after N evals with no improvement
