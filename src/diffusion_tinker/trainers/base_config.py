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

    # Mini-batch size for the gradient-tracked replay/training pass. The rollout
    # collects a full batch (num_prompts * num_samples_per_prompt trajectories),
    # but replaying all of them through the transformer at once OOMs once the
    # group size is large. When set, the training step splits the batch into
    # chunks of this many trajectories and accumulates gradients across them,
    # so memory is bounded by train_batch_size instead of the full batch. This
    # is what lets you use many diverse prompts at a large group size (FlowGRPO's
    # actual setup) on a single GPU. None = replay the whole batch at once.
    train_batch_size: int | None = None

    # When train_prompts is a large pool, sample this many prompts per epoch
    # (a fresh random subset each epoch) instead of rolling out every prompt.
    # Lets you train on a big, diverse prompt set - which gives the reward room
    # to climb above the base model - without OOMing on num_prompts * group_size
    # trajectories per epoch. None = use all prompts every epoch.
    prompts_per_epoch: int | None = None

    # Sampling / rollout
    num_samples_per_prompt: int = 4
    num_inference_steps: int = 28
    num_eval_inference_steps: int = 28
    guidance_scale: float = 7.0
    # SDE exploration noise. For FlowGRPO-style RL with a KL anchor, use ~0.7
    # (the FlowGRPO value). Beware: the per-step KL penalty scales as
    # 1/noise_level**2 (its variance is std_dev_t**2 ~ noise_level**2), so a
    # small noise_level with kl_beta>0 silently over-weights the anchor and
    # freezes the policy at the reference - stable but the reward never moves.
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
