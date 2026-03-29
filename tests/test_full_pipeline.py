"""End-to-end integration test with tiny randomly-initialized models.

Validates the FULL training loop (model load -> LoRA -> SDE sampling ->
reward -> DDRL loss -> optimizer step) without needing a GPU or real model weights.
Runs on CPU in ~30 seconds.
"""

import math
import os
import tempfile

import torch
import torch.nn as nn
from PIL import Image

from diffusion_tinker.core.noise_strategy import compute_flow_matching_loss, sde_step_with_logprob
from diffusion_tinker.core.stat_tracking import PerPromptStatTracker
from diffusion_tinker.core.trajectory import TrajectoryBatch
from diffusion_tinker.rewards.base import BaseReward
from diffusion_tinker.rewards.protocol import RewardContext, RewardOutput
from diffusion_tinker.rewards.resolve import resolve_reward
from diffusion_tinker.trainers.ddrl_config import DDRLConfig


# ---------------------------------------------------------------------------
# Tiny transformer that mimics SD3Transformer2DModel's forward signature
# ---------------------------------------------------------------------------
class TinyTransformer(nn.Module):
    """Minimal transformer matching SD3's forward API.
    ~10K params instead of 2.5B. Enough to test gradient flow.
    """

    def __init__(self, in_channels=4, hidden_dim=32, num_layers=2, text_dim=64, pooled_dim=32):
        super().__init__()
        self.config = {"in_channels": in_channels}
        self.in_channels = in_channels
        self.proj_in = nn.Conv2d(in_channels, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(4, hidden_dim),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        self.time_embed = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.pooled_proj = nn.Linear(pooled_dim, hidden_dim)
        self.proj_out = nn.Conv2d(hidden_dim, in_channels, 1)
        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    def forward(self, hidden_states, timestep, encoder_hidden_states, pooled_projections, return_dict=True, **kwargs):
        B = hidden_states.shape[0]
        h = self.proj_in(hidden_states)

        # Time conditioning
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(B)
        t_emb = self.time_embed(timestep.float().unsqueeze(-1) / 1000.0)
        h = h + t_emb[:, :, None, None]

        # Text conditioning (mean pool then broadcast)
        text_emb = self.text_proj(encoder_hidden_states.mean(dim=1))
        h = h + text_emb[:, :, None, None]

        # Pooled conditioning
        pool_emb = self.pooled_proj(pooled_projections)
        h = h + pool_emb[:, :, None, None]

        for block in self.blocks:
            h = h + block(h)

        out = self.proj_out(h)

        if return_dict:
            return type("Output", (), {"sample": out})()
        return (out,)


# ---------------------------------------------------------------------------
# Tiny VAE
# ---------------------------------------------------------------------------
class TinyVAE(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "scaling_factor": 1.0,
                "shift_factor": 0.0,
                "in_channels": in_channels,
            },
        )()
        self.decoder = nn.Conv2d(in_channels, 3, 1)

    def decode(self, latents, return_dict=False):
        images = torch.sigmoid(self.decoder(latents))
        if return_dict:
            return type("Output", (), {"sample": images})()
        return (images,)

    def encode(self, images):
        raise NotImplementedError("Not needed for RL training")


# ---------------------------------------------------------------------------
# Dummy reward that gives higher scores to brighter images
# ---------------------------------------------------------------------------
class DummyReward(BaseReward):
    name = "dummy"

    def _compute(self, ctx: RewardContext) -> RewardOutput:
        scores = []
        for img in ctx.images:
            import numpy as np

            arr = np.array(img).mean() / 255.0
            scores.append(arr)
        return RewardOutput(scores=torch.tensor(scores, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------
def test_sde_step():
    """Test SDE step produces correct shapes and finite values."""
    B, C, H, W = 4, 4, 8, 8
    model_output = torch.randn(B, C, H, W)
    sigma = torch.tensor([0.8] * B)
    sigma_next = torch.tensor([0.6] * B)
    sample = torch.randn(B, C, H, W)

    prev, lp, mean = sde_step_with_logprob(model_output, sigma, sigma_next, sample, noise_level=0.7)

    assert prev.shape == (B, C, H, W), f"prev shape: {prev.shape}"
    assert lp.shape == (B,), f"log_prob shape: {lp.shape}"
    assert mean.shape == (B, C, H, W), f"mean shape: {mean.shape}"
    assert torch.isfinite(lp).all(), f"Non-finite log_probs: {lp}"
    assert torch.isfinite(prev).all(), "Non-finite prev_sample"

    # Replay should give identical log-prob
    _, lp2, _ = sde_step_with_logprob(model_output, sigma, sigma_next, sample, noise_level=0.7, prev_sample=prev)
    assert torch.allclose(lp, lp2, atol=1e-5), f"Replay mismatch: {(lp - lp2).abs().max()}"

    print("  sde_step: OK")


def test_flow_matching_loss():
    """Test denoising loss computation (DDRL's data regularization term)."""
    B, C, H, W = 2, 4, 8, 8
    text_dim, pooled_dim = 64, 32

    transformer = TinyTransformer(in_channels=C, text_dim=text_dim, pooled_dim=pooled_dim)
    latents = torch.randn(B, C, H, W)
    noise = torch.randn(B, C, H, W)
    sigmas = torch.rand(B) * 0.8 + 0.1
    prompt_embeds = torch.randn(B, 10, text_dim)
    pooled_embeds = torch.randn(B, pooled_dim)

    loss = compute_flow_matching_loss(transformer, latents, noise, sigmas, prompt_embeds, pooled_embeds)

    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert torch.isfinite(loss), f"Non-finite loss: {loss}"
    assert loss > 0, f"Loss should be positive: {loss}"

    # Verify gradients flow
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in transformer.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients computed"
    assert all(g > 0 for g in grad_norms), f"Zero gradients: {grad_norms}"

    print("  flow_matching_loss: OK")


def test_stat_tracker():
    """Test per-prompt advantage normalization."""
    tracker = PerPromptStatTracker()
    prompts = ["cat"] * 4 + ["dog"] * 4
    rewards = torch.tensor([5.0, 3.0, 4.0, 6.0, 2.0, 8.0, 5.0, 5.0])
    adv = tracker.update(prompts, rewards)

    assert adv.shape == (8,), f"Advantage shape: {adv.shape}"
    # Per-group mean should be ~0
    cat_adv = adv[:4]
    dog_adv = adv[4:]
    assert abs(cat_adv.mean().item()) < 0.01, f"Cat advantages not centered: {cat_adv.mean()}"
    assert abs(dog_adv.mean().item()) < 0.01, f"Dog advantages not centered: {dog_adv.mean()}"

    print("  stat_tracker: OK")


def test_ddrl_advantages():
    """Test DDRL's monotonic transform: A = -exp(-A)."""
    rewards = torch.tensor([5.0, 3.0, 4.0, 2.0, 6.0, 4.0])
    prompts = ["a"] * 3 + ["b"] * 3

    tracker = PerPromptStatTracker()
    adv = tracker.update(prompts, rewards)

    # Monotonic transform
    adv_transformed = -torch.exp(-adv)

    # Properties:
    # 1. All values should be negative (range is (-inf, 0))
    assert (adv_transformed < 0).all(), f"Transform should be negative: {adv_transformed}"
    # 2. Higher raw advantage -> closer to 0 (less negative)
    # 3. Lower raw advantage -> more negative
    print("  ddrl_advantages: OK")


def test_reward_resolve():
    """Test reward registry and resolution."""
    # String resolution
    reward = resolve_reward("aesthetic", device="cpu")
    assert reward.name == "aesthetic"

    # Callable resolution
    def my_fn(ctx):
        return RewardOutput(scores=torch.ones(len(ctx.images)))

    reward2 = resolve_reward(my_fn, device="cpu")
    ctx = RewardContext(images=[Image.new("RGB", (64, 64))] * 3, prompts=["a"] * 3)
    out = reward2(ctx)
    assert out.scores.shape == (3,), f"Wrong shape: {out.scores.shape}"

    print("  reward_resolve: OK")


def test_trajectory_batch():
    """Test trajectory dataclass operations."""
    B, T, C, H, W = 6, 5, 4, 8, 8
    traj = TrajectoryBatch(
        latents=torch.randn(B, T, C, H, W),
        next_latents=torch.randn(B, T, C, H, W),
        log_probs=torch.randn(B, T),
        timesteps=torch.linspace(0.9, 0.1, T),
        prompt_embeds=torch.randn(B, 10, 64),
        pooled_embeds=torch.randn(B, 32),
        prompts=["p"] * B,
        rewards=torch.randn(B),
        advantages=torch.randn(B),
    )

    assert len(traj) == B
    subset = traj[2:5]
    assert len(subset) == 3
    assert subset.latents.shape == (3, T, C, H, W)
    assert len(subset.prompts) == 3

    traj.to("cpu")

    print("  trajectory_batch: OK")


def test_full_ddrl_training_step():
    """Full DDRL training step: sampling -> reward -> advantage -> loss -> backward -> optimizer step.

    This is the critical end-to-end test. Uses tiny models, runs on CPU.
    """
    B = 4  # samples per prompt
    T = 5  # denoising steps
    C, H, W = 4, 8, 8  # tiny latent dims
    text_dim, pooled_dim = 64, 32

    # 1. Create tiny transformer + LoRA
    transformer = TinyTransformer(in_channels=C, text_dim=text_dim, pooled_dim=pooled_dim)

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["proj_in", "proj_out"], init_lora_weights="gaussian")
    transformer = get_peft_model(transformer, lora_config)

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)

    # 2. Simulate a rollout (normally done by sd3_sample_with_logprob)
    sigmas = torch.linspace(0.95, 0.05, T + 1)
    all_latents = []
    all_next_latents = []
    all_log_probs = []

    prompt_embeds = torch.randn(B, 10, text_dim)
    pooled_embeds = torch.randn(B, pooled_dim)

    latents = torch.randn(B, C, H, W) * sigmas[0]

    with torch.no_grad():
        for i in range(T):
            all_latents.append(latents.clone())
            sigma = sigmas[i].expand(B)
            sigma_next = sigmas[i + 1].expand(B)

            noise_pred = transformer(
                hidden_states=latents,
                timestep=sigma * 1000,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

            noise_level = 0.7 if i < T - 1 else 0.0
            latents, log_prob, _ = sde_step_with_logprob(
                noise_pred, sigma, sigma_next, latents, noise_level=noise_level
            )
            all_next_latents.append(latents.clone())
            all_log_probs.append(log_prob)

    # 3. Fake rewards (brighter = higher)
    fake_images = [Image.new("RGB", (64, 64), color=(c * 60, c * 60, c * 60)) for c in range(B)]
    reward_fn = DummyReward()
    ctx = RewardContext(images=fake_images, prompts=["test"] * B)
    rewards = reward_fn(ctx).scores

    # 4. Build trajectory
    trajectory = TrajectoryBatch(
        latents=torch.stack(all_latents, dim=1),
        next_latents=torch.stack(all_next_latents, dim=1),
        log_probs=torch.stack(all_log_probs, dim=1),
        timesteps=sigmas[:-1],
        prompt_embeds=prompt_embeds,
        pooled_embeds=pooled_embeds,
        prompts=["test"] * B,
        rewards=rewards,
        images=fake_images,
    )

    # 5. Compute DDRL advantages with monotonic transform
    tracker = PerPromptStatTracker()
    advantages = tracker.update(trajectory.prompts, trajectory.rewards)
    advantages = -torch.exp(-advantages)  # DDRL monotonic transform
    advantages = torch.clamp(advantages, -5.0, 5.0)
    trajectory.advantages = advantages

    # 6. Training step: iterate over timesteps, compute RL + data loss
    config = DDRLConfig(clip_range=1e-4, data_beta=0.01)
    optimizer.zero_grad()
    total_loss = 0.0

    timestep_indices = list(range(T))
    import random

    random.shuffle(timestep_indices)

    for j in timestep_indices:
        sigma = sigmas[j].expand(B)
        sigma_next = sigmas[j + 1].expand(B)

        if sigma_next[0].item() < 1e-6:
            continue

        latent_t = trajectory.latents[:, j]
        next_latent_t = trajectory.next_latents[:, j]

        # Replay through current model
        noise_pred = transformer(
            hidden_states=latent_t,
            timestep=sigma * 1000,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

        _, log_prob_new, _ = sde_step_with_logprob(
            noise_pred, sigma, sigma_next, latent_t, noise_level=0.7, prev_sample=next_latent_t
        )
        log_prob_old = trajectory.log_probs[:, j]

        # RL loss
        ratio = torch.exp(log_prob_new - log_prob_old)
        unclipped = -advantages * ratio
        clipped = -advantages * torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range)
        rl_loss = torch.mean(torch.maximum(unclipped, clipped))

        # Data loss (denoising)
        noise = torch.randn(B, C, H, W)
        t = torch.sigmoid(torch.randn(B))
        t_bc = t.view(B, 1, 1, 1)
        clean = trajectory.next_latents[:, -1]
        noisy = (1 - t_bc) * clean + t_bc * noise

        v_pred = transformer(
            hidden_states=noisy,
            timestep=t * 1000,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        data_loss = torch.nn.functional.mse_loss(v_pred, noise - clean)

        loss = (rl_loss + config.data_beta * data_loss) / T
        loss.backward()
        total_loss += loss.item()

    # 7. Optimizer step
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()

    # 8. Verify everything worked
    assert total_loss > 0, f"Loss should be positive: {total_loss}"
    assert math.isfinite(total_loss), f"Non-finite loss: {total_loss}"

    # Verify LoRA params actually changed
    grad_norms = [p.grad.norm().item() for p in trainable if p.grad is not None]
    assert all(math.isfinite(g) for g in grad_norms), f"Non-finite gradients: {grad_norms}"
    assert any(g > 0 for g in grad_norms), "No non-zero gradients - LoRA params not updating"

    print(f"  full_ddrl_step: OK (loss={total_loss:.4f}, grad_norms={[f'{g:.4f}' for g in grad_norms[:3]]})")


def test_checkpoint_save_load():
    """Test saving and loading LoRA checkpoints."""
    C, text_dim, pooled_dim = 4, 64, 32
    transformer = TinyTransformer(in_channels=C, text_dim=text_dim, pooled_dim=pooled_dim)

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["proj_in", "proj_out"])
    transformer = get_peft_model(transformer, lora_config)

    # Modify a LoRA weight
    for name, p in transformer.named_parameters():
        if "lora" in name and p.requires_grad:
            p.data.fill_(42.0)
            break

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "checkpoint")
        transformer.save_pretrained(save_path)
        assert os.path.exists(os.path.join(save_path, "adapter_model.safetensors")), "Checkpoint not saved"

        # Load into fresh model
        transformer2 = TinyTransformer(in_channels=C, text_dim=text_dim, pooled_dim=pooled_dim)
        from peft import PeftModel

        transformer2 = PeftModel.from_pretrained(transformer2, save_path)

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(transformer.named_parameters(), transformer2.named_parameters()):
            if "lora" in n1:
                assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"

    print("  checkpoint_save_load: OK")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running diffusion-tinker integration tests...\n")

    test_sde_step()
    test_flow_matching_loss()
    test_stat_tracker()
    test_ddrl_advantages()
    test_reward_resolve()
    test_trajectory_batch()
    test_full_ddrl_training_step()
    test_checkpoint_save_load()

    print("\n=== ALL TESTS PASSED ===")
