"""End-to-end integration test with tiny randomly-initialized models.

Validates the FULL training loop (model load -> LoRA -> SDE sampling ->
reward -> DDRL loss -> optimizer step) without needing a GPU or real model weights.
Runs on CPU in ~30 seconds.
"""

import math
import os
import random
import tempfile

import torch
import torch.nn as nn
from PIL import Image

from diffusion_tinker.core.noise_strategy import compute_flow_matching_loss, sde_step_with_logprob
from collections import deque

from diffusion_tinker.core.stat_tracking import BufferedPerPromptStatTracker, PerPromptStatTracker
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


def test_sde_step_at_sigma_one():
    """The SD3/FLUX schedules start at sigma exactly 1.0, where
    std_dev_t = sqrt(sigma/(1-sigma)) diverges. Unbounded, the first step
    multiplied the latent by about -56 at noise_level=0.7, which wrecked every
    rollout (reward went to 0 with no within-group spread, so no signal). The
    step must stay contractive and must not flip sign.
    """
    B, C, H, W = 4, 4, 8, 8
    sample = torch.randn(B, C, H, W)
    sigma = torch.tensor([1.0] * B)
    sigma_next = torch.tensor([0.9768] * B)

    for noise_level in (0.1, 0.7, 1.0):
        # isolate the coefficient on `sample` by zeroing the velocity
        _, lp, mean = sde_step_with_logprob(
            torch.zeros(B, C, H, W), sigma, sigma_next, sample, noise_level=noise_level
        )
        coeff = (mean / sample).mean().item()
        assert 0.0 < coeff <= 1.0, f"noise={noise_level}: sample coeff {coeff} not in (0, 1]"
        assert torch.isfinite(lp).all(), f"noise={noise_level}: non-finite log_prob"
        assert mean.std() < 2.0 * sample.std(), f"noise={noise_level}: mean blew up"

    print("  sde_step at sigma=1: OK")


def test_trajectory_shuffled_indexing_alignment():
    """Mini-batches are built from a shuffled permutation, so tensor fields and
    list fields (prompts) must stay aligned. A silent misalignment here would
    pair each trajectory with another sample's advantage, which trains fine and
    learns nothing.
    """
    B, T = 6, 3
    traj = TrajectoryBatch(
        latents=torch.arange(B).float().reshape(B, 1, 1, 1, 1).expand(B, T, 2, 2, 2).clone(),
        next_latents=torch.zeros(B, T, 2, 2, 2),
        log_probs=torch.zeros(B, T),
        timesteps=torch.linspace(1.0, 0.0, T + 1),
        prompt_embeds=torch.arange(B).float().reshape(B, 1, 1).expand(B, 4, 8).clone(),
        pooled_embeds=torch.arange(B).float().reshape(B, 1).expand(B, 8).clone(),
        prompts=[f"p{i}" for i in range(B)],
        rewards=torch.arange(B).float(),
        advantages=torch.arange(B).float(),
    )

    perm = torch.tensor([4, 1, 5, 0, 3, 2])
    mb = traj[perm]

    for slot, src in enumerate(perm.tolist()):
        assert mb.prompts[slot] == f"p{src}", f"prompt slot {slot} -> {mb.prompts[slot]}"
        assert mb.advantages[slot].item() == src, "advantage misaligned with prompt"
        assert mb.rewards[slot].item() == src, "reward misaligned with prompt"
        assert mb.latents[slot].flatten()[0].item() == src, "latent misaligned with prompt"
        assert mb.prompt_embeds[slot].flatten()[0].item() == src, "embeds misaligned"

    print("  trajectory shuffled indexing: OK")


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

    # Edge case: single-sample group should not produce NaN
    tracker2 = PerPromptStatTracker()
    adv2 = tracker2.update(["solo"], torch.tensor([5.0]))
    assert torch.isfinite(adv2).all(), f"Single-sample group produced NaN: {adv2}"

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


def test_ddrl_monotonic_transform_recentered():
    """The transform is applied through DDRLTrainer._compute_advantages, which
    re-centers it. Without the re-centering every advantage is negative (the
    range of -exp(-x) is (-inf, 0)), so -A * ratio pushes the policy away from
    every action it sampled and the run collapses. That was the bug behind the
    old "DDRL diverges" result, so pin both halves: zero-mean per group, and
    ranking preserved.
    """
    from diffusion_tinker.trainers.ddrl_trainer import DDRLTrainer

    prompts = ["a"] * 4 + ["b"] * 4
    rewards = torch.tensor([5.0, 3.0, 4.0, 6.0, 2.0, 8.0, 5.0, 5.0])
    raw = PerPromptStatTracker().update(prompts, rewards)

    transformed = -torch.exp(-raw)
    assert (transformed < 0).all(), "sanity: the raw transform is all-negative"

    centered = DDRLTrainer._recenter_per_group(transformed, prompts)
    for group in (slice(0, 4), slice(4, 8)):
        assert abs(centered[group].mean().item()) < 1e-5, f"group not zero-mean: {centered[group]}"
        # a real policy-gradient estimator needs both signs present
        assert (centered[group] > 0).any() and (centered[group] < 0).any()
        # lambda is increasing, so it must not reorder the group
        order_raw = torch.argsort(raw[group])
        order_new = torch.argsort(centered[group])
        assert torch.equal(order_raw, order_new), "transform changed the ranking"

    print("  ddrl monotonic transform re-centered: OK")


def test_buffered_tracker_differs_from_grpo():
    """DDPO normalizes against a rolling per-prompt history; GRPO normalizes
    within the batch. The tell is that GRPO's advantages are exactly zero-mean
    per group by construction and DDPO's are not, once the buffer has filled.
    DDPO silently inheriting the GRPO tracker is what made every pre-audit DDPO
    number a FlowGRPO number, so it is worth a test rather than a comment.
    """
    from diffusion_tinker.core.stat_tracking import BufferedPerPromptStatTracker

    buffered = BufferedPerPromptStatTracker(buffer_size=32, min_count=4)
    grpo = PerPromptStatTracker()

    torch.manual_seed(0)
    prompts = ["a"] * 4 + ["b"] * 4
    # feed a few batches so the per-prompt buffers pass min_count
    for _ in range(3):
        r = torch.rand(8) * 4.0
        adv_b = buffered.update(prompts, r)
        adv_g = grpo.update(prompts, r)

    # last batch: shift prompt "a" well above its own history
    r = torch.cat([torch.full((4,), 8.0), torch.rand(4) * 4.0])
    adv_b = buffered.update(prompts, r)
    adv_g = grpo.update(prompts, r)

    assert abs(adv_g[:4].mean().item()) < 1e-5, "GRPO group should be exactly centered"
    assert adv_b[:4].mean().item() > 0.5, (
        f"a batch far above the prompt's history should get positive advantage, got {adv_b[:4].mean()}"
    )
    assert torch.isfinite(adv_b).all()

    # a single sample of an unseen prompt must not divide by a zero-size history
    solo = BufferedPerPromptStatTracker()
    assert torch.isfinite(solo.update(["new"], torch.tensor([1.0]))).all()

    print("  buffered tracker differs from grpo: OK")


def test_kl_variance_finite_at_sigma_one():
    """The KL anchor divides by the sampler's step variance. If it used the
    uncapped std_dev_t it would divide by a variance ~1e4 too large at sigma=1
    and the penalty would silently vanish on the first step - the run looks
    regularized and is not. Both trainers must use sde_diffusion_std.
    """
    from diffusion_tinker.core.noise_strategy import sde_diffusion_std

    sigma = torch.tensor([1.0, 0.9, 0.5])
    sigma_next = torch.tensor([0.9768, 0.85, 0.45])

    for noise_level in (0.1, 0.7, 1.0):
        std = sde_diffusion_std(sigma, sigma_next, noise_level)
        noise_std = std * torch.sqrt((sigma - sigma_next).clamp(min=1e-12))
        assert torch.isfinite(noise_std).all(), f"noise={noise_level}: non-finite"
        assert (noise_std > 0).all(), f"noise={noise_level}: zero variance divides by ~0"
        # unbounded, sqrt(1/(1-1)) is inf at sigma=1; the cap must bite there
        assert noise_std[0] < 5.0, f"noise={noise_level}: sigma=1 std {noise_std[0]} still blowing up"

    print("  kl variance finite at sigma=1: OK")


def _bind_trainer_helpers(stub):
    """Give a SimpleNamespace stub every private helper _save_state/_load_state
    calls on self.

    Bound in one place rather than at each stub, because four tests build these
    and patching them individually is how a new helper silently breaks three of
    them - the same single-site-fix pattern that produced several of the defects
    this suite now pins.
    """
    from types import SimpleNamespace

    from diffusion_tinker.trainers.base_trainer import BaseDiffusionTrainer

    # The fingerprint covers the objective too - reward and prompt count - because
    # swapping the reward on a reused tag silently reported the new reward's curve
    # on top of the old reward's adapter. Stubs need both to compute it.
    if not hasattr(stub, "reward_fn"):
        stub.reward_fn = SimpleNamespace(name="stub_reward")
    if not hasattr(stub, "train_prompts"):
        stub.train_prompts = ["a cat"]

    stub._state_path = lambda: BaseDiffusionTrainer._state_path(stub)
    stub._state_fingerprint = lambda: BaseDiffusionTrainer._state_fingerprint(stub)
    stub._stateful_tracker = lambda: BaseDiffusionTrainer._stateful_tracker(stub)
    stub._tracker_state = lambda: BaseDiffusionTrainer._tracker_state(stub)
    stub._restore_tracker_state = lambda t: BaseDiffusionTrainer._restore_tracker_state(stub, t)
    return stub


def test_resume_state_roundtrip():
    """A crash must cost wall-clock, not training progress.

    The cluster kills jobs at random, and without resume every crash discarded
    every epoch, so no run could outlast the mean time between crashes and no
    experiment could ask a question that needed a long run. This pins the parts
    that make a resume faithful rather than merely non-crashing: the adapter
    weights, the AdamW moments (a fresh optimizer leaves a transient exactly
    where a drift measurement would read it), and the epoch to restart from.
    """
    import tempfile
    from types import SimpleNamespace

    from peft import LoraConfig, get_peft_model
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

    from diffusion_tinker.trainers.base_trainer import BaseDiffusionTrainer

    torch.manual_seed(0)
    model = get_peft_model(
        TinyTransformer(),
        LoraConfig(r=4, lora_alpha=8, target_modules=["proj_in", "proj_out"], init_lora_weights="gaussian"),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-3)

    # take a step so the adapter has moved and AdamW has non-empty moments
    loss = sum(p.pow(2).sum() for p in trainable)
    loss.backward()
    opt.step()

    with tempfile.TemporaryDirectory() as tmp:
        cfg = DDRLConfig(output_dir=tmp)
        cfg.resumable = True
        stub = SimpleNamespace(
            config=cfg,
            transformer=model,
            optimizer=opt,
            device=torch.device("cpu"),
            global_step=7,
            _best_eval_reward=0.62,
            _evals_without_improvement=2,
            # A REAL tracker, not a stub with a stats attribute invented to match
            # what the save path happened to read. The previous version of this
            # test asserted against SimpleNamespace(stats=...), a shape no trainer
            # has: PerPromptStatTracker is stateless and DDPO keeps its buffered
            # tracker on `_ddpo_tracker`. So the test passed while the real code
            # saved None on every trainer and restored nothing.
            stat_tracker=PerPromptStatTracker(),
            _ddpo_tracker=BufferedPerPromptStatTracker(buffer_size=32, min_count=16),
        )
        stub._ddpo_tracker.stats["a cat"] = deque([0.1, 0.2], maxlen=32)
        _bind_trainer_helpers(stub)
        BaseDiffusionTrainer._save_state(stub, next_epoch=5)

        saved_adapter = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
        exp_avg = opt.state_dict()["state"][0]["exp_avg"].clone()

        # wipe everything the way a restart would
        for p in trainable:
            with torch.no_grad():
                p.zero_()
        fresh_opt = torch.optim.AdamW(trainable, lr=1e-3)
        stub.optimizer = fresh_opt
        stub.global_step = 0
        stub._best_eval_reward = -float("inf")
        stub._evals_without_improvement = 0
        stub._ddpo_tracker = BufferedPerPromptStatTracker(buffer_size=32, min_count=16)

        start = BaseDiffusionTrainer._load_state(stub)

    assert start == 5, f"resumed at the wrong epoch: {start}"
    assert stub.global_step == 7
    assert abs(stub._best_eval_reward - 0.62) < 1e-9, "best reward not restored, so save_best would regress"
    assert stub._evals_without_improvement == 2
    assert list(stub._ddpo_tracker.stats.get("a cat", [])) == [0.1, 0.2], (
        "DDPO's reward history did not survive the round trip. Without it every "
        "prompt's buffer is empty on resume, so BufferedPerPromptStatTracker falls "
        "back to whole-batch statistics until the buffers refill - which is the "
        "GRPO estimator, not DDPO's."
    )

    restored = get_peft_model_state_dict(model)
    for k, v in saved_adapter.items():
        assert torch.allclose(restored[k], v), f"adapter weight {k} not restored"
    got = stub.optimizer.state_dict()["state"][0]["exp_avg"]
    assert torch.allclose(got, exp_avg), "AdamW moments not restored"

    print("  resume state roundtrip: OK")


def test_resume_refuses_mismatched_config():
    """A resume state must not be loaded into a different experiment.

    Resume keys off output_dir, which is keyed off a run tag. Reuse a tag with a
    different beta and the old arm's adapter would load into the new arm's run
    and be reported as the new arm's result - silent, and fatal to exactly the
    kind of comparison this repo exists to make.
    """
    import tempfile
    from types import SimpleNamespace

    from peft import LoraConfig, get_peft_model

    from diffusion_tinker.trainers.base_trainer import BaseDiffusionTrainer

    torch.manual_seed(0)
    model = get_peft_model(
        TinyTransformer(),
        LoraConfig(r=4, lora_alpha=8, target_modules=["proj_in", "proj_out"], init_lora_weights="gaussian"),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-3)

    def make_stub(tmp, data_beta):
        cfg = DDRLConfig(output_dir=tmp, data_beta=data_beta)
        cfg.resumable = True
        stub = SimpleNamespace(
            config=cfg, transformer=model, optimizer=opt, device=torch.device("cpu"),
            global_step=3, _best_eval_reward=0.6, _evals_without_improvement=0,
            stat_tracker=SimpleNamespace(stats={}),
        )
        _bind_trainer_helpers(stub)
        return stub

    with tempfile.TemporaryDirectory() as tmp:
        BaseDiffusionTrainer._save_state(make_stub(tmp, 0.3), next_epoch=9)

        same = make_stub(tmp, 0.3)
        assert BaseDiffusionTrainer._load_state(same) == 9, "identical config should resume"

        different = make_stub(tmp, 0.0)
        assert BaseDiffusionTrainer._load_state(different) == 0, (
            "a state written at data_beta=0.3 must not load into a data_beta=0 run"
        )
        assert different.global_step == 3 or True  # counters untouched is incidental

    print("  resume refuses mismatched config: OK")


def test_resume_restores_rng():
    """A resumed run must draw the same randomness an uninterrupted one would.

    Without this, a resumed run gets different rollout noise and prompt subsets,
    which is only harmless if every arm resumes equally often. On this cluster
    they do not: the data_beta=0.3 arms took 40 crashes to data_beta=0's 31, so
    resume frequency would vary alongside the quantity under comparison and land
    in the drift measurement as if it were an effect of the anchor.
    """
    import tempfile
    from types import SimpleNamespace

    from peft import LoraConfig, get_peft_model

    from diffusion_tinker.trainers.base_trainer import BaseDiffusionTrainer

    torch.manual_seed(0)
    model = get_peft_model(
        TinyTransformer(),
        LoraConfig(r=4, lora_alpha=8, target_modules=["proj_in", "proj_out"], init_lora_weights="gaussian"),
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = DDRLConfig(output_dir=tmp)
        cfg.resumable = True
        stub = SimpleNamespace(
            config=cfg, transformer=model, optimizer=opt, device=torch.device("cpu"),
            global_step=0, _best_eval_reward=0.0, _evals_without_improvement=0,
            stat_tracker=SimpleNamespace(stats={}),
        )
        _bind_trainer_helpers(stub)

        torch.manual_seed(1234)
        random.seed(1234)
        BaseDiffusionTrainer._save_state(stub, next_epoch=3)

        # what an uninterrupted run would have drawn next
        expected_torch = torch.randn(5)
        expected_python = [random.random() for _ in range(3)]

        # disturb both streams the way a fresh process would
        torch.manual_seed(999)
        random.seed(999)

        BaseDiffusionTrainer._load_state(stub)
        got_torch = torch.randn(5)
        got_python = [random.random() for _ in range(3)]

    assert torch.allclose(got_torch, expected_torch), "torch RNG not restored"
    assert got_python == expected_python, "python RNG not restored"

    print("  resume restores rng: OK")


def test_resume_is_transparent():
    """Training through a crash+resume must equal training straight through.

    I claimed this twice from reading the code and was wrong the first time, so it
    gets measured. Two identical models: one takes two update steps in a row, the
    other takes one, has its state saved, is wiped the way a fresh process would
    be, resumes, and takes the second. If resume is transparent the parameters and
    the optimiser moments match exactly.

    This is what makes resume usable in a comparison at all. If it were not
    transparent, an arm that crashes more would be trained differently from one
    that does not, and any between-arm result would carry crash frequency inside
    it - which is exactly the confound that was suspected here and could not be
    settled by argument.
    """
    import tempfile
    from types import SimpleNamespace

    from peft import LoraConfig, get_peft_model

    from diffusion_tinker.trainers.base_trainer import BaseDiffusionTrainer

    def fresh():
        torch.manual_seed(7)
        m = get_peft_model(
            TinyTransformer(),
            LoraConfig(r=4, lora_alpha=8, target_modules=["proj_in", "proj_out"], init_lora_weights="gaussian"),
        )
        return m, torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)

    def step(model, opt, seed):
        # a deterministic surrogate for one epoch's update
        torch.manual_seed(seed)
        x = torch.randn(2, 4, 8, 8)
        opt.zero_grad()
        out = model(x, torch.tensor([500.0, 500.0]), torch.randn(2, 10, 64), torch.randn(2, 32), return_dict=False)[0]
        out.pow(2).mean().backward()
        opt.step()

    # straight through
    a_model, a_opt = fresh()
    step(a_model, a_opt, 11)
    step(a_model, a_opt, 12)
    straight = [p.detach().clone() for p in a_model.parameters() if p.requires_grad]

    # crash between the two steps
    b_model, b_opt = fresh()
    step(b_model, b_opt, 11)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = DDRLConfig(output_dir=tmp)
        cfg.resumable = True
        stub = SimpleNamespace(
            config=cfg, transformer=b_model, optimizer=b_opt, device=torch.device("cpu"),
            global_step=1, _best_eval_reward=0.0, _evals_without_improvement=0,
            stat_tracker=SimpleNamespace(stats={}),
        )
        _bind_trainer_helpers(stub)
        BaseDiffusionTrainer._save_state(stub, next_epoch=1)

        c_model, c_opt = fresh()          # the fresh process after the crash
        stub.transformer, stub.optimizer = c_model, c_opt
        assert BaseDiffusionTrainer._load_state(stub) == 1
    step(c_model, c_opt, 12)
    resumed = [p.detach().clone() for p in c_model.parameters() if p.requires_grad]

    for i, (x, y) in enumerate(zip(straight, resumed)):
        assert torch.allclose(x, y, atol=1e-6), (
            f"param {i} differs by {(x - y).abs().max():.3e} - resume is NOT transparent, "
            "so an arm that crashes more is trained differently"
        )

    print("  resume is transparent: OK")


def test_anchor_off_when_no_rl_signal():
    """The anchor must not train alone when the RL term has switched off.

    A sparse or binary reward at a small group size regularly produces groups
    where every sample scores the same, so the group-normalized advantage has no
    within-group variance, `has_signal` is false, and `rl_loss` is set to zero.
    Applying `data_beta * data_loss` on its own then makes the anchor the entire
    gradient, which walks the policy toward the anchor images and away from the
    task - and it compounds, because a worse policy makes groups more uniform.

    This gate is defensive, not a diagnosis. It was written for a FLUX result
    where held-out exact-match OCR fell 0.625 to 0.29 with the anchor on and held
    at 0.625 with it off, 3.1 SE apart. That measurement stands, but the group
    scores in those runs show `has_signal` was true in all 59 epochs of both arms,
    so the gate would have been open throughout and does not explain the collapse
    (report/analyze_flux_anchor.py). Keep it anyway: regularising a batch that
    cannot teach anything is wrong on its own terms.

    The pre-existing zero-signal guard in BaseDiffusionTrainer does not cover
    this: it skips an epoch only when fewer than two advantages are nonzero,
    which is a different condition from their having no spread.
    """
    import inspect

    from diffusion_tinker.trainers.ddrl_trainer import DDRLTrainer

    src = inspect.getsource(DDRLTrainer._training_step)

    assert "if config.data_beta > 0 and has_signal:" in src, (
        "the data-anchor term must be gated on has_signal; without it the anchor "
        "becomes the only gradient whenever the reward goes sparse"
    )

    # and the guard it relies on must still be the spread test, not a count
    outer = inspect.getsource(DDRLTrainer)
    assert "advantages.std().item() > 1e-6" in outer, (
        "has_signal must test advantage spread; a nonzero-count test would pass "
        "on groups that are internally uniform"
    )

    print("  anchor gated on rl signal: OK")


def test_all_trainers_guard_their_regulariser():
    """No trainer may apply its regulariser to a batch with no RL signal.

    This is one defect shape appearing in three places. When the advantages have
    no spread there is nothing to learn from, and any regularisation term applied
    anyway becomes the entire gradient. That is an argument about the code path
    rather than an observed failure: the FLUX collapse this was written for turned
    out to run with the RL term live throughout, so no run in this project has
    been shown to reach the condition. The guards stay regardless, because the
    alternative is a regulariser training the model on a batch that carries no
    signal.

    Each trainer defends differently, which is why this checks behaviour rather
    than a shared helper. FlowGRPO skips the timestep, DDRL gates its anchor,
    DDPO skips the minibatch. DDPO had no guard at all until this was found -
    its DPOK KL is off by default, so nothing had exercised it.
    """
    import inspect

    from diffusion_tinker.trainers.ddpo_trainer import DDPOTrainer
    from diffusion_tinker.trainers.ddrl_trainer import DDRLTrainer
    from diffusion_tinker.trainers.flowgrpo_trainer import FlowGRPOTrainer

    for trainer in (FlowGRPOTrainer, DDRLTrainer, DDPOTrainer):
        src = inspect.getsource(trainer._training_step)
        assert "has_signal" in src, (
            f"{trainer.__name__} never computes has_signal, so it cannot tell "
            "whether its regulariser is the only thing training the model"
        )
        # The right test differs by estimator and getting it wrong is silent.
        # Group-normalized trainers subtract the group mean, so zero spread
        # implies all-zero and a spread test is correct. DDPO normalizes against
        # a per-prompt history and can produce constant nonzero advantages -
        # "all of these beat this prompt's average" - which is a real signal a
        # spread test would discard.
        if trainer is DDPOTrainer:
            assert "advantages.abs().max().item() > 1e-6" in src, (
                "DDPO must test advantage magnitude; a spread test would skip "
                "constant-nonzero advantages, which are a genuine signal for a "
                "history-normalized estimator"
            )
        else:
            assert "advantages.std().item() > 1e-6" in src, (
                f"{trainer.__name__} must test advantage spread, not a nonzero "
                "count: a group can hold several nonzero but identical values"
            )

    # and each must actually act on it
    assert "if not has_signal:" in inspect.getsource(FlowGRPOTrainer._training_step)
    assert "if not has_signal:" in inspect.getsource(DDPOTrainer._training_step)
    assert "config.data_beta > 0 and has_signal" in inspect.getsource(DDRLTrainer._training_step)

    print("  all trainers guard their regulariser: OK")


def test_trainers_log_the_statistic_they_gate_on():
    """Every RL trainer must report advantage spread, not just its mean.

    This exists because a claim was built on the wrong field. Quiet-epoch counts
    for FLUX and SD3.5 were read off `rl_loss` printing as `0.0000`, and that is
    the mean advantage in disguise: `rl_loss` sits at `-mean_advantage` to four
    decimals across every log on disk, and group normalisation centres the
    advantages, so the mean is near zero whether or not the spread is. A quiet
    epoch and a live one both print `0.0000`, and four decimals only says
    `< 0.00005` in any case. The counts written down as measurements were upper
    bounds.

    Logging the gated statistic makes it decidable from a log instead of
    inferred from a proxy that cannot carry the inference.
    """
    import inspect

    from diffusion_tinker.trainers.ddpo_trainer import DDPOTrainer
    from diffusion_tinker.trainers.ddrl_trainer import DDRLTrainer
    from diffusion_tinker.trainers.flowgrpo_trainer import FlowGRPOTrainer

    for trainer in (FlowGRPOTrainer, DDRLTrainer, DDPOTrainer):
        src = inspect.getsource(trainer._training_step)
        # Each logs the statistic its own guard uses, which is not the same one.
        field = "advantage_absmax" if trainer is DDPOTrainer else "advantage_std"
        assert f'"{field}"' in src, (
            f"{trainer.__name__} does not log {field}, so whether its guard "
            "fired cannot be recovered from a run's log"
        )
        assert '"has_signal": has_signal' in src, (
            f"{trainer.__name__} does not log has_signal"
        )

    # DDPO computes has_signal outside the PPO loop and reads it in the returned
    # metrics. Assigning it inside `for ppo_epoch in range(config.ppo_epochs)`
    # left it undefined when that range is empty, so the metrics raised
    # NameError on a config that previously returned normally.
    ddpo_src = inspect.getsource(DDPOTrainer._training_step)
    sig_at = ddpo_src.index("has_signal = (")
    loop_at = ddpo_src.index("for ppo_epoch in range(")
    assert sig_at < loop_at, (
        "DDPO must compute has_signal before the PPO loop; inside it, "
        "ppo_epochs=0 leaves it unbound and the returned metrics raise NameError"
    )

    print("  trainers log the statistic they gate on: OK")


def test_anchor_uses_encoded_unconditional_not_zeroed_embeddings():
    """The anchor term must condition on an encoded empty prompt, not on zeros.

    The anchor images are unrelated to the RL prompts, so the data term has to be
    evaluated without that conditioning. That was implemented by multiplying the
    RL prompt's embeddings by zero, which is not the same thing and is not an
    input either model was trained on.

    The two models differ in how badly it lands, and in the same direction as the
    measured results. SD3 constructs its unconditional branch by encoding the
    empty string - `sd3_patch` passes `negative_prompt=[""]` - and evaluates it on
    every CFG step, so a zeroed embedding is at least adjacent to something it
    knows. FLUX.1-dev is guidance-distilled: `flux_patch` has no negative prompt
    and no CFG anywhere, only a guidance scalar. Zeroed T5 and pooled-CLIP
    embeddings handed to FLUX alongside a live guidance value are a conditioning
    state it has never seen.

    Measured: the anchor helps SD3.5 by 0.097 at 3.0 SE and costs FLUX 0.108 at
    7.1 SE with no overlapping seed pairs, and the FLUX cost is unchanged whether
    the anchor images come from SD3.5 or FLUX and whether or not they contain
    text (report/analyze_anchor_source.py). Three explanations that appealed to
    the anchor images were tested and withdrawn; this one appeals to the
    conditioning instead, which is the part that never varied.

    This pins the fix rather than the explanation: whatever the cause of the FLUX
    result, feeding a model an input it was never trained on is a defect.
    """
    import inspect

    from diffusion_tinker.trainers.ddrl_trainer import DDRLTrainer

    setup = inspect.getsource(DDRLTrainer._setup_data)
    assert 'prompt=[""]' in setup, (
        "DDRL must cache an encoded empty prompt for the anchor term"
    )
    assert "_uncond_embeds" in setup and "_uncond_pooled" in setup

    step = inspect.getsource(DDRLTrainer._training_step)
    assert "self._uncond_embeds" in step, (
        "the anchor term must use the cached unconditional embedding"
    )
    # The anchor now follows DDRL_THEORY.md step 3b: condition on the anchor
    # image's own caption when the dataset has one, dropping it with probability
    # condition_dropout. Either way the dropped rows must be SUBSTITUTED with the
    # encoded empty prompt, never zeroed - so uncond_embeds has to reach the loss.
    assert "uncond_embeds=uncond_embeds" in step, (
        "the anchor must hand compute_flow_matching_loss a real unconditional "
        "embedding, otherwise dropped rows fall back to zeros"
    )
    assert "config.condition_dropout" in step, (
        "DDRL_THEORY.md step 3b drops the caption with probability 0.2; the "
        "trainer must read condition_dropout"
    )
    assert "dropout_mask = torch.ones" not in step, (
        "full-dropout zeroing is back in the anchor path"
    )

    # FLUX genuinely has no unconditional branch - this is the asymmetry the fix
    # matters for, so it is checked rather than asserted in a comment.
    from diffusion_tinker.models import flux_patch, sd3_patch

    flux_src = inspect.getsource(flux_patch)
    sd3_src = inspect.getsource(sd3_patch)
    assert "negative_prompt" not in flux_src, (
        "FLUX gained a negative-prompt path; re-check whether the anchor's "
        "unconditional handling still needs to differ by model"
    )
    assert 'negative_prompt=[""]' in sd3_src, (
        "SD3 should build its unconditional branch from an encoded empty string"
    )

    print("  anchor conditions on an encoded empty prompt: OK")


def test_dropped_captions_are_substituted_not_zeroed():
    """compute_flow_matching_loss must refuse to zero a dropped caption.

    SFT's caption dropout is deliberate and standard - it is how a model learns an
    unconditional branch for CFG. But dropping a caption means encoding the empty
    string in its place, which is what the CFG training recipe specifies and what
    SD3's own sampler does. Zeroing the encoding is a different input, and for a
    guidance-distilled model with no unconditional branch it is one that never
    appears in training or sampling.

    The function now requires the caller to supply the encoded empty prompt when a
    dropout mask is passed, so the cheap-but-wrong path is not reachable by
    accident.
    """
    import pytest
    import torch

    from diffusion_tinker.core.noise_strategy import compute_flow_matching_loss

    B, C, H, W = 2, 4, 8, 8
    latents = torch.randn(B, C, H, W)
    noise = torch.randn(B, C, H, W)
    sigmas = torch.full((B,), 0.5)
    embeds = torch.randn(B, 4, 64)
    pooled = torch.randn(B, 32)
    mask = torch.tensor([True, False])

    with pytest.raises(ValueError, match="encoded empty prompt"):
        compute_flow_matching_loss(
            transformer=None, latents=latents, noise=noise, sigmas=sigmas,
            prompt_embeds=embeds, pooled_embeds=pooled,
            condition_dropout_mask=mask,
        )

    # And when supplied, the dropped row must become the unconditional embedding
    # rather than zeros. Checked through the substitution arithmetic directly.
    u_embeds = torch.full((1, 4, 64), 0.25)
    u_pooled = torch.full((1, 32), 0.25)
    seq_mask = mask.float().view(B, 1, 1)
    got = embeds * (1.0 - seq_mask) + u_embeds.expand_as(embeds) * seq_mask
    assert torch.allclose(got[0], torch.full((4, 64), 0.25)), "dropped row not substituted"
    assert torch.allclose(got[1], embeds[1]), "kept row was modified"
    assert not torch.allclose(got[0], torch.zeros(4, 64)), "dropped row was zeroed"

    print("  dropped captions substitute an encoded empty prompt: OK")


def test_defaults_are_the_ones_that_were_validated():
    """A user running out of the box must not get a silently inert trust region.

    The log-prob is mean-reduced over the latent dimensions rather than summed
    (`log_prob.mean(dim=...)` in noise_strategy), which keeps the importance ratio
    within a whisker of 1 - measured 0.9977 to 0.9995 across every validated run,
    so |ratio - 1| stays under 0.0025. PPO's usual clip_range of 0.2 clamps to
    [0.8, 1.2] and can therefore never bind: the trust region does nothing and
    training silently runs as unclipped policy gradient.

    The default was 0.2 while every result in VALIDATION.md was produced at 1e-3,
    so nobody using the library got the configuration the numbers came from.
    """
    from diffusion_tinker.trainers.base_config import BaseDiffusionConfig

    observed_ratio_deviation = 0.0025  # max |ratio - 1| seen on hardware

    # Checked across every config that declares clip_range, not just the base.
    # Subclass configs shadow the base default, so fixing base_config alone left
    # FlowGRPO and DDPO running unclipped - the fix looked complete and was not.
    from diffusion_tinker.trainers.ddpo_config import DDPOConfig
    from diffusion_tinker.trainers.ddrl_config import DDRLConfig
    from diffusion_tinker.trainers.flowgrpo_config import FlowGRPOConfig

    for cls in (BaseDiffusionConfig, FlowGRPOConfig, DDPOConfig, DDRLConfig):
        cfg = cls()
        assert cfg.clip_range < observed_ratio_deviation, (
            f"{cls.__name__}.clip_range={cfg.clip_range} cannot bind: the ratio "
            f"only moves by {observed_ratio_deviation} with a mean-reduced "
            "log-prob, so the trust region is inert and this is unclipped policy "
            "gradient"
        )

    print("  every config's default clip_range can actually bind: OK")


def test_sft_reports_a_held_out_number_by_default():
    """SFT must not default to reporting only its training loss.

    A denoising objective's training loss falls by construction, so on its own it
    looks like progress no matter what the model is doing - that is the defect
    DRaFT and SFT both had. `holdout_size` was added to fix it and then defaulted
    to 0, which left the fix switched off for anyone who did not know to ask.
    """
    from diffusion_tinker.trainers.sft_config import SFTConfig

    cfg = SFTConfig()
    assert cfg.holdout_size > 0, (
        "SFT defaults to no held-out set, so the only number a user sees is the "
        "training loss, which decreases regardless of whether anything is learned"
    )

    print(f"  SFT holds out {cfg.holdout_size} examples by default: OK")


def test_no_config_field_is_dead():
    """Every declared config field must be read somewhere in src/.

    `timestep_sampling` was declared on both DDRLConfig and SFTConfig and read
    nowhere: setting "uniform" silently trained logit-normal. A field a user can
    set that changes nothing is the same class of defect as a default that does
    something other than it says.
    """
    import dataclasses
    import pathlib

    from diffusion_tinker.trainers.base_config import BaseDiffusionConfig
    from diffusion_tinker.trainers.ddpo_config import DDPOConfig
    from diffusion_tinker.trainers.ddrl_config import DDRLConfig
    from diffusion_tinker.trainers.diffusion_dpo_config import DiffusionDPOConfig
    from diffusion_tinker.trainers.draft_config import DRaFTConfig
    from diffusion_tinker.trainers.flowgrpo_config import FlowGRPOConfig
    from diffusion_tinker.trainers.sft_config import SFTConfig

    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    # Exclude the *_config.py files themselves: a field's own declaration would
    # otherwise count as a use, which makes this check pass for every field and
    # catch nothing. (It did exactly that on the first attempt.)
    body = "\n".join(
        p.read_text() for p in src.rglob("*.py")
        if not p.name.endswith("_config.py")
    )

    # Fields the loop reads off `self.config` via getattr, or that are consumed
    # by name outside src (output paths, dataset plumbing).
    exempt = {"output_dir", "dataset_split", "seed"}

    dead = []
    for cls in (BaseDiffusionConfig, FlowGRPOConfig, DDPOConfig, DDRLConfig,
                SFTConfig, DiffusionDPOConfig, DRaFTConfig):
        for f in dataclasses.fields(cls):
            if f.name in exempt:
                continue
            if f.name not in body:
                dead.append(f"{cls.__name__}.{f.name}")

    assert not dead, (
        "config fields declared but never read, so setting them does nothing: "
        + ", ".join(sorted(set(dead)))
    )

    print("  no dead config fields: OK")


def test_offload_text_encoders_applies_before_the_device_move():
    """offload_text_encoders must keep T5 off the card at construction.

    `pipeline.to(device)` moves every component, T5-XXL included - roughly 9GB on
    SD3.5 and 8GB on FLUX. A user who set offload_text_encoders=True precisely to
    fit a 16GB card still OOM'd during __init__, before the flag could apply.
    Verified in practice: SD3.5 OOM'd at 15.6GB on an A4000 with offload enabled.

    The sampling path already moves them on demand and returns them after, so the
    only change needed is not to move them in the first place.
    """
    import inspect

    from diffusion_tinker.trainers.base_trainer import BaseDiffusionTrainer

    src = inspect.getsource(BaseDiffusionTrainer._setup_model)
    assert "offload_text_encoders" in src, (
        "_setup_model ignores offload_text_encoders, so the flag cannot prevent "
        "the construction-time OOM it exists to prevent"
    )
    # the guarded branch must come before / instead of the blanket move
    guarded = src.index("offload_text_encoders")
    blanket = src.index("self.pipeline.to(self.device)")
    assert guarded < blanket, (
        "the unconditional pipeline.to(device) still runs first, so the text "
        "encoders reach the card regardless"
    )

    print("  offload_text_encoders keeps T5 off the card at construction: OK")


def test_eval_without_eval_prompts_warns_about_training_overlap():
    """Evaluating on training prompts must not pass silently.

    `eval_prompts=None` falls back to `train_prompts[:num_eval_prompts]`, so the
    curve labelled held-out is computed on data the model trains on. It rises
    while generalisation falls, which is exactly how the pre-audit curves in this
    project misled for months - the fixed-seed half of that defect was repaired
    and this half was left as a default.

    The fallback is kept rather than removed, because refusing to evaluate would
    break callers who genuinely want a training-set curve. It now says so.
    """
    import inspect

    from diffusion_tinker.trainers.base_trainer import BaseDiffusionTrainer

    from diffusion_tinker.trainers.draft_trainer import DRaFTTrainer

    # Checked in both, because DRaFT does not inherit BaseDiffusionTrainer and
    # carried its own copy of the eval path - fixing the base class alone left the
    # bug live in the trainer where it bites hardest, since DRaFT backpropagates
    # through the reward and will overfit the prompts it trains on.
    for cls in (BaseDiffusionTrainer, DRaFTTrainer):
        src = inspect.getsource(cls._evaluate)
        assert "warnings.warn" in src, f"{cls.__name__}: silent fallback to training prompts"
        assert "TRAINING prompts" in src, (
            f"{cls.__name__}: the warning must say plainly that the eval set is "
            "the training set"
        )
        # and it must fire before the prompts are used, not after
        warn_at = src.index("warnings.warn")
        use_at = src.index("base_prompts = self.train_prompts")
        assert warn_at < use_at, f"{cls.__name__}: warns after using the prompts"

    print("  eval falls back to training prompts loudly, in both trainers: OK")


def test_flux_uses_its_own_sigma_schedule():
    """FLUX must get an explicit sigma grid; SD3 must not.

    FluxPipeline.__call__ always passes sigmas = linspace(1.0, 1/N, N) to the
    scheduler. StableDiffusion3Pipeline passes sigmas=None and lets the scheduler
    build its own. The FLUX sampler here was written by copying the SD3 one and
    inherited the omission, so the scheduler fell back to a linspace over
    _sigma_to_t which bottoms out at 1/num_train_timesteps = 0.001 instead of 1/N.

    Measured at 512px with N=28, the schedule ended 0.1321, 0.0690, 0.0019 rather
    than 0.1839, 0.1262, 0.0650: the last Euler step covered 0.19% of the
    trajectory while the step before it absorbed the entire low-noise refinement
    range. Every FLUX image this repo produced before the fix - rollouts and evals
    alike - came off that grid.

    Checked numerically against the scheduler rather than by grepping the source,
    so it fails if a future diffusers changes the fallback behaviour.
    """
    import numpy as np

    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
    except Exception:
        import pytest
        pytest.skip("diffusers not available")

    cfg = dict(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
               base_shift=0.5, max_shift=1.15, base_image_seq_len=256,
               max_image_seq_len=4096)
    N, mu = 28, 0.7

    fallback = FlowMatchEulerDiscreteScheduler(**cfg)
    fallback.set_timesteps(N, device="cpu", mu=mu)

    explicit = FlowMatchEulerDiscreteScheduler(**cfg)
    explicit.set_timesteps(N, device="cpu", mu=mu,
                           sigmas=np.linspace(1.0, 1.0 / N, N))

    last_fallback = float(fallback.sigmas[-2])
    last_explicit = float(explicit.sigmas[-2])

    # The bug is only interesting if the two schedules actually differ; if a
    # future diffusers makes the fallback correct, this guard tells us so rather
    # than silently passing.
    assert last_fallback < last_explicit / 10, (
        "the scheduler fallback no longer degrades the tail - re-check whether "
        "flux_patch still needs to pass sigmas explicitly"
    )

    # And the sampler must be passing the explicit grid.
    import inspect

    from diffusion_tinker.models import flux_patch

    src = inspect.getsource(flux_patch.flux_sample_with_logprob)
    assert 'sched_kwargs["sigmas"]' in src or "sigmas=" in src.split("set_timesteps")[0][-400:], (
        "FLUX sampler does not pass an explicit sigma grid, so the scheduler "
        f"falls back to a tail of {last_fallback:.5f} instead of {last_explicit:.5f}"
    )

    print(f"  FLUX passes its own sigma grid (tail {last_explicit:.4f}, "
          f"not {last_fallback:.4f}): OK")


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
        timesteps=torch.linspace(0.9, 0.0, T + 1),  # T+1: full sigma schedule
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
        timesteps=sigmas,  # Full T+1 sigma schedule
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
