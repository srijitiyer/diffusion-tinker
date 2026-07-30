from __future__ import annotations

from collections import deque

import torch


class PerPromptStatTracker:
    """Computes per-prompt (GRPO group-relative) advantages.

    advantage = (reward - group_mean) / (std + eps), where the mean is always
    per-prompt. The std is either per-prompt (default) or the global std across
    the whole batch (global_std=True). Global std matters for low-variance
    rewards like aesthetic: dividing by a tiny per-prompt std there just
    amplifies sampling noise into large advantages, so the policy chases noise.
    eps=1e-4 matches FlowGRPO.
    """

    def __init__(self, eps: float = 1e-4, global_std: bool = False):
        self.eps = eps
        self.global_std = global_std

    def update(self, prompts: list[str], rewards: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        unique_prompts = list(dict.fromkeys(prompts))

        batch_std = rewards.std() if rewards.numel() > 1 else torch.tensor(0.0, device=rewards.device)

        for prompt in unique_prompts:
            indices = [i for i, p in enumerate(prompts) if p == prompt]
            group_rewards = rewards[indices]
            mean = group_rewards.mean()
            if self.global_std:
                std = batch_std
            elif len(group_rewards) > 1:
                std = group_rewards.std()  # std() on one element is NaN; guarded above
            else:
                std = torch.tensor(0.0, device=rewards.device)
            advantages[indices] = (group_rewards - mean) / (std + self.eps)

        return advantages


class BufferedPerPromptStatTracker:
    """DDPO's per-prompt advantage estimator (Black et al., 2023).

    Unlike the GRPO-style tracker above, which normalizes against only the
    samples in the current batch, this keeps a rolling per-prompt reward history
    and normalizes against that. Until a prompt has min_count samples in its
    buffer it falls back to the statistics of the whole batch, so early epochs
    are not normalized against one or two samples.

    That history is the actual algorithmic difference between DDPO and GRPO:
    DDPO can tell "this sample beat this prompt's recent average" even when the
    batch holds a single sample per prompt, whereas a within-batch group needs
    several samples of the same prompt to say anything. Using the within-batch
    tracker for DDPO makes it identical to FlowGRPO up to the clip range, which
    is not a test of DDPO.

    std is the population (biased) std here, matching the reference numpy
    implementation rather than torch's default Bessel correction.
    """

    def __init__(self, buffer_size: int = 32, min_count: int = 16, eps: float = 1e-6):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.eps = eps
        self.stats: dict[str, deque] = {}

    def update(self, prompts: list[str], rewards: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        batch_mean = rewards.mean()
        batch_std = rewards.std(unbiased=False) if rewards.numel() > 1 else torch.tensor(0.0, device=rewards.device)

        for prompt in list(dict.fromkeys(prompts)):
            indices = [i for i, p in enumerate(prompts) if p == prompt]
            group_rewards = rewards[indices]

            buf = self.stats.setdefault(prompt, deque(maxlen=self.buffer_size))
            buf.extend(group_rewards.detach().flatten().tolist())

            if len(buf) < self.min_count:
                mean, std = batch_mean, batch_std
            else:
                hist = torch.tensor(list(buf), device=rewards.device, dtype=rewards.dtype)
                mean, std = hist.mean(), hist.std(unbiased=False)

            advantages[indices] = (group_rewards - mean) / (std + self.eps)

        return advantages
