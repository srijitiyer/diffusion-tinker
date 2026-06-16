from __future__ import annotations

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
