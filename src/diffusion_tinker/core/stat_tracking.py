from __future__ import annotations

from collections import defaultdict

import torch


class PerPromptStatTracker:
    """Tracks per-prompt reward statistics for advantage normalization.

    Groups rewards by prompt and normalizes within each group:
        advantage = (reward - group_mean) / (group_std + eps)

    Based on FlowGRPO's implementation. Uses eps=1e-4 (not 1e-8).
    """

    def __init__(self, eps: float = 1e-4):
        self.eps = eps
        self.stats: dict[str, list[float]] = defaultdict(list)

    def update(self, prompts: list[str], rewards: torch.Tensor) -> torch.Tensor:
        """Compute per-prompt normalized advantages.

        Args:
            prompts: list of B prompt strings (may contain duplicates for GRPO groups)
            rewards: tensor of shape (B,)

        Returns:
            advantages: tensor of shape (B,), per-prompt normalized
        """
        advantages = torch.zeros_like(rewards)
        unique_prompts = list(dict.fromkeys(prompts))  # preserve order, deduplicate

        for prompt in unique_prompts:
            indices = [i for i, p in enumerate(prompts) if p == prompt]
            group_rewards = rewards[indices]

            self.stats[prompt].extend(group_rewards.tolist())

            mean = group_rewards.mean()
            std = group_rewards.std()
            advantages[indices] = (group_rewards - mean) / (std + self.eps)

        return advantages

    def clear(self):
        self.stats.clear()
