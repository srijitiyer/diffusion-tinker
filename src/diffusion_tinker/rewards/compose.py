"""Multi-reward composition for combining multiple reward signals."""

from __future__ import annotations

import torch

from diffusion_tinker.rewards.base import BaseReward
from diffusion_tinker.rewards.protocol import RewardContext, RewardOutput


class ComposedReward(BaseReward):
    """Combines multiple reward functions with weighted aggregation.

    Two aggregation modes:
    - "weighted_sum": score = sum(w_i * r_i) - simple weighted sum of raw scores
    - "advantage_level": normalize each reward per-batch, then weight and sum.
      This is DanceGRPO's approach - prevents rewards on different scales from
      dominating.

    Usage:
        composed = ComposedReward(
            rewards=["aesthetic", "clip_score"],
            weights=[0.6, 0.4],
            mode="advantage_level",
            device="cuda",
        )
        output = composed(ctx)
    """

    name = "composed"

    def __init__(
        self,
        rewards: list[BaseReward],
        weights: list[float] | None = None,
        mode: str = "weighted_sum",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device=device, dtype=dtype)
        self.rewards = rewards
        self.weights = weights or [1.0 / len(rewards)] * len(rewards)
        self.mode = mode

        if len(self.rewards) != len(self.weights):
            raise ValueError(f"Got {len(self.rewards)} rewards but {len(self.weights)} weights")

    def _compute(self, ctx: RewardContext) -> RewardOutput:
        all_scores = []
        metadata = {}

        for i, reward_fn in enumerate(self.rewards):
            output = reward_fn(ctx)
            all_scores.append(output.scores)
            metadata[f"reward_{reward_fn.name}"] = output.scores.mean().item()

        if self.mode == "advantage_level":
            # DanceGRPO: normalize each reward per-batch, then weight
            combined = torch.zeros_like(all_scores[0])
            for scores, w in zip(all_scores, self.weights):
                mean = scores.mean()
                std = scores.std()
                normalized = (scores - mean) / (std + 1e-4)
                combined = combined + w * normalized
        else:
            # Simple weighted sum
            combined = torch.zeros_like(all_scores[0])
            for scores, w in zip(all_scores, self.weights):
                combined = combined + w * scores

        return RewardOutput(scores=combined, metadata=metadata)

    def to(self, device: str | torch.device) -> ComposedReward:
        super().to(device)
        for r in self.rewards:
            r.to(device)
        return self
