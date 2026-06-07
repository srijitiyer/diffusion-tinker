"""Multi-reward composition for combining multiple reward signals."""

from __future__ import annotations

import torch

from diffusion_tinker.rewards.base import BaseReward
from diffusion_tinker.rewards.protocol import RewardContext, RewardOutput


class ComposedReward(BaseReward):
    """Combines multiple reward functions with weighted aggregation."""

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
        if not rewards:
            raise ValueError("ComposedReward requires at least one reward.")
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
            # Normalize each reward to zero-mean/unit-variance across the batch.
            # Use population std (correction=0) so a single element gives 0, not
            # NaN. NOTE: do not pair this mode with an RL trainer that also
            # normalizes rewards into advantages (FlowGRPO/DDPO/DDRL) - that
            # double-normalizes. Intended for DRaFT / reward inspection.
            combined = torch.zeros_like(all_scores[0])
            for scores, w in zip(all_scores, self.weights):
                mean = scores.mean()
                std = scores.std(correction=0)
                normalized = (scores - mean) / (std + 1e-4)
                combined = combined + w * normalized
        else:
            combined = torch.zeros_like(all_scores[0])
            for scores, w in zip(all_scores, self.weights):
                combined = combined + w * scores

        return RewardOutput(scores=combined, metadata=metadata)

    def differentiable_forward(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor | None:
        all_scores = []
        for reward_fn in self.rewards:
            scores = reward_fn.differentiable_forward(images, prompts)
            if scores is None:
                return None
            all_scores.append(scores)

        combined = torch.zeros_like(all_scores[0])
        if self.mode == "advantage_level":
            for scores, w in zip(all_scores, self.weights):
                mean = scores.mean()
                std = scores.std(correction=0)
                combined = combined + w * (scores - mean) / (std + 1e-4)
        else:
            for scores, w in zip(all_scores, self.weights):
                combined = combined + w * scores
        return combined

    def to(self, device: str | torch.device) -> ComposedReward:
        super().to(device)
        for r in self.rewards:
            r.to(device)
        return self
