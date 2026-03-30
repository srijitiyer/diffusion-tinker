from __future__ import annotations

from typing import Callable

from diffusion_tinker.rewards.base import BaseReward
from diffusion_tinker.rewards.protocol import RewardContext, RewardFunc, RewardOutput

REWARD_REGISTRY: dict[str, type[BaseReward]] = {}


def register_reward(name: str):
    """Decorator to register a reward class by string name."""

    def decorator(cls: type[BaseReward]):
        REWARD_REGISTRY[name] = cls
        return cls

    return decorator


class _CallableWrapper(BaseReward):
    """Wraps a plain function as a BaseReward."""

    name = "custom"

    def __init__(self, fn: Callable, device: str = "cpu"):
        super().__init__(device=device)
        self.fn = fn

    def _compute(self, ctx: RewardContext) -> RewardOutput:
        result = self.fn(ctx)
        if isinstance(result, RewardOutput):
            return result
        # Allow returning just a list of floats or a tensor
        import torch

        if isinstance(result, list):
            return RewardOutput(scores=torch.tensor(result, dtype=torch.float32))
        if isinstance(result, torch.Tensor):
            return RewardOutput(scores=result.float())
        raise TypeError(f"Reward function returned unsupported type: {type(result)}")


def resolve_reward(
    reward_func: RewardFunc,
    device: str = "cpu",
    reward_weights: list[float] | None = None,
    reward_mode: str = "weighted_sum",
) -> BaseReward:
    """Resolve a reward specification into a callable BaseReward.

    Args:
        reward_func: one of:
            - str: look up in REWARD_REGISTRY (e.g., "aesthetic")
            - Callable: wrap as BaseReward
            - BaseReward: return directly
            - list: multiple rewards, combined via ComposedReward
        device: device for the reward model
        reward_weights: weights for multi-reward (only used when reward_func is a list)
        reward_mode: aggregation mode for multi-reward ("weighted_sum" or "advantage_level")

    Returns:
        A BaseReward instance ready to call.
    """
    if isinstance(reward_func, list):
        from diffusion_tinker.rewards.compose import ComposedReward

        resolved = [resolve_reward(r, device=device) for r in reward_func]
        return ComposedReward(rewards=resolved, weights=reward_weights, mode=reward_mode, device=device)
    elif isinstance(reward_func, str):
        if reward_func not in REWARD_REGISTRY:
            available = ", ".join(sorted(REWARD_REGISTRY.keys()))
            raise ValueError(f"Unknown reward '{reward_func}'. Available: {available}")
        return REWARD_REGISTRY[reward_func](device=device)
    elif isinstance(reward_func, BaseReward):
        return reward_func.to(device)
    elif callable(reward_func):
        return _CallableWrapper(reward_func, device=device)
    else:
        raise TypeError(f"Cannot resolve reward of type {type(reward_func)}. Expected str, callable, or BaseReward.")
