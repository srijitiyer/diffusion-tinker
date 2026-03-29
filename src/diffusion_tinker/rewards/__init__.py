from diffusion_tinker.rewards.protocol import RewardContext, RewardOutput
from diffusion_tinker.rewards.resolve import register_reward, resolve_reward

__all__ = ["RewardContext", "RewardOutput", "resolve_reward", "register_reward"]


def _ensure_builtins_registered():
    """Lazy-load builtin rewards to trigger registration."""
    import diffusion_tinker.rewards.builtin  # noqa: F401


_ensure_builtins_registered()
