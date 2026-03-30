from diffusion_tinker.rewards.builtin import (
    aesthetic,  # noqa: F401
    clip_score,  # noqa: F401
)

# Optional rewards with heavy dependencies
try:
    from diffusion_tinker.rewards.builtin import ocr  # noqa: F401
except ImportError:
    pass

try:
    from diffusion_tinker.rewards.builtin import hps_v2  # noqa: F401
except ImportError:
    pass
