"""Aesthetic score predictor reward.

Architecture: CLIP-ViT-L/14 image features -> 5-layer MLP -> scalar score.
Trained on the AVA dataset with SAC+LOGOS augmentation.
Output range: roughly 1-10 (AVA aesthetic ratings).

Reference: REWARD_MODELS_TECHNICAL.md Section 2
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffusion_tinker.rewards.base import BaseReward
from diffusion_tinker.rewards.protocol import RewardContext, RewardOutput
from diffusion_tinker.rewards.resolve import register_reward


class _AestheticMLP(nn.Module):
    """The aesthetic predictor MLP: 768 -> 1024 -> 128 -> 64 -> 16 -> 1.

    5 linear layers with dropout after layers 1-3. NO activation functions -
    the original model uses a pure linear stack with dropout regularization.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@register_reward("aesthetic")
class AestheticReward(BaseReward):
    """CLIP + MLP aesthetic score predictor.

    Usage:
        reward = AestheticReward(device="cuda")
        output = reward(RewardContext(images=pil_images, prompts=prompts))
        # output.scores is shape (B,) with values ~1-10
    """

    name = "aesthetic"

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self._clip = None
        self._processor = None
        self._mlp = None

    def _ensure_loaded(self):
        if self._clip is not None:
            return

        from transformers import CLIPModel, CLIPProcessor

        self._clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device, self.dtype)
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self._mlp = _AestheticMLP()
        # Load pretrained weights
        state_dict = torch.hub.load_state_dict_from_url(
            "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth",
            map_location="cpu",
        )
        self._mlp.load_state_dict(state_dict)
        self._mlp = self._mlp.to(self.device, self.dtype)

        self._clip.eval()
        self._mlp.eval()
        self._clip.requires_grad_(False)
        self._mlp.requires_grad_(False)

    def _compute(self, ctx: RewardContext) -> RewardOutput:
        self._ensure_loaded()

        inputs = self._processor(images=ctx.images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)

        # Get CLIP image embeddings via the vision model + projection
        vision_out = self._clip.vision_model(pixel_values=pixel_values)
        # Pool: use the CLS token output (first token), then project
        pooled = vision_out.pooler_output  # (B, 768)
        embed = self._clip.visual_projection(pooled)  # (B, 768)

        # L2 normalize
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)

        # MLP forward (expects 768-dim input)
        scores = self._mlp(embed).squeeze(-1)

        return RewardOutput(scores=scores.float().cpu())

    def to(self, device: str | torch.device) -> AestheticReward:
        super().to(device)
        if self._clip is not None:
            self._clip = self._clip.to(self.device)
            self._mlp = self._mlp.to(self.device)
        return self
