"""CLIP Score reward - cosine similarity between CLIP text and image embeddings.

Uses openai/clip-vit-large-patch14. Output is the per-sample diagonal of the
similarity matrix, normalized by a factor (default 100, matching torchmetrics convention).
"""

from __future__ import annotations

import torch

from diffusion_tinker.rewards.base import BaseReward
from diffusion_tinker.rewards.protocol import RewardContext, RewardOutput
from diffusion_tinker.rewards.resolve import register_reward


@register_reward("clip_score")
class CLIPScoreReward(BaseReward):
    name = "clip_score"

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self._clip = None
        self._processor = None

    def _ensure_loaded(self):
        if self._clip is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device, self.dtype)
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self._clip.eval()
        self._clip.requires_grad_(False)

    def _compute(self, ctx: RewardContext) -> RewardOutput:
        self._ensure_loaded()

        # Process images and text together
        inputs = self._processor(
            images=ctx.images,
            text=ctx.prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Get image and text features
        image_features = self._clip.get_image_features(pixel_values=pixel_values)
        if not isinstance(image_features, torch.Tensor):
            image_features = self._clip.visual_projection(
                self._clip.vision_model(pixel_values=pixel_values).pooler_output
            )

        text_features = self._clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        if not isinstance(text_features, torch.Tensor):
            text_features = text_features.text_embeds if hasattr(text_features, "text_embeds") else text_features[0]

        # L2 normalize
        image_features = image_features / torch.linalg.vector_norm(image_features, dim=-1, keepdim=True)
        text_features = text_features / torch.linalg.vector_norm(text_features, dim=-1, keepdim=True)

        # Per-sample cosine similarity (diagonal of the cross-similarity matrix)
        # Multiply by 100 to match torchmetrics CLIP score convention
        scores = (image_features * text_features).sum(dim=-1) * 100.0

        return RewardOutput(scores=scores.float().cpu())

    def to(self, device: str | torch.device) -> CLIPScoreReward:
        super().to(device)
        if self._clip is not None:
            self._clip = self._clip.to(self.device)
        return self
