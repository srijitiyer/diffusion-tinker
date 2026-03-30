"""HPS v2 reward - Human Preference Score using OpenCLIP ViT-H-14.

Trained on the HPD (Human Preference Dataset) with 798K+ preference pairs.
Requires: pip install open-clip-torch

Output range: roughly 0.20-0.35 for generated images.
"""

from __future__ import annotations

import torch

from diffusion_tinker.rewards.base import BaseReward
from diffusion_tinker.rewards.protocol import RewardContext, RewardOutput
from diffusion_tinker.rewards.resolve import register_reward


@register_reward("hps_v2")
class HPSv2Reward(BaseReward):
    """Human Preference Score v2.1 reward.

    Uses OpenCLIP ViT-H-14 fine-tuned on human preference data.

    Usage:
        reward = HPSv2Reward(device="cuda")
        output = reward(RewardContext(images=pil_images, prompts=prompts))
    """

    name = "hps_v2"

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        import open_clip
        from huggingface_hub import hf_hub_download

        # Load base OpenCLIP ViT-H-14
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2B-s32B-b79K", device=self.device
        )

        # Download and load HPS v2.1 fine-tuned weights
        ckpt_path = hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])

        model = model.to(self.device, self.dtype)
        model.eval()
        model.requires_grad_(False)

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-H-14")

    def _compute(self, ctx: RewardContext) -> RewardOutput:
        self._ensure_loaded()

        # Preprocess images
        image_tensors = torch.stack([self._preprocess(img) for img in ctx.images])
        image_tensors = image_tensors.to(self.device, self.dtype)

        # Tokenize prompts
        text_tokens = self._tokenizer(ctx.prompts).to(self.device)

        # Forward pass
        image_features = self._model.encode_image(image_tensors)
        text_features = self._model.encode_text(text_tokens)

        # L2 normalize
        image_features = image_features / torch.linalg.vector_norm(image_features, dim=-1, keepdim=True)
        text_features = text_features / torch.linalg.vector_norm(text_features, dim=-1, keepdim=True)

        # Per-sample cosine similarity
        scores = (image_features * text_features).sum(dim=-1)

        return RewardOutput(scores=scores.float().cpu())

    def to(self, device: str | torch.device) -> HPSv2Reward:
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self
