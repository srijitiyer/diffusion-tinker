"""PickScore reward (human-preference CLIP-H, yuvalkirstain/PickScore_v1)."""

from __future__ import annotations

import torch

from diffusion_tinker.rewards.base import BaseReward
from diffusion_tinker.rewards.protocol import RewardContext, RewardOutput
from diffusion_tinker.rewards.resolve import register_reward

_MODEL_ID = "yuvalkirstain/PickScore_v1"
_PROCESSOR_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@register_reward("pickscore")
class PickScoreReward(BaseReward):
    """Learned human-preference score. Higher means more preferred.

    PickScore is a CLIP-H model fine-tuned on the Pick-a-Pic preference data,
    so it gives a much more discriminative training signal than the LAION
    aesthetic predictor for RL on image quality.
    """

    name = "pickscore"

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(_PROCESSOR_ID)
        self._model = AutoModel.from_pretrained(_MODEL_ID).to(self.device, self.dtype)
        self._model.eval()
        self._model.requires_grad_(False)

    def _image_embs(self, pixel_values):
        # Go through the submodules directly - get_image_features can return the
        # raw model output on some loaded checkpoints depending on transformers
        # version, so mirror the explicit path used in clip_score.
        vision_out = self._model.vision_model(pixel_values=pixel_values)
        return self._model.visual_projection(vision_out.pooler_output)

    def _text_embs(self, input_ids, attention_mask):
        text_out = self._model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return self._model.text_projection(text_out.pooler_output)

    def _score(self, image_embs, text_embs):
        image_embs = image_embs / torch.linalg.vector_norm(image_embs, dim=-1, keepdim=True)
        text_embs = text_embs / torch.linalg.vector_norm(text_embs, dim=-1, keepdim=True)
        scale = self._model.logit_scale.exp()
        return scale * (image_embs * text_embs).sum(dim=-1)

    def _compute(self, ctx: RewardContext) -> RewardOutput:
        self._ensure_loaded()

        img_inputs = self._processor(images=ctx.images, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(self.device, self.dtype)
        txt_inputs = self._processor(
            text=ctx.prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        input_ids = txt_inputs["input_ids"].to(self.device)
        attention_mask = txt_inputs["attention_mask"].to(self.device)

        image_embs = self._image_embs(pixel_values)
        text_embs = self._text_embs(input_ids, attention_mask)
        scores = self._score(image_embs, text_embs)

        return RewardOutput(scores=scores.float().cpu())

    def differentiable_forward(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        from torchvision.transforms.functional import normalize, resize

        self._ensure_loaded()

        images_resized = resize(images, [224, 224], antialias=True)
        images_normalized = normalize(images_resized, mean=_CLIP_MEAN, std=_CLIP_STD)
        image_embs = self._image_embs(images_normalized.to(self._model.dtype))

        txt_inputs = self._processor(
            text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        text_embs = self._text_embs(
            txt_inputs["input_ids"].to(self.device), txt_inputs["attention_mask"].to(self.device)
        )

        return self._score(image_embs, text_embs.detach())

    def to(self, device: str | torch.device) -> PickScoreReward:
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self
