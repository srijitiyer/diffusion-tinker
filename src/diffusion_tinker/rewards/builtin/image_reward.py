"""ImageReward reward (BLIP-based human preference, THUDM/ImageReward-v1.0)."""

from __future__ import annotations

import torch

from diffusion_tinker.rewards.base import BaseReward
from diffusion_tinker.rewards.protocol import RewardContext, RewardOutput
from diffusion_tinker.rewards.resolve import register_reward

_MODEL_ID = "ImageReward-v1.0"
_MAX_TOKENS = 35


@register_reward("imagereward")
class ImageRewardReward(BaseReward):
    """Human-preference score from the ImageReward BLIP model.

    This is the third quality axis DDRL reports alongside ClipScore and
    PickScore. Scores are standardised by the checkpoint's own mean/std, so
    they sit near zero and are signed.
    """

    name = "imagereward"

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self._model = None

    @staticmethod
    def _patch_transformers_for_blip():
        """ImageReward's vendored BLIP imports three helpers from
        transformers.modeling_utils. Two moved to transformers.pytorch_utils
        and the third was dropped outright by transformers 5. Supply them,
        because pinning transformers low enough for the old path breaks the
        SD3.5 and FLUX pipelines sharing this environment.

        find_pruneable_heads_and_indices only runs when pruning attention
        heads, which BLIP never does here, but it has to exist to import.
        """
        import transformers.modeling_utils as mu
        import transformers.pytorch_utils as pu

        for n in ("apply_chunking_to_forward", "prune_linear_layer"):
            if not hasattr(mu, n):
                setattr(mu, n, getattr(pu, n))

        if not hasattr(mu, "find_pruneable_heads_and_indices"):

            def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                mask = torch.ones(n_heads, head_size)
                heads = set(heads) - already_pruned_heads
                for head in heads:
                    head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                    mask[head] = 0
                mask = mask.view(-1).contiguous().eq(1)
                index = torch.arange(len(mask))[mask].long()
                return heads, index

            mu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    @staticmethod
    def _check_transformers_supported():
        """transformers 5 also dropped additional_special_tokens_ids, which
        BLIP's init_tokenizer assigns from. That one cannot be shimmed from
        out here without reaching into the tokenizer class, so say so plainly
        rather than surfacing an AttributeError three frames into vendored
        BLIP code.
        """
        import transformers

        major = int(transformers.__version__.split(".")[0])
        if major >= 5:
            raise RuntimeError(
                f"ImageReward vendors BLIP written against transformers 4.x and this "
                f"environment has {transformers.__version__}. Score saved images in a "
                f"separate environment pinned to transformers<5 instead of downgrading "
                f"here, which would break the SD3/FLUX pipelines."
            )

    def _ensure_loaded(self):
        if self._model is not None:
            return
        self._check_transformers_supported()
        self._patch_transformers_for_blip()
        import ImageReward as RM

        self._model = RM.load(_MODEL_ID, device=str(self.device))
        self._model.eval()
        self._model.requires_grad_(False)

    def _compute(self, ctx: RewardContext) -> RewardOutput:
        self._ensure_loaded()

        # ImageReward's own score() takes one pair at a time and calls .item(),
        # so drive the underlying BLIP stack directly to score the batch in one
        # pass. This mirrors upstream score() exactly, including the final
        # (r - mean) / std standardisation.
        pixel_values = torch.stack(
            [self._model.preprocess(img) for img in ctx.images]
        ).to(self.device)

        text_input = self._model.blip.tokenizer(
            ctx.prompts,
            padding="max_length",
            truncation=True,
            max_length=_MAX_TOKENS,
            return_tensors="pt",
        ).to(self.device)

        image_embeds = self._model.blip.visual_encoder(pixel_values)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)

        text_output = self._model.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()
        scores = self._model.mlp(txt_features).squeeze(-1)
        scores = (scores - self._model.mean) / self._model.std

        return RewardOutput(scores=scores.float().cpu())

    def to(self, device: str | torch.device) -> ImageRewardReward:
        super().to(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self
