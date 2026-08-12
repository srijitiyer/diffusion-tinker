"""GenEval-style object-counting reward using OWLv2 open-vocabulary detection.

Prompts are templated as "a photo of <N> <object>" (e.g. "a photo of three
apples"). The reward detects <object> with OWLv2, counts confident boxes, and
scores how close the count is to <N>. SD3.5 is poor at exact object counts, so
this reward has real headroom - unlike PickScore, RL can visibly improve it.
"""

from __future__ import annotations

import re

import torch

from diffusion_tinker.rewards.base import BaseReward
from diffusion_tinker.rewards.protocol import RewardContext, RewardOutput
from diffusion_tinker.rewards.resolve import register_reward

_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8}
_COLORS = "red|blue|green|yellow|white|black|orange|purple|brown|pink"


def _parse_target(prompt: str) -> tuple[int | None, str | None]:
    """Pull the target count and object noun out of a templated prompt.

    "a photo of three red apples" -> (3, "apple")
    """
    m = re.search(rf"of\s+(\w+)\s+(?:(?:{_COLORS})\s+)?([a-z]+?)s?\b", prompt.lower())
    if not m:
        return None, None
    word, obj = m.group(1), m.group(2)
    n = _NUM.get(word)
    if n is None:
        try:
            n = int(word)
        except ValueError:
            return None, None
    return n, obj


@register_reward("count")
class CountReward(BaseReward):
    """Object-counting correctness via OWLv2 (GenEval-style, has headroom)."""

    name = "count"

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32, threshold: float = 0.1):
        super().__init__(device=device, dtype=dtype)
        self._model = None
        self._proc = None
        self.threshold = threshold

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        name = "google/owlv2-base-patch16-ensemble"
        self._proc = Owlv2Processor.from_pretrained(name)
        self._model = Owlv2ForObjectDetection.from_pretrained(name).to(self.device).eval()

    def _count(self, image, obj: str) -> int:
        self._ensure_loaded()
        queries = [[f"a photo of a {obj}"]]
        inputs = self._proc(text=queries, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        # transformers changed this across versions: <=4.x exposes
        # post_process_object_detection; 5.x removed it and requires text_labels
        # on post_process_grounded_object_detection (without it, zero detections).
        if hasattr(self._proc, "post_process_object_detection"):
            results = self._proc.post_process_object_detection(
                outputs, threshold=self.threshold, target_sizes=target_sizes
            )[0]
        else:
            results = self._proc.post_process_grounded_object_detection(
                outputs, threshold=self.threshold, target_sizes=target_sizes, text_labels=queries
            )[0]
        return int(len(results["scores"]))

    def _score_single(self, image, n: int | None, obj: str | None) -> float:
        if n is None or obj is None:
            return 0.0
        c = self._count(image, obj)
        # Smooth: 1.0 at the exact count, falling off with the miss so GRPO gets a
        # gradient toward the right number instead of an all-or-nothing signal.
        #
        # Divided by (n + 1), not max(n, 1). The old denominator made this exactly
        # binary at n=1 - reachable, since "a", "an" and "one" all parse to 1 - and
        # flat at 0 for every c >= 2n, so a near miss and a wild overcount were
        # indistinguishable and there was no gradient between them. Both cases
        # contradicted the rationale in the line above, which is the whole reason
        # this deviates from GenEval's binary score. With (n + 1): n=1 gives 0.5 at
        # c=0 and c=2 instead of 0.0, and n=3 gives 0.25 at c=6 instead of 0.0.
        return float(max(0.0, 1.0 - abs(c - n) / (n + 1.0)))

    def to(self, device):
        # Every other model-backed reward overrides this (aesthetic, clip_score,
        # pickscore, hps_v2). Without it BaseReward.to updates self.device only,
        # so a CountReward whose OWLv2 is already loaded starts sending inputs to
        # the new device while the model stays on the old one - a device-mismatch
        # RuntimeError at scoring time.
        super().to(device)
        if getattr(self, "_model", None) is not None:
            self._model = self._model.to(device)
        return self

    def _compute(self, ctx: RewardContext) -> RewardOutput:
        self._ensure_loaded()
        scores = []
        for image, prompt in zip(ctx.images, ctx.prompts):
            n, obj = _parse_target(prompt)
            scores.append(self._score_single(image, n, obj))
        return RewardOutput(scores=torch.tensor(scores, dtype=torch.float32))
