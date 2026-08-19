"""OCR reward (edit distance between generated text and prompt target)."""

from __future__ import annotations

import os
import re

import numpy as np
import torch

from diffusion_tinker.rewards.base import BaseReward
from diffusion_tinker.rewards.protocol import RewardContext, RewardOutput
from diffusion_tinker.rewards.resolve import register_reward


def _extract_quoted_text(prompt: str) -> str:
    """Extract text between double quotes from a prompt."""
    match = re.search(r'"([^"]+)"', prompt)
    if match:
        return match.group(1)
    match = re.search(r"'([^']+)'", prompt)
    if match:
        return match.group(1)
    return ""


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    try:
        from Levenshtein import distance

        return distance(s1, s2)
    except ImportError:
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i - 1] == s2[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]


@register_reward("ocr")
class OCRReward(BaseReward):
    """OCR accuracy reward.

    Uses EasyOCR (preferred) or PaddleOCR. The prompt must contain
    the target text in double quotes, e.g.:
        'A sign that says "Hello World"'
    """

    name = "ocr"

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self._reader = None
        self._backend = None

    def _ensure_loaded(self):
        if self._reader is not None:
            return

        # Run the detector on the same device as training when that is a GPU.
        # A rollout scores group_size * num_prompts images every epoch, and the
        # recognizer on CPU is slow enough to dominate epoch time; its weights are
        # small next to the transformer, so co-residency is not the problem.
        use_gpu = "cuda" in str(self.device)

        # Which recognizer is a measurement-defining choice, not an implementation
        # detail: the two engines read different amounts of text off the same
        # image, and this reward gives a perfect score whenever the target appears
        # anywhere in what was read, so a chattier engine scores higher. Measured
        # on SD3.5-Medium at 512px/40-step ODE/CFG 4.5, EasyOCR puts the untrained
        # baseline at 0.771 where the DDRL paper reports 0.566. FlowGRPO - whose
        # numbers the paper is aligned with - uses PaddleOCR, so set
        # DT_OCR_BACKEND=paddle for anything compared against published results.
        # "auto" (default) keeps the previous preference order.
        want = os.environ.get("DT_OCR_BACKEND", "auto").lower()

        if want in ("auto", "easyocr"):
            try:
                import easyocr

                self._reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
                self._backend = "easyocr"
                return
            except ImportError:
                if want == "easyocr":
                    raise

        if want in ("auto", "paddle", "paddleocr"):
            try:
                from paddleocr import PaddleOCR

                try:
                    self._reader = PaddleOCR(use_angle_cls=False, lang="en",
                                             use_gpu=use_gpu, show_log=False)
                except (TypeError, ValueError):
                    self._reader = PaddleOCR(lang="en")
                self._backend = "paddleocr"
                return
            except ImportError:
                if want in ("paddle", "paddleocr"):
                    raise

        raise ImportError(
            f"OCR reward found no backend for DT_OCR_BACKEND={want!r}. "
            "Install easyocr or paddleocr."
        )

    def _ocr_image(self, image) -> str:
        """Run OCR and return concatenated recognized text."""
        img_array = np.array(image)

        if self._backend == "easyocr":
            results = self._reader.readtext(img_array, detail=0)
            return "".join(results)

        # PaddleOCR. Matches flow_grpo/ocr.py line for line, including the
        # confidence gate: `res[1][0] if res[1][1] > 0 else ''`. Ours dropped the
        # confidence check, which is nearly always satisfied but not always, and
        # this reward is meant to be comparable to theirs.
        results = self._reader.ocr(img_array, cls=False)
        recognized = ""
        if results and results[0]:
            for res in results[0]:
                if res and len(res) >= 2 and res[1] and res[1][1] > 0:
                    recognized += res[1][0]
        return recognized

    def _score_single(self, image, target: str) -> float:
        if not target:
            return 0.0

        recognized = self._ocr_image(image)

        recognized_norm = recognized.replace(" ", "").lower()
        target_norm = target.replace(" ", "").lower()

        if not target_norm:
            return 0.0

        # Canonical FlowGRPO behaviour: if the target appears anywhere in the
        # recognized text the distance is 0, so incidental extra text in the image
        # does not penalise a correct render (REWARD_MODELS_TECHNICAL.md:555).
        #
        # This previously carried a `len(recognized_norm) <= 1.5 * len(target_norm)`
        # guard, added to stop "cat" matching inside "concatenate". It did stop
        # that, and it also scored a *perfectly rendered* target as 0.0 whenever
        # the image contained any other text: with a 3-character target the guard
        # tolerated at most 4 recognized characters, so one watermark or a second
        # sign took a correct image from 1.0 to 0.0. That false-negative cost is
        # far larger than the false positive it prevented, and it is one-directional
        # (downward), which is the worst kind for an RL reward.
        if target_norm in recognized_norm:
            return 1.0

        dist = _edit_distance(recognized_norm, target_norm)
        return max(0.0, 1.0 - dist / len(target_norm))

    def _compute(self, ctx: RewardContext) -> RewardOutput:
        self._ensure_loaded()

        scores = []
        for image, prompt in zip(ctx.images, ctx.prompts):
            target = _extract_quoted_text(prompt)
            score = self._score_single(image, target)
            scores.append(score)

        return RewardOutput(scores=torch.tensor(scores, dtype=torch.float32))


@register_reward("ocr_exact")
class OCRExactReward(OCRReward):
    """OCR reward scored by exact match rather than edit distance.

    The default reward normalizes edit distance by target length, which makes
    longer targets *more* forgiving: one wrong character costs 1/3 on "CAT" but
    1/12 on "PURPLE ORCHID" (spaces are stripped before the division). That is why using harder prompts to create headroom
    does not work - FLUX.1-dev scores 0.792 on short words and still 0.691 on
    two-word phrases, because partial credit absorbs the extra difficulty.

    Exact match removes the partial credit, so the score reflects whether the
    model actually rendered the string. Sparse, but sparse is what leaves room to
    climb on a model that is already good at text.
    """

    name = "ocr_exact"

    def _score_single(self, image, target: str) -> float:
        if not target:
            return 0.0
        recognized = self._ocr_image(image)
        recognized_norm = recognized.replace(" ", "").lower()
        target_norm = target.replace(" ", "").lower()
        if not target_norm:
            return 0.0
        # Equality, which is what "exact match" means and what this class is named
        # for. It previously returned `target_norm in recognized_norm` - bare
        # containment with no guard at all, so "cat" scored 1.0 inside
        # "concatenate", and it was *more* permissive than the edit-distance
        # reward it claims to be stricter than. It was also hackable in the
        # direction RL exploits: rendering more text raises the chance of
        # containing the target.
        #
        # Every FLUX result recorded in VALIDATION.md used the containment
        # version. Those comparisons stay internally valid - every arm was scored
        # by the same function - but the absolute numbers meant "the target
        # appears somewhere in the image's text", not "the model rendered the
        # target". Numbers from before this change are not comparable with
        # numbers after it.
        return 1.0 if recognized_norm == target_norm else 0.0

