"""OCR reward - text accuracy via edit distance.

Extracts target text from the prompt (between double quotes), runs OCR on the
generated image, and returns 1 - edit_distance / target_length.

Supports EasyOCR (preferred, pip install easyocr) or PaddleOCR as fallback.
"""

from __future__ import annotations

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

        # Try EasyOCR first (works with PyTorch, no conflicts)
        try:
            import easyocr

            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            self._backend = "easyocr"
            return
        except ImportError:
            pass

        # Fall back to PaddleOCR
        try:
            from paddleocr import PaddleOCR

            try:
                self._reader = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
            except (TypeError, ValueError):
                self._reader = PaddleOCR(lang="en")
            self._backend = "paddleocr"
            return
        except ImportError:
            pass

        raise ImportError("OCR reward requires easyocr or paddleocr. Install with: pip install easyocr")

    def _ocr_image(self, image) -> str:
        """Run OCR and return concatenated recognized text."""
        img_array = np.array(image)

        if self._backend == "easyocr":
            results = self._reader.readtext(img_array, detail=0)
            return "".join(results)

        # PaddleOCR
        results = self._reader.ocr(img_array, cls=False)
        recognized = ""
        if results and results[0]:
            for line in results[0]:
                if line and len(line) >= 2 and line[1]:
                    recognized += line[1][0]
        return recognized

    def _score_single(self, image, target: str) -> float:
        if not target:
            return 0.0

        recognized = self._ocr_image(image)

        recognized_norm = recognized.replace(" ", "").lower()
        target_norm = target.replace(" ", "").lower()

        if not target_norm:
            return 0.0

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
