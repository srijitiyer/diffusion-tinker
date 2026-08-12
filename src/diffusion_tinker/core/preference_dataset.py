"""Preference dataset for DiffusionDPO training.

Wraps a HuggingFace dataset of image preference pairs (winner/loser per prompt).
"""

from __future__ import annotations

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


def _as_image(value) -> Image.Image:
    """Coerce a dataset image column into a PIL image.

    Pick-a-Pic and its mirrors store jpg_0/jpg_1 as raw encoded bytes rather than
    as a decoded image feature, so a loader that assumes PIL fails with
    'bytes' object has no attribute 'mode'. Paths and file objects show up in
    local imagefolder datasets.
    """
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        import io

        return Image.open(io.BytesIO(bytes(value)))
    if isinstance(value, str):
        return Image.open(value)
    if isinstance(value, dict):
        # datasets sometimes hands back {"bytes": ..., "path": ...}
        if value.get("bytes") is not None:
            import io

            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    raise TypeError(f"cannot read an image from {type(value).__name__}")


class PreferenceDataset(Dataset):
    """Dataset of (prompt, winner_image, loser_image) triplets."""

    def __init__(
        self,
        hf_dataset,
        winner_col: str = "jpg_0",
        loser_col: str = "jpg_1",
        caption_col: str = "caption",
        label_col: str | None = "label_0",
        resolution: int = 512,
    ):
        self.dataset = hf_dataset
        self.winner_col = winner_col
        self.loser_col = loser_col
        self.caption_col = caption_col
        self.label_col = label_col
        self.transform = T.Compose(
            [
                T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(resolution),
                T.ToTensor(),
            ]
        )

        # Drop tied pairs. Pick-a-Pic encodes "no preference" as label_0 == 0.5,
        # and Diffusion-DPO filters those out before training. The swap test is
        # `label < 0.5`, which a tie fails, so ties previously trained with jpg_0
        # silently promoted to winner: a fraction of steps pushing the model to
        # prefer an arbitrary image over an equally-rated one. That is zero-mean
        # gradient noise fed into a loss with beta=5000, i.e. the saturating tail
        # of logsigmoid, which is where this trainer is most sensitive.
        self._indices = list(range(len(hf_dataset)))
        if label_col:
            keep = []
            for i in range(len(hf_dataset)):
                try:
                    if abs(float(hf_dataset[i][label_col]) - 0.5) < 1e-6:
                        continue
                except (TypeError, ValueError, KeyError):
                    pass
                keep.append(i)
            dropped = len(self._indices) - len(keep)
            if dropped:
                print(f"PreferenceDataset: dropped {dropped} tied pairs of "
                      f"{len(self._indices)}")
            self._indices = keep

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset[self._indices[idx]]

        winner_img = row[self.winner_col]
        loser_img = row[self.loser_col]

        # Some datasets encode preference via a label column
        # label_0 > 0.5 means jpg_0 is preferred, else jpg_1
        if self.label_col and self.label_col in row:
            try:
                if float(row[self.label_col]) < 0.5:
                    winner_img, loser_img = loser_img, winner_img
            except (TypeError, ValueError):
                pass  # Non-numeric label, assume default ordering

        winner_img = _as_image(winner_img)
        loser_img = _as_image(loser_img)

        if winner_img.mode != "RGB":
            winner_img = winner_img.convert("RGB")
        if loser_img.mode != "RGB":
            loser_img = loser_img.convert("RGB")

        return {
            "winner": self.transform(winner_img),
            "loser": self.transform(loser_img),
            "prompt": row[self.caption_col],
        }


def preference_collate_fn(batch: list[dict]) -> dict:
    """Collate preference triplets into batched tensors."""
    return {
        "winner": torch.stack([b["winner"] for b in batch]),
        "loser": torch.stack([b["loser"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
    }
