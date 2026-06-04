"""Training callbacks for hooking into the training loop."""

from __future__ import annotations

from typing import Any


class TrainerCallback:
    """Base class for trainer callbacks.

    Override any method to hook into that training event.
    All methods receive the trainer instance plus event-specific arguments.
    """

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        pass

    def on_sample_end(self, trainer: Any, epoch: int, trajectory: Any) -> None:
        pass

    def on_evaluate(self, trainer: Any, epoch: int, mean_reward: float) -> None:
        pass

    def on_save(self, trainer: Any, epoch: int, path: str) -> None:
        pass
