"""Exponential moving average of the trainable parameters.

FlowGRPO turns this on for its SD3.5 OCR run (`config.train.ema = True`) and the
DDRL paper's image experiments follow FlowGRPO's settings, so a run without it is
not comparable to their published numbers. The point is not stability during
training - it is that *evaluation is done on the averaged weights*. RL updates on
a 64-trajectory rollout are noisy enough that the live weights wander between
epochs, and the reported curve then measures where the wander happened to land.

Matches flow_grpo/ema.py: decay 0.9, updated every 8 optimizer steps, with the
warmup `min((1 + step) / (10 + step), decay)` that keeps the average from being
dominated by the initialization in the first few steps.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


class ParameterEMA:
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9,
        update_step_interval: int = 8,
    ):
        params = [p for p in parameters if p.requires_grad]
        self.shadow = [p.detach().clone() for p in params]
        self.decay = decay
        self.update_step_interval = update_step_interval
        self._stashed: list[torch.Tensor] | None = None

    def current_decay(self, step: int) -> float:
        # Ramps from 1/10 up to `decay`, so early steps are averaged over what
        # exists rather than being pinned to the starting weights.
        return min((1 + step) / (10 + step), self.decay)

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], step: int) -> None:
        if (step + 1) % self.update_step_interval != 0:
            return
        one_minus = 1.0 - self.current_decay(step)
        for shadow, p in zip(self.shadow, [q for q in parameters if q.requires_grad]):
            shadow.add_(one_minus * (p.detach().to(shadow.device) - shadow))

    @torch.no_grad()
    def swap_in(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Put the averaged weights into the model, stashing the live ones."""
        params = [p for p in parameters if p.requires_grad]
        if self._stashed is not None:
            raise RuntimeError("swap_in called twice without an intervening swap_out")
        self._stashed = [p.detach().clone() for p in params]
        for p, shadow in zip(params, self.shadow):
            p.data.copy_(shadow.to(p.device, dtype=p.dtype))

    @torch.no_grad()
    def swap_out(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Restore the live weights. Training must never continue from the average."""
        if self._stashed is None:
            raise RuntimeError("swap_out called without a preceding swap_in")
        for p, saved in zip([q for q in parameters if q.requires_grad], self._stashed):
            p.data.copy_(saved)
        self._stashed = None

    def state_dict(self) -> dict:
        return {
            "shadow": [t.cpu() for t in self.shadow],
            "decay": self.decay,
            "update_step_interval": self.update_step_interval,
        }

    def load_state_dict(self, state: dict) -> None:
        # Resuming without the shadow would restart the average from the resumed
        # weights, which shows up as a step in the eval curve exactly where a
        # crash happened.
        self.shadow = [t.clone() for t in state["shadow"]]
        self.decay = state.get("decay", self.decay)
        self.update_step_interval = state.get(
            "update_step_interval", self.update_step_interval
        )
