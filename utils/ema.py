"""Exponential Moving Average (EMA) for model parameters.

Usage:
    ema = EMAModel(model.parameters(), decay=0.9999)

    # training loop
    optimizer.step()
    ema.step(model.parameters())

    # validation: swap to EMA weights
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    validate(model)
    ema.restore(model.parameters())

    # checkpoint
    torch.save(ema.state_dict(), "ema.pt")
    ema.load_state_dict(torch.load("ema.pt"))
"""

import copy
import logging
from typing import Iterable, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EMAModel:
    """Maintains an exponential moving average of model parameters.

    Args:
        parameters: iterable of parameters to track (typically model.parameters()).
        decay: EMA decay rate. Higher = slower update (more smoothing).
        min_decay: minimum decay rate (used during warmup).
        update_after_step: start EMA updates only after this many optimizer steps.
        use_ema_warmup: if True, ramp decay from min_decay to decay over warmup period.
        inv_gamma: inverse multiplicative factor for warmup schedule (default 1.0).
        power: exponential factor for warmup schedule (default 2/3).
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = True,
        inv_gamma: float = 1.0,
        power: float = 2.0 / 3.0,
    ):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.collected_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power

        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay = 1 - (1 + step / self.inv_gamma) ** -self.power
            cur_decay = min(cur_decay, self.decay)
            cur_decay = max(cur_decay, self.min_decay)
        else:
            cur_decay = self.decay

        return cur_decay

    @torch.no_grad()
    def step(self, parameters: Iterable[nn.Parameter]):
        """Update shadow parameters with one EMA step."""
        parameters = list(parameters)
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_((1.0 - decay) * (s_param - param.data))
            else:
                s_param.copy_(param.data)

    def copy_to(self, parameters: Iterable[nn.Parameter]):
        """Copy shadow parameters into model parameters (for inference)."""
        for s_param, param in zip(self.shadow_params, list(parameters)):
            param.data.copy_(s_param)

    def store(self, parameters: Iterable[nn.Parameter]):
        """Store current model parameters for later restore()."""
        self.collected_params = [p.clone() for p in parameters]

    def restore(self, parameters: Iterable[nn.Parameter]):
        """Restore model parameters from previously stored values."""
        if self.collected_params is None:
            raise RuntimeError("No parameters stored. Call store() before restore().")
        for c_param, param in zip(self.collected_params, list(parameters)):
            param.data.copy_(c_param.data)
        self.collected_params = None

    def state_dict(self) -> dict:
        return {
            "shadow_params": [p.clone() for p in self.shadow_params],
            "optimization_step": self.optimization_step,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict: dict):
        self.shadow_params = [p.clone() for p in state_dict["shadow_params"]]
        self.optimization_step = state_dict["optimization_step"]
        if "decay" in state_dict:
            self.decay = state_dict["decay"]

    def to(self, device=None, dtype=None):
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if device or dtype else p
            for p in self.shadow_params
        ]
        if self.collected_params is not None:
            self.collected_params = [
                p.to(device=device, dtype=dtype) if device or dtype else p
                for p in self.collected_params
            ]
        return self
