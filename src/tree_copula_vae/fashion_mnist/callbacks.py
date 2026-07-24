import math
from typing import Optional

import torch
from lightning.pytorch import Callback


class KLAnealingCallback(Callback):
    def __init__(self, warmup_epochs: int, max_beta: float):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_beta = max_beta

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup_epochs:
            beta = self.max_beta * trainer.current_epoch / self.warmup_epochs
        else:
            beta = self.max_beta

        pl_module.hparams.kl_coeff = torch.tensor(beta)
        pl_module.log("train/beta", beta, on_epoch=True)


class AnnealAttributeCallback(Callback):
    def __init__(
        self,
        attribute_to_schedule: str,
        num_epochs: int,
        start_value: float,
        end_value: float,
        mode: str,
        log_key: str,
        warmup_frac: Optional[float],
    ):
        super().__init__()
        self.attribute_to_schedule = attribute_to_schedule
        self.num_epochs = num_epochs
        self.start_value = start_value
        self.end_value = end_value
        self.mode = mode
        self.log_key = log_key
        self.warmup_frac = 0.3 if warmup_frac is None else warmup_frac

    def _value(self, epoch: int) -> float:
        progress = min(1.0, max(0.0, epoch / max(1, self.num_epochs)))
        if self.mode == "cosine":
            return self.end_value + 0.5 * (self.start_value - self.end_value) * (
                1.0 + math.cos(math.pi * progress)
            )
        if self.mode == "exp":
            if self.start_value <= 0.0 or self.end_value <= 0.0:
                return self.start_value + (self.end_value - self.start_value) * progress
            return self.start_value * (self.end_value / self.start_value) ** progress
        if self.mode == "two_phase":
            warmup_epochs = int(self.warmup_frac * self.num_epochs)
            if epoch < warmup_epochs:
                return self.start_value
            annealing_progress = (epoch - warmup_epochs) / max(1, self.num_epochs - warmup_epochs)
            if self.start_value <= 0.0 or self.end_value <= 0.0:
                return self.start_value + (self.end_value - self.start_value) * annealing_progress
            return self.start_value * (self.end_value / self.start_value) ** annealing_progress
        raise ValueError("Unsupported annealing mode: {}".format(self.mode))

    def on_train_epoch_start(self, trainer, pl_module):
        value = torch.tensor(self._value(trainer.current_epoch))
        attribute = getattr(pl_module, self.attribute_to_schedule)
        if isinstance(attribute, torch.Tensor):
            attribute.copy_(value.to(device=attribute.device, dtype=attribute.dtype))
        else:
            setattr(pl_module, self.attribute_to_schedule, value)
        pl_module.log(self.log_key, value, on_epoch=True)