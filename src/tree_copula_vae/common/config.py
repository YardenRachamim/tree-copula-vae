"""Common configuration dataclasses shared across all datasets."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BaseTrainingParamsConfig:
    """Base training hyperparameters."""
    max_epochs: int = 100
    check_val_every_n_epoch: int = 1
    batch_size: int = 128
    hidden_dim: int = 4
    beta: float = 1.0
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.95, 0.999)
    name: str = 'Default'
    logger_save_dir: str = "./experiments/"
    seed: int = 42


@dataclass
class KLAnnealingCallbackConfig:
    """KL annealing callback configuration."""
    warmup_epochs: int = 10
    max_beta: float = 1.0


@dataclass
class AnnealAttributeCallbackConfig:
    """Attribute annealing callback configuration (temperature, learning rate, etc.)."""
    start_value: float = 2.0
    end_value: float = 0.1
    mode: str = "cosine"
    warmup_frac: Optional[float] = None  # Optional for two_phase mode


@dataclass
class BaseDataConfig:
    """Base data module configuration."""
    data_dir: str = './data/'
    batch_size: int = 128
    pin_memory: bool = True
    num_workers: int = 4


@dataclass
class BaseTrainerConfig:
    """Base Lightning Trainer configuration."""
    accelerator: str = "gpu"
    devices: List[int] = field(default_factory=lambda: [0])
    precision: int = 32
    gradient_clip_val: float = 0.5
    deterministic: bool = False
    log_every_n_steps: int = 50
    min_epochs: int = 1
    detect_anomaly: bool = False
