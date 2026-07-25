from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from enum import Enum
from tree_copula_vae.common.config import (
    BaseTrainingParamsConfig,
    KLAnnealingCallbackConfig,
    AnnealAttributeCallbackConfig,
    BaseDataConfig,
    BaseTrainerConfig,
)
from tree_copula_vae.common.copulas import PairCopulaType


class ModelType(str, Enum):
    """Enum for selecting between VAE model architectures."""
    COPULA_VAE = "CopulaVAE"
    MEAN_FIELD_VAE = "MeanFieldVAE"


@dataclass
class BaseModelConfig:
    """Base model configuration shared between all architectures."""
    model_type: ModelType = ModelType.MEAN_FIELD_VAE
    latent_dim: int = 3
    learning_rate: float = 0.005
    kl_coeff: float = 0.0
    decoder_rank: int = 0
    use_nf_prior: bool = False
    use_copula_decoder: bool = False


@dataclass
class MeanFieldVAEConfig(BaseModelConfig):
    """MeanFieldVAE model configuration."""
    model_type: ModelType = ModelType.MEAN_FIELD_VAE
    # MeanFieldVAE only needs the base params


@dataclass
class CopulaVAEConfig(BaseModelConfig):
    """CopulaVAE model configuration with copula-specific parameters."""
    model_type: ModelType = ModelType.COPULA_VAE
    pair_copula_type: PairCopulaType = PairCopulaType.BiVariateGaussianCopula
    start_temperature: float = 2.0
    use_copula_prior: bool = False


@dataclass
class TrainingParamsConfig(BaseTrainingParamsConfig):
    """Training hyperparameters for DSprite Screw."""
    seed: int = 1265
    max_epochs: int = 50
    batch_size: int = 128
    hidden_dim: int = 3
    lr: float = 0.005
    name: str = 'ScrewDSprites'
    logger_save_dir: str = "/home/yarden/gpufs/experiments/"


@dataclass
class DataConfig(BaseDataConfig):
    """DSprite Screw data module configuration."""
    data_dir: str = '/home/yarden/TreeCopulaNew/data/'
    tolerance: float = 0.15


@dataclass
class KLAnnealingCallbackConfig(KLAnnealingCallbackConfig):
    """KLAnealingCallback configuration - DSprite Screw specific."""
    warmup_epochs: int = 10
    max_beta: float = 1.0


@dataclass
class AnnealAttributeCallbackConfig(AnnealAttributeCallbackConfig):
    """AnnealAttributeCallback configuration - DSprite Screw specific."""
    start_value: float = 2.0
    end_value: float = 0.1
    mode: str = "two_phase"
    warmup_frac: float = 0.2


@dataclass
class CallbacksConfig:
    """All callbacks configuration."""
    kl_annealing: KLAnnealingCallbackConfig = field(default_factory=KLAnnealingCallbackConfig)
    anneal_attribute: AnnealAttributeCallbackConfig = field(default_factory=AnnealAttributeCallbackConfig)
    learning_rate_monitor: bool = True


@dataclass
class TrainerConfig(BaseTrainerConfig):
    """Lightning Trainer configuration - DSprite Screw specific."""
    accelerator: str = "gpu"
    devices: List[int] = field(default_factory=lambda: [2])
    precision: int = 32
    gradient_clip_val: float = 0.5
    detect_anomaly: bool = False


@dataclass
class CheckpointConfig:
    """Checkpoint management configuration."""
    monitor: str = "val/elbo"
    mode: str = "max"
    save_top_k: int = 1
    save_last: bool = False


@dataclass
class Config:
    """Main configuration dataclass nesting all sub-configs for DSprite Screw."""
    training_params: TrainingParamsConfig = field(default_factory=TrainingParamsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: Union[MeanFieldVAEConfig, CopulaVAEConfig] = field(default_factory=MeanFieldVAEConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    def __post_init__(self):
        """Validate configuration consistency."""
        # Ensure latent_dim matches hidden_dim
        if self.model.latent_dim != self.training_params.hidden_dim:
            raise ValueError(
                f"model.latent_dim ({self.model.latent_dim}) must match "
                f"training_params.hidden_dim ({self.training_params.hidden_dim})"
            )
        
        # Ensure model type matches config type
        if isinstance(self.model, MeanFieldVAEConfig) and self.model.model_type != ModelType.MEAN_FIELD_VAE:
            raise ValueError(
                f"MeanFieldVAEConfig requires model_type=MEAN_FIELD_VAE, got {self.model.model_type}"
            )
        if isinstance(self.model, CopulaVAEConfig) and self.model.model_type != ModelType.COPULA_VAE:
            raise ValueError(
                f"CopulaVAEConfig requires model_type=COPULA_VAE, got {self.model.model_type}"
            )
