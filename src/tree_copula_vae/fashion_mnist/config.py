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
    VTREE_COPULA_VAE2 = "VTreeCopulaVAE2"
    MF_VAE = "MF_VAE"


class E_OBSERVED_DIST_TYPE(str, Enum):
    """Enum for likelihood distribution types."""
    ContinuousBernoulli = "ContinuousBernoulli"
    Bernoulli = "Bernoulli"
    Gaussian = "Gaussian"


ObservedDistributionType = E_OBSERVED_DIST_TYPE


@dataclass
class TrainingParamsConfig(BaseTrainingParamsConfig):
    """Training hyperparameters for Fashion MNIST."""
    max_epochs: int = 100
    batch_size: int = 128
    hidden_dim: int = 4
    lr: float = 1e-3
    name: str = 'FashionMnist'
    logger_save_dir: str = "/data/yarden/experiments/"
    seed: int = 42


@dataclass
class DataConfig(BaseDataConfig):
    """Data module configuration for Fashion MNIST."""
    data_dir: str = '/home/yarden/TreeCopulaNew/data/'
    observed_distribution_type: E_OBSERVED_DIST_TYPE = E_OBSERVED_DIST_TYPE.ContinuousBernoulli


@dataclass
class BaseModelConfig:
    """Base model configuration shared between all architectures."""
    model_type: ModelType
    input_dim: int = 28 * 28
    hidden_dim: int = 4
    learning_rate: float = 1e-3
    kl_coeff: float = 0.0
    K_eval: int = 50
    K_test: int = 512
    decoder_rank: int = 1
    learn_prior_marginals: bool = True


@dataclass
class MFVAEConfig(BaseModelConfig):
    """MF_VAE model configuration."""
    model_type: ModelType = ModelType.MF_VAE
    # MF_VAE only needs the base params


@dataclass
class VTreeCopulaVAE2Config(BaseModelConfig):
    """VTreeCopulaVAE2 model configuration with copula-specific parameters."""
    model_type: ModelType = ModelType.VTREE_COPULA_VAE2
    pair_copula_type: PairCopulaType = PairCopulaType.BiVariateGaussianCopula
    start_temperature: float = 2.0
    use_copula_prior: bool = True
    inject_noise: bool = False
    use_soft_mi: bool = False
    learn_copula_prior_tree_prior: bool = False


@dataclass
class KLAnnealingCallbackConfig(KLAnnealingCallbackConfig):
    """KLAnealingCallback configuration - Fashion MNIST specific."""
    warmup_epochs: int = 25
    max_beta: float = 1.0


@dataclass
class AnnealAttributeCallbackConfig(AnnealAttributeCallbackConfig):
    """AnnealAttributeCallback configuration - Fashion MNIST specific."""
    start_value: float = 2.0
    end_value: float = 0.1
    mode: str = "cosine"


@dataclass
class ModelCheckpointCallbackConfig:
    """ModelCheckpoint callback configuration."""
    monitor: str = "val/elbo"
    mode: str = "max"
    save_top_k: int = 1
    save_last: bool = True


@dataclass
class CallbacksConfig:
    """All callbacks configuration."""
    kl_annealing: KLAnnealingCallbackConfig = field(default_factory=KLAnnealingCallbackConfig)
    anneal_attribute: AnnealAttributeCallbackConfig = field(default_factory=AnnealAttributeCallbackConfig)
    model_checkpoint: ModelCheckpointCallbackConfig = field(default_factory=ModelCheckpointCallbackConfig)
    log_reconstruction_grid: bool = False
    learning_rate_monitor: bool = True


@dataclass
class TrainerConfig(BaseTrainerConfig):
    """Lightning Trainer configuration - Fashion MNIST specific."""
    accelerator: str = "gpu"
    devices: List[int] = field(default_factory=lambda: [3])
    precision: int = 32
    gradient_clip_val: float = 0.5


@dataclass
class CheckpointConfig:
    """Checkpoint management configuration."""
    resume_training: bool = False
    monitor_in_same_experiment: bool = False
    ckpt_dir_format: str = "/data/yarden/experiments/checkpoints/{}/mf_stage1/last.ckpt"
    run_id: Optional[str] = None


@dataclass
class WandbRunConfig:
    """W&B run selection for post-training analysis."""
    entity: Optional[str] = None
    project: Optional[str] = None
    run_id: Optional[str] = None
    config_filename: Optional[str] = None


@dataclass
class TreeAnalysisConfig:
    """Configuration for latent-tree analysis of a trained Fashion-MNIST model."""
    wandb: WandbRunConfig = field(default_factory=WandbRunConfig)
    checkpoint_path: Optional[str] = None
    device: str = "cuda"
    output_dir: Optional[str] = None

    def __post_init__(self):
        missing_fields = [
            field_name
            for field_name in ("entity", "project", "run_id")
            if not getattr(self.wandb, field_name)
        ]
        if missing_fields:
            raise ValueError(
                "Tree analysis requires W&B {}.".format(
                    ", ".join("wandb.{}".format(field_name) for field_name in missing_fields)
                )
            )


@dataclass
class Config:
    """Main configuration dataclass nesting all sub-configs."""
    training_params: TrainingParamsConfig = field(default_factory=TrainingParamsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: Union[MFVAEConfig, VTreeCopulaVAE2Config] = field(default_factory=VTreeCopulaVAE2Config)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Global settings
    eps: float = 1e-6
    train_generative_part: bool = True
    
    def __post_init__(self):
        """Validate configuration consistency."""
        if self.checkpoint.monitor_in_same_experiment and not self.checkpoint.resume_training:
            raise ValueError(
                "Invalid combination of arguments - if 'monitor_in_same_experiment' "
                "then 'resume_training' must be true as well"
            )
        
        # Ensure model type matches config type
        if isinstance(self.model, MFVAEConfig) and self.model.model_type != ModelType.MF_VAE:
            raise ValueError(
                f"MFVAEConfig requires model_type=MF_VAE, got {self.model.model_type}"
            )
        if isinstance(self.model, VTreeCopulaVAE2Config) and self.model.model_type != ModelType.VTREE_COPULA_VAE2:
            raise ValueError(
                f"VTreeCopulaVAE2Config requires model_type=VTREE_COPULA_VAE2, got {self.model.model_type}"
            )
