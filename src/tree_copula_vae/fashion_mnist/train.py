import os
from dataclasses import asdict
from typing import List, Optional, Tuple

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from tree_copula_vae.fashion_mnist.callbacks import AnnealAttributeCallback, KLAnealingCallback
from tree_copula_vae.fashion_mnist.config import Config, MFVAEConfig, ModelType, VTreeCopulaVAE2Config
from tree_copula_vae.fashion_mnist.data import FashionMNISTDataModule
from tree_copula_vae.fashion_mnist.vae import MF_VAE, VTreeCopulaVAE2


def _model_kwargs(config) -> dict:
    kwargs = asdict(config)
    kwargs.pop("model_type")
    return kwargs


def build_model(config: Config):
    if config.model.model_type == ModelType.VTREE_COPULA_VAE2:
        return VTreeCopulaVAE2(**_model_kwargs(config.model))
    if config.model.model_type == ModelType.MF_VAE:
        return MF_VAE(**_model_kwargs(config.model))
    raise ValueError("Unsupported model type: {}".format(config.model.model_type))


def build_datamodule(config: Config) -> FashionMNISTDataModule:
    return FashionMNISTDataModule(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        pin_memory=config.data.pin_memory,
        num_workers=config.data.num_workers,
        observed_distribution_type=config.data.observed_distribution_type,
    )


def _resume_state(config: Config) -> Tuple[Optional[str], Optional[bool]]:
    checkpoint = config.checkpoint
    if checkpoint.resume_training:
        return checkpoint.ckpt_dir_format.format(checkpoint.run_id), True
    if checkpoint.monitor_in_same_experiment:
        return None, True
    return None, None


def checkpoint_directory(config: Config, run_id: str) -> str:
    return os.path.dirname(config.checkpoint.ckpt_dir_format.format(run_id))


def build_callbacks(config: Config, checkpoint_dir: str) -> List:
    callbacks = [
        KLAnealingCallback(**asdict(config.callbacks.kl_annealing)),
        AnnealAttributeCallback(
            attribute_to_schedule="temperature",
            num_epochs=config.training_params.max_epochs,
            log_key="train/temperature",
            **asdict(config.callbacks.anneal_attribute)
        ),
    ]
    if config.callbacks.learning_rate_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best",
            **asdict(config.callbacks.model_checkpoint)
        )
    )
    return callbacks


def run(config: Config) -> None:
    seed_everything(config.training_params.seed, workers=True)
    datamodule = build_datamodule(config)
    model = build_model(config)
    checkpoint_path, resume = _resume_state(config)

    if config.checkpoint.resume_training:
        model = model.__class__.load_from_checkpoint(checkpoint_path, map_location="cpu")

    experiment_name = "{} FASHION_MNIST HD{} {}".format(
        model.__class__.__name__, config.training_params.hidden_dim, config.training_params.name
    )
    logger = WandbLogger(
        name=experiment_name,
        save_dir=config.training_params.logger_save_dir,
        version=config.checkpoint.run_id,
        resume=resume,
    )
    logger.watch(model, log="all", log_freq=10)

    checkpoint_dir = checkpoint_directory(config, logger.version)
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        min_epochs=config.trainer.min_epochs,
        max_epochs=config.training_params.max_epochs,
        deterministic=config.trainer.deterministic,
        check_val_every_n_epoch=config.training_params.check_val_every_n_epoch,
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=logger,
        callbacks=build_callbacks(config, checkpoint_dir),
        precision=config.trainer.precision,
        gradient_clip_val=config.trainer.gradient_clip_val,
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")