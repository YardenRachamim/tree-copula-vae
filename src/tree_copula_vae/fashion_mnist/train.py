import os
from pathlib import Path
from typing import List, Optional, Tuple

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from tree_copula_vae.fashion_mnist.callbacks import AnnealAttributeCallback, KLAnealingCallback
from tree_copula_vae.fashion_mnist.config import Config, MFVAEConfig, ModelType, VTreeCopulaVAE2Config
from tree_copula_vae.fashion_mnist.data import FashionMNISTDataModule
from tree_copula_vae.fashion_mnist.vae import MF_VAE, VTreeCopulaVAE2


def build_model(config: Config):
    model_config = config.model
    if model_config.model_type == ModelType.VTREE_COPULA_VAE2:
        return VTreeCopulaVAE2(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            learning_rate=model_config.learning_rate,
            kl_coeff=model_config.kl_coeff,
            start_temperature=model_config.start_temperature,
            inject_noise=model_config.inject_noise,
            K_eval=model_config.K_eval,
            K_test=model_config.K_test,
            pair_copula_class=model_config.pair_copula_type.pair_copula_class,
            decoder_rank=model_config.decoder_rank,
            use_soft_mi=model_config.use_soft_mi,
            use_copula_prior=model_config.use_copula_prior,
            learn_copula_prior_tree_prior=model_config.learn_copula_prior_tree_prior,
            learn_prior_marginals=model_config.learn_prior_marginals,
        )
    if model_config.model_type == ModelType.MF_VAE:
        return MF_VAE(
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            learning_rate=model_config.learning_rate,
            kl_coeff=model_config.kl_coeff,
            K_eval=model_config.K_eval,
            K_test=model_config.K_test,
            decoder_rank=model_config.decoder_rank,
            learn_prior_marginals=model_config.learn_prior_marginals,
        )
    raise ValueError("Unsupported model type: {}".format(model_config.model_type))


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
    kl_annealing = config.callbacks.kl_annealing
    anneal_attribute = config.callbacks.anneal_attribute
    model_checkpoint = config.callbacks.model_checkpoint
    callbacks = [
        KLAnealingCallback(
            warmup_epochs=kl_annealing.warmup_epochs,
            max_beta=kl_annealing.max_beta,
        ),
        AnnealAttributeCallback(
            attribute_to_schedule="temperature",
            num_epochs=config.training_params.max_epochs,
            start_value=anneal_attribute.start_value,
            end_value=anneal_attribute.end_value,
            mode=anneal_attribute.mode,
            log_key="train/temperature",
            warmup_frac=anneal_attribute.warmup_frac,
        ),
    ]
    if config.callbacks.learning_rate_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best",
            monitor=model_checkpoint.monitor,
            mode=model_checkpoint.mode,
            save_top_k=model_checkpoint.save_top_k,
            save_last=model_checkpoint.save_last,
        )
    )
    return callbacks


def save_config_to_wandb(logger: WandbLogger, config_path: Path) -> None:
    logger.experiment.save(
        str(config_path),
        base_path=str(config_path.parent),
        policy="now",
    )


def run(config: Config, config_path: Path) -> None:
    seed_everything(config.training_params.seed, workers=True)
    datamodule = build_datamodule(config)
    model = build_model(config)
    checkpoint_path, resume = _resume_state(config)

    if config.checkpoint.resume_training:
        model = model.__class__.load_from_checkpoint(checkpoint_path, map_location="cpu")

    experiment_name = "{} FASHION_MNIST HD{} {} {}".format(
        model.__class__.__name__,
        config.training_params.hidden_dim,
        config.training_params.name,
        config_path.stem,
    )
    logger = WandbLogger(
        name=experiment_name,
        save_dir=config.training_params.logger_save_dir,
        version=config.checkpoint.run_id,
        resume=resume,
    )
    save_config_to_wandb(logger, config_path)
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