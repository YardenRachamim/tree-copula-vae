"""Create a W&B likelihood-gap comparison for dSprite Screw Copula and MF VAEs."""

import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pyrallis
import torch
import wandb
from lightning.pytorch import seed_everything

from tree_copula_vae.dsprite_screw.config import (
    Config,
    ComparisonSourceRunConfig,
    DspriteScrewComparisonConfig,
    ModelType,
)
from tree_copula_vae.dsprite_screw.train import build_datamodule, build_model


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "dsprite_screw_presets"
    / "compare_runs.yml"
)


def run_path(source: ComparisonSourceRunConfig) -> str:
    return "{}/{}/{}".format(source.entity, source.project, source.run_id)


def download_training_config(source: ComparisonSourceRunConfig, output_dir: Path) -> Path:
    source_path = run_path(source)
    try:
        source_run = wandb.Api().run(source_path)
    except Exception as error:
        raise RuntimeError("Could not access W&B run '{}'.".format(source_path)) from error

    yaml_files = [
        run_file
        for run_file in source_run.files()
        if run_file.name.lower().endswith((".yml", ".yaml"))
    ]
    if source.config_filename:
        candidates = [run_file for run_file in yaml_files if run_file.name == source.config_filename]
    else:
        candidates = [
            run_file
            for run_file in yaml_files
            if Path(run_file.name).name.lower() not in {"config.yml", "config.yaml"}
        ]

    if len(candidates) != 1:
        names = ", ".join(run_file.name for run_file in yaml_files) or "none"
        raise ValueError(
            "Expected one training preset for W&B run '{}', found {}. "
            "Set config_filename when needed. Available YAML files: {}".format(
                source_path,
                len(candidates),
                names,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(candidates[0].download(root=str(output_dir), replace=True).name)


def best_checkpoint_path(config: Config, source: ComparisonSourceRunConfig) -> Path:
    if source.checkpoint_path:
        checkpoint_path = Path(source.checkpoint_path)
    else:
        checkpoint_path = Path(config.checkpoint.ckpt_dir_format.format(source.run_id)).with_name("best.ckpt")
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Best checkpoint not found at '{}'".format(checkpoint_path))
    return checkpoint_path


def load_source_model(
    config: Config,
    source: ComparisonSourceRunConfig,
    expected_model_type: ModelType,
    device: torch.device,
):
    if config.model.model_type != expected_model_type:
        raise ValueError(
            "Run '{}' must use model_type={}, got {}.".format(
                source.run_id,
                expected_model_type.value,
                config.model.model_type.value,
            )
        )
    model = build_model(config)
    checkpoint = torch.load(best_checkpoint_path(config, source), map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def validate_compatible_configs(copula_config: Config, mean_field_config: Config) -> None:
    if copula_config.model.latent_dim != mean_field_config.model.latent_dim:
        raise ValueError("Source models must have matching latent dimensions.")
    if copula_config.data.data_dir != mean_field_config.data.data_dir:
        raise ValueError("Source models must use the same data.data_dir.")
    if copula_config.data.tolerance != mean_field_config.data.tolerance:
        raise ValueError("Source models must use the same dSprite Screw tolerance.")


def posterior_mean(model, x: torch.Tensor) -> torch.Tensor:
    distribution = model.variational_distribution(x)
    if hasattr(distribution, "marginal_distributions"):
        return distribution.marginal_distributions.base_dist.loc
    return distribution.base_dist.loc


def reconstruction_and_pixel_nll(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z = posterior_mean(model, x)
    likelihood = model.likelihood_distribution(z)
    decoder_parameters = model.decoder(z)
    mean, diagonal_scale = decoder_parameters[:2]
    if len(decoder_parameters) == 3:
        factors = decoder_parameters[2]
        diagonal_scale = torch.sqrt(diagonal_scale.pow(2) + factors.pow(2).sum(dim=-1))

    pixel_nll = -torch.distributions.Normal(mean, diagonal_scale).log_prob(x.flatten(start_dim=1))
    return likelihood.mean.view_as(x).clamp(0, 1), pixel_nll.view_as(x)


def likelihood_gap_figure(
    original: torch.Tensor,
    copula_reconstruction: torch.Tensor,
    mean_field_reconstruction: torch.Tensor,
    gap: torch.Tensor,
    title: str,
):
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    images = [original, copula_reconstruction, mean_field_reconstruction]
    titles = ["Original input", "Copula reconstruction", "Mean-field reconstruction"]
    for axis, image, panel_title in zip(axes[:3], images, titles):
        axis.imshow(image.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(panel_title)
        axis.axis("off")

    gap_image = gap.squeeze().cpu().numpy()
    limit = max(float(abs(gap_image).max()), 1e-8)
    plot = axes[3].imshow(gap_image, cmap="bwr", vmin=-limit, vmax=limit)
    axes[3].set_title("Likelihood gap\n(Red = Copula wins)")
    axes[3].axis("off")
    figure.colorbar(plot, ax=axes[3], fraction=0.046, pad=0.04)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


@torch.no_grad()
def create_comparison_images(copula_model, mean_field_model, data_loader, num_examples: int):
    x, _ = next(iter(data_loader))
    device = next(copula_model.parameters()).device
    x = x.to(device)
    copula_reconstruction, copula_nll = reconstruction_and_pixel_nll(copula_model, x)
    mean_field_reconstruction, mean_field_nll = reconstruction_and_pixel_nll(mean_field_model, x)
    gap = mean_field_nll - copula_nll
    sample_gap = gap.flatten(start_dim=1).sum(dim=1)
    number_to_log = min(num_examples, x.size(0))

    ranked_indices = torch.argsort(sample_gap)
    groups = {
        "mean_field_wins": ranked_indices[:number_to_log],
        "copula_wins": ranked_indices[-number_to_log:].flip(0),
    }
    return x, copula_reconstruction, mean_field_reconstruction, gap, sample_gap, groups


def run(comparison_config: DspriteScrewComparisonConfig) -> None:
    temporary_directory = tempfile.TemporaryDirectory() if comparison_config.output_dir is None else None
    output_dir = Path(temporary_directory.name) if temporary_directory else Path(comparison_config.output_dir)
    comparison_run = None
    try:
        copula_config_path = download_training_config(comparison_config.copula, output_dir / "copula")
        mean_field_config_path = download_training_config(comparison_config.mean_field, output_dir / "mean_field")
        copula_config = pyrallis.parse(Config, config_path=str(copula_config_path), args=[])
        mean_field_config = pyrallis.parse(Config, config_path=str(mean_field_config_path), args=[])
        validate_compatible_configs(copula_config, mean_field_config)

        device = torch.device(comparison_config.device)
        seed_everything(copula_config.training_params.seed, workers=True)
        copula_model = load_source_model(copula_config, comparison_config.copula, ModelType.COPULA_VAE, device)
        mean_field_model = load_source_model(
            mean_field_config,
            comparison_config.mean_field,
            ModelType.MEAN_FIELD_VAE,
            device,
        )
        datamodule = build_datamodule(copula_config)
        datamodule.test_batch_size = copula_config.data.batch_size
        datamodule.setup(stage="test")
        results = create_comparison_images(
            copula_model,
            mean_field_model,
            datamodule.test_dataloader(),
            comparison_config.num_examples,
        )
        originals, copula_reconstructions, mean_field_reconstructions, gaps, sample_gaps, groups = results

        comparison_run = wandb.init(
            entity=comparison_config.output_entity,
            project=comparison_config.output_project,
            name="dSprite Screw likelihood gap | Copula {} vs MF {}".format(
                comparison_config.copula.run_id,
                comparison_config.mean_field.run_id,
            ),
            config={
                "copula_run_id": comparison_config.copula.run_id,
                "mean_field_run_id": comparison_config.mean_field.run_id,
                "num_examples": comparison_config.num_examples,
            },
        )
        for group_name, indices in groups.items():
            for rank, index in enumerate(indices.tolist()):
                figure = likelihood_gap_figure(
                    originals[index],
                    copula_reconstructions[index],
                    mean_field_reconstructions[index],
                    gaps[index],
                    "{} | rank {} | gap {:.2f}".format(group_name, rank + 1, sample_gaps[index].item()),
                )
                try:
                    comparison_run.log(
                        {"comparison/{}".format(group_name): wandb.Image(figure)}
                    )
                finally:
                    plt.close(figure)
    finally:
        if comparison_run is not None:
            comparison_run.finish()
        if temporary_directory is not None:
            temporary_directory.cleanup()


if __name__ == "__main__":
    config_path = DEFAULT_CONFIG_PATH
    if len(sys.argv) > 1 and sys.argv[1] == "--config_path":
        config_path = Path(sys.argv[2])
        del sys.argv[1:3]
    run(pyrallis.parse(DspriteScrewComparisonConfig, config_path=str(config_path)))