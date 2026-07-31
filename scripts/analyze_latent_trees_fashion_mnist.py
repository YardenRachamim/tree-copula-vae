"""Analyze Fashion-MNIST latent tree structures for an existing W&B run."""

import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyrallis
import seaborn as sns
import torch
import wandb
from lightning.pytorch import seed_everything
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from tree_copula_vae.fashion_mnist.config import Config, ModelType, TreeAnalysisConfig
from tree_copula_vae.fashion_mnist.train import build_datamodule, build_model
from tree_copula_vae.utils.graph_utils import make_complete_graph


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "fashion_mnist_presets"
    / "tree_analysis.yml"
)


def download_config(
    run_path: str,
    config_filename: Optional[str],
    output_dir: Path,
) -> Path:
    try:
        run = wandb.Api().run(run_path)
    except Exception as error:
        raise RuntimeError(
            "Could not access W&B run '{}'. Check WANDB_ENTITY, WANDB_PROJECT, "
            "and W&B authentication.".format(run_path)
        ) from error

    yaml_files = [
        run_file
        for run_file in run.files()
        if run_file.name.lower().endswith((".yml", ".yaml"))
    ]
    available_names = [run_file.name for run_file in yaml_files]

    if config_filename is not None:
        matching_files = [run_file for run_file in yaml_files if run_file.name == config_filename]
        if not matching_files:
            raise FileNotFoundError(
                "W&B run '{}' has no YAML file named '{}'. Available YAML files: {}".format(
                    run_path,
                    config_filename,
                    ", ".join(available_names) or "none",
                )
            )
        selected_file = matching_files[0]
    else:
        preset_files = [
            run_file
            for run_file in yaml_files
            if Path(run_file.name).name.lower() not in {"config.yml", "config.yaml"}
        ]
        if len(preset_files) == 1:
            selected_file = preset_files[0]
        elif not preset_files:
            raise FileNotFoundError(
                "W&B run '{}' does not contain an uploaded YAML preset besides W&B's "
                "generated config.yml/config.yaml file.".format(
                    run_path
                )
            )
        else:
            raise ValueError(
                "W&B run '{}' has multiple uploaded YAML presets. Select one with "
                "wandb.config_filename. Available YAML files: {}".format(
                    run_path,
                    ", ".join(available_names),
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(selected_file.download(root=str(output_dir), replace=True).name)


def resume_wandb_run(analysis_config: TreeAnalysisConfig):
    try:
        run = wandb.init(
            entity=analysis_config.wandb.entity,
            project=analysis_config.wandb.project,
            id=analysis_config.wandb.run_id,
            resume="must",
        )
    except Exception as error:
        raise RuntimeError(
            "Could not resume W&B run '{}/{}/{}'. Check the analysis configuration "
            "and W&B authentication.".format(
                analysis_config.wandb.entity,
                analysis_config.wandb.project,
                analysis_config.wandb.run_id,
            )
        ) from error
    return run


def resolve_checkpoint_path(config: Config, run_id: str, override: Optional[Path]) -> Path:
    checkpoint_path = override or Path(config.checkpoint.ckpt_dir_format.format(run_id))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Checkpoint not found at '{}'. Pass --checkpoint-path to use a different local file.".format(
                checkpoint_path
            )
        )
    return checkpoint_path


def load_model(config: Config, checkpoint_path: Path, device: torch.device):
    if config.model.model_type != ModelType.VTREE_COPULA_VAE2:
        raise ValueError(
            "Tree analysis requires model_type=VTREE_COPULA_VAE2, got {}.".format(
                config.model.model_type
            )
        )

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def prufer_sequence(tree_mask: torch.Tensor, hidden_dim: int) -> str:
    edge_indices = torch.nonzero(tree_mask.bool(), as_tuple=False).flatten()
    if len(edge_indices) != hidden_dim - 1:
        raise ValueError(
            "Expected {} selected edges, found {}.".format(hidden_dim - 1, len(edge_indices))
        )

    complete_graph = make_complete_graph(hidden_dim).T
    tree = nx.Graph()
    tree.add_nodes_from(range(hidden_dim))
    tree.add_edges_from(complete_graph[edge_indices].cpu().tolist())
    if not nx.is_tree(tree):
        raise ValueError("Hard tree mask did not form a valid tree.")

    return "-".join(str(node) for node in nx.to_prufer_sequence(tree))


@torch.no_grad()
def collect_tree_assignments(model, data_loader: Iterable, device: torch.device) -> pd.DataFrame:
    tree_sequences: List[str] = []
    labels: List[int] = []

    for x, y in data_loader:
        _, _, hard_trees = model.variational_distribution(x.to(device), ret_tree_params=True)
        for tree_mask, label in zip(hard_trees, y):
            tree_sequences.append(prufer_sequence(tree_mask, model.hidden_dim))
            labels.append(label.item())

    if not tree_sequences:
        raise RuntimeError("No tree assignments were collected from the test dataloader.")
    return pd.DataFrame({"prufer_sequence": tree_sequences, "label": labels})


def calculate_metrics(assignments: pd.DataFrame) -> Dict[str, float]:
    if assignments.empty:
        raise ValueError("Cannot calculate tree metrics from an empty assignment table.")

    counts = assignments.groupby(["prufer_sequence", "label"]).size().unstack(fill_value=0)
    tree_totals = counts.sum(axis=1)
    tree_purity = counts.max(axis=1) / tree_totals
    probabilities = counts.div(tree_totals, axis=0)
    tree_entropy = [entropy(row) for _, row in probabilities.iterrows()]

    return {
        "nmi": normalized_mutual_info_score(assignments["label"], assignments["prufer_sequence"]),
        "ari": adjusted_rand_score(assignments["label"], assignments["prufer_sequence"]),
        "purity": float(np.average(tree_purity, weights=tree_totals)),
        "entropy": float(np.average(tree_entropy, weights=tree_totals)),
    }


def build_heatmaps(assignments: pd.DataFrame) -> Sequence[Tuple[str, plt.Figure]]:
    counts = assignments.groupby(["label", "prufer_sequence"]).size().unstack(fill_value=0)
    figures: List[Tuple[str, plt.Figure]] = []

    figure, axis = plt.subplots(figsize=(16, 10))
    sns.heatmap(counts, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={"label": "Count"}, ax=axis)
    axis.set_xlabel("Tree structure (Prufer sequence)")
    axis.set_ylabel("Class label")
    figure.tight_layout()
    figures.append(("tree_analysis/tree_label_heatmap_counts", figure))

    normalized_counts = counts.div(counts.sum(axis=1), axis=0)
    dominant_class = normalized_counts.idxmax(axis=0)
    normalized_counts = normalized_counts[dominant_class.sort_values().index]

    figure, axis = plt.subplots(figsize=(16, 10))
    sns.heatmap(
        normalized_counts,
        cmap="Blues",
        cbar_kws={"label": "Probability P(tree | class)"},
        vmin=0.0,
        ax=axis,
    )
    axis.set_xlabel("Tree structure (grouped by dominant class)")
    axis.set_ylabel("Class label")
    figure.tight_layout()
    figures.append(("tree_analysis/tree_label_heatmap_normalized", figure))
    return figures


def run(analysis_config: TreeAnalysisConfig) -> None:
    output_dir_context = (
        tempfile.TemporaryDirectory() if analysis_config.output_dir is None else None
    )
    output_dir = Path(output_dir_context.name) if output_dir_context else Path(analysis_config.output_dir)
    wandb_run = None
    try:
        run_path = "{}/{}/{}".format(
            analysis_config.wandb.entity,
            analysis_config.wandb.project,
            analysis_config.wandb.run_id,
        )
        config_path = download_config(
            run_path,
            analysis_config.wandb.config_filename,
            output_dir,
        )
        config = pyrallis.parse(config_class=Config, config_path=str(config_path), args=[])
        checkpoint_path = resolve_checkpoint_path(
            config,
            analysis_config.wandb.run_id,
            Path(analysis_config.checkpoint_path) if analysis_config.checkpoint_path else None,
        )
        device = torch.device(analysis_config.device)

        seed_everything(config.training_params.seed, workers=True)
        model = load_model(config, checkpoint_path, device)
        datamodule = build_datamodule(config)
        datamodule.setup(stage="test")
        assignments = collect_tree_assignments(model, datamodule.test_dataloader(), device)
        metrics = calculate_metrics(assignments)

        wandb_run = resume_wandb_run(analysis_config)
        wandb_run.log({"tree_analysis/{}".format(name): value for name, value in metrics.items()})
        for key, figure in build_heatmaps(assignments):
            try:
                wandb_run.log({key: wandb.Image(figure)})
            finally:
                plt.close(figure)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if output_dir_context is not None:
            output_dir_context.cleanup()


if __name__ == "__main__":
    config_path = DEFAULT_CONFIG_PATH
    if len(sys.argv) > 1 and sys.argv[1] == "--config_path":
        config_path = Path(sys.argv[2])
        del sys.argv[1:3]
    run(pyrallis.parse(config_class=TreeAnalysisConfig, config_path=str(config_path)))