# Beyond Mean-Field: Tree-Copula Variational Autoencoders

Official implementation accompanying *Beyond Mean-Field: Tree-Copula Variational Autoencoders for Structured Latent Dependencies* (UAI 2026).

This project studies variational autoencoders whose posterior is a sample-specific tree copula instead of a mean-field Gaussian. Candidate pairwise dependencies are scored by the encoder, a maximum-weight spanning tree (MWST) gives the hard posterior structure, and Matrix-Tree/Kirchhoff marginals provide a differentiable soft-tree surrogate during training. The repository also includes factorized and rank-1 Gaussian-copula observation models, plus scripts for the paper's reconstruction-gap and latent-topology analyses.

## Resources

- [Project page](https://yardenrachamim.github.io/tree-copula-vae/)
- [Paper: Beyond Mean-Field: Tree-Copula Variational Autoencoders for Structured Latent Dependencies](https://proceedings.mlr.press/v337/rachamim26a.html)
- [Official GitHub repository](https://github.com/YardenRachamim/tree-copula-vae/tree/main)

## Repository layout

- [configs](configs): runnable YAML presets for each reported dSprites and Fashion-MNIST model.
- [scripts](scripts): training and post-training analysis entry points.
- [src/tree_copula_vae](src/tree_copula_vae): model, data, copula, and training implementations.
- [tests](tests): focused tests for the copula utilities and training paths.
- [data](data): expected location for local datasets. This directory is intentionally not populated by the package.

## Installation

This project requires Python 3.8.

```bash
git clone https://github.com/YardenRachamim/tree-copula-vae.git
cd tree-copula-vae
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Training and the analysis scripts use Weights & Biases. Authenticate before running commands that log runs, download source configurations, or resume existing runs:

```bash
wandb login
```

## Before training

Download the required public datasets into a local data directory:

- dSprites: `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`
- Fashion-MNIST: use the layout expected by `torchvision.datasets.FashionMNIST` under `<data_dir>/FashionMNIST`

Every preset has environment-specific fields. Copy a preset and update at least these values before launching an experiment:

- `data.data_dir`: directory containing the datasets.
- `trainer.accelerator` and `trainer.devices`: local CPU/GPU configuration.
- `training_params.logger_save_dir` and `checkpoint.ckpt_dir_format`: writable output paths.
- W&B `entity` and `project` in analysis presets when using your own runs.

The shipped presets use GPU acceleration. For CPU-only execution, set `trainer.accelerator: cpu` and remove or adjust `trainer.devices` in the copied YAML file.

## Model naming

Experiment names follow `Prior-Variational-Likelihood` from the paper.

| Component | Code | Meaning |
| --- | --- | --- |
| Prior | `SN` | Standard normal prior |
| Variational posterior | `MF` | Mean-field diagonal Gaussian |
| Variational posterior | `TC_GC` | Tree copula with Gaussian pair copulas |
| Variational posterior | `TC_ST` | Tree copula with Student-t pair copulas |
| Likelihood | `FG` | Factorized Gaussian likelihood |
| Likelihood | `R1GC` | Rank-1 Gaussian-copula likelihood |

The concrete YAML fields encode the same choices: `model.model_type` selects mean-field or tree-copula inference, `model.pair_copula_type` selects Gaussian or Student-t edges, and `model.decoder_rank`/decoder flags select the dependent likelihood.

## Run experiments

All training entry points accept an optional `--config_path <preset.yml>` argument. Run commands from the repository root.

### Correlated dSprites

The dSprites experiment uses a three-dimensional latent space and filters the original dataset into the correlated Screw subset with `data.tolerance: 0.15`.

```bash
python scripts/train_dsprite_screw.py --config_path configs/dsprite_screw_presets/SN-MF-FG.yml
python scripts/train_dsprite_screw.py --config_path configs/dsprite_screw_presets/SN-TC_GC-FG.yml
python scripts/train_dsprite_screw.py --config_path configs/dsprite_screw_presets/SN-TC_ST-FG.yml
python scripts/train_dsprite_screw.py --config_path configs/dsprite_screw_presets/SN-MF-R1GC.yml
python scripts/train_dsprite_screw.py --config_path configs/dsprite_screw_presets/SN-TC_GC-R1GC.yml
python scripts/train_dsprite_screw.py --config_path configs/dsprite_screw_presets/SN-TC_ST-R1GC.yml
```

### Fashion-MNIST

Fashion-MNIST presets use a four-dimensional latent space, uniform dequantization in the data pipeline, a rank-1 decoder, and 512 importance samples for test IWAE estimation.

```bash
python scripts/train_fashion_mnist.py --config_path configs/fashion_mnist_presets/SN-MF-R1GC.yml
python scripts/train_fashion_mnist.py --config_path configs/fashion_mnist_presets/SN-TC_GC-R1GC.yml
python scripts/train_fashion_mnist.py --config_path configs/fashion_mnist_presets/SN-TC_ST-R1GC.yml
```

The default configuration hard-coded in each training script is the respective mean-field R1GC preset. Passing `--config_path` is recommended so the selected experiment is explicit.

## Preset schedules

Both datasets train for 50 epochs with batch size 128 and beta/KL annealing. Tree-copula variants also anneal the soft-MWST temperature from 2.0 toward 0.1.

| Dataset | Latent dimensions | Learning rate | KL warmup | Temperature schedule |
| --- | ---: | ---: | ---: | --- |
| Correlated dSprites | 3 | 5e-3 | 10 epochs | Two-phase, 2.0 to 0.1 |
| Fashion-MNIST | 4 | 1e-3 | 25 epochs | Cosine, 2.0 to 0.1 |

See [configs/dsprite_screw_presets](configs/dsprite_screw_presets) and [configs/fashion_mnist_presets](configs/fashion_mnist_presets) for the complete, editable hyperparameters.

## Post-training analyses

### dSprites per-pixel likelihood gaps

`compare_dsprite_screw_runs.py` loads one tree-copula and one mean-field run from W&B, restores their best checkpoints, and logs the highest and lowest per-example marginal likelihood gaps. Edit `compare_runs.yml` to reference your run IDs, W&B entity/project, optional local checkpoint paths, device, and output directory.

```bash
python scripts/compare_dsprite_screw_runs.py --config_path configs/dsprite_screw_presets/compare_runs.yml
```

The comparison requires matching latent dimensions, dSprites data directories, and Screw tolerances in the two source configurations.

### Fashion-MNIST latent-tree topology

`analyze_latent_trees_fashion_mnist.py` restores a tree-copula W&B run, extracts the hard MWST for each test point, encodes it as a Prufer sequence, and logs NMI, ARI, weighted purity, entropy, and class-conditioned topology heatmaps. Set the target run and optional local checkpoint override in `tree_analysis.yml`.

```bash
python scripts/analyze_latent_trees_fashion_mnist.py --config_path configs/fashion_mnist_presets/tree_analysis.yml
```

This analysis only accepts Fashion-MNIST tree-copula checkpoints (`model_type: VTREE_COPULA_VAE2`).

## Tests

Run the focused test suite with:

```bash
pytest
```

## Citation

See [CITATION.cff](CITATION.cff) for citation metadata.
