import math
from typing import Type
import wandb
import torch
import torch.distributions as dist
import torch.nn as nn
import zuko
from torch import optim
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger


from tree_copula_vae.common.vae import EvaluationMetricResults, SharedVae,  get_iwae_from_log_weights, LoggingNames
from tree_copula_vae.fashion_mnist.decoder import GaussianCopulaDecoder, SimpleGaussianDecoder
from tree_copula_vae.fashion_mnist.encoder import MNISTEncoderBackbone
from tree_copula_vae.torch_copulas.base import PairCopula
from tree_copula_vae.torch_copulas.graphical_structures import CopulaTreeStructure, E_TreeStructureBackEndTypes, CopulaFullGraphStructure
from tree_copula_vae.torch_copulas.multivariate_copulas import TreeCopula, TreeAveragedCopula
from tree_copula_vae.torch_copulas.multivariate_distributions import MultivariateDistributionUsingCopula
from tree_copula_vae.torch_copulas.pair_copulas import BiVariateGaussianCopula
from tree_copula_vae.utils.graph_utils import sample_soft_tree, sample_hard_tree

    
class VTreeCopulaVAE2(SharedVae):
    def __init__(
        self, 
        input_dim=28*28, 
        hidden_dim=4, 
        learning_rate=0.001, 
        
        # --- New Hyperparameters ---
        kl_coeff: float = 1.0,           # Beta-VAE parameter
        start_temperature: float = 1.0,  # For Gumbel-Softmax tree sampling
        inject_noise: bool = False,       # Noise for structure exploration
        K_train: int = 1, 
        K_eval: int = 50,                # IWAE samples for validation
        K_test: int = 512,               # IWAE samples for test
        precision: float = 1e-6,
        pair_copula_class: Type[PairCopula] = BiVariateGaussianCopula,
        prior_pair_copula_class: Type[PairCopula] = None,
        tree_structure_backend: E_TreeStructureBackEndTypes = E_TreeStructureBackEndTypes.dgl,
        decoder_rank: int = 0,
        use_nf_prior:  bool = False,
        use_soft_mi: bool = True,
        use_copula_prior: bool = False,
        learn_copula_prior_tree_prior: bool = True,
        learn_prior_marginals: bool = False
    ):
        super().__init__(input_dim, hidden_dim, learning_rate, K_train, K_eval, K_test, precision)
        self.logging_names = LoggingNames()
        self.save_hyperparameters() # Pytorch Lightning Magic

        self.use_soft_mi = use_soft_mi
        if prior_pair_copula_class is None:
            self.hparams.prior_pair_copula_class = pair_copula_class
            prior_pair_copula_class = pair_copula_class

        self._tree_structure_backend = tree_structure_backend
        self._precision = precision
        self._pair_copula_class = pair_copula_class
        
        # State variables
        # self.temperature = start_temperature
        self.register_buffer("temperature", torch.tensor(start_temperature))
        self.inject_noise = inject_noise
        
        self.n_edges = (self.hidden_dim ** 2 - self.hidden_dim) // 2

        # --- 1. Encoder (Backbone + Heads) ---
        self.encoder_backbone = MNISTEncoderBackbone(hidden_dim=256)
        
        self.head_mu = nn.Linear(256, self.hidden_dim)
        self.head_logvar = nn.Linear(256, self.hidden_dim)

        self.head_copula_params = nn.Linear(256, self.n_edges)
        nn.init.normal_(self.head_copula_params.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.head_copula_params.bias, mean=0.0, std=1e-3)  # or zeros_ if you prefer

        # --- 2. Decoder ---
        # Use our 'simple' decoder (important for experiemnt)
        if decoder_rank == 0:
            self.decoder = SimpleGaussianDecoder(latent_dim=self.hidden_dim)
        else:
            self.decoder = GaussianCopulaDecoder(latent_dim=self.hidden_dim, rank=decoder_rank)

        if use_copula_prior:
            self.raw_copula_params = nn.Parameter(torch.empty(self.n_edges))

            ######################################
            if learn_copula_prior_tree_prior:
                self.raw_edge_logits = nn.Parameter(torch.empty(self.n_edges))
            else:
                self.raw_edge_logits = torch.zeros(self.n_edges, device=self.device, dtype=self.dtype)
            ######################################

            nn.init.normal_(self.raw_edge_logits, mean=0.0, std=1e-3)
            # Initialize with tiny noise (standard deviation of 0.001)
            # This creates correlations of ~0.0001 (negligible), but breaks symmetry
            nn.init.normal_(self.raw_copula_params, mean=0.0, std=1e-3)

        if learn_prior_marginals:
            self.raw_loc = nn.Parameter(torch.empty(self.hidden_dim))
            self.raw_scale = nn.Parameter(torch.empty(self.hidden_dim))
            nn.init.normal_(self.raw_loc, mean=0.0, std=1e-3)
            nn.init.normal_(self.raw_scale, mean=0.0, std=1e-3)

        self.learn_prior_marginals = learn_prior_marginals

        # Good: Register as buffer (saved in state_dict but not updated by optimizer)
        self.register_buffer("kl_coeff", torch.tensor(kl_coeff))

    def forward(self, x):
        # 1. q(z|x)
        variational_distribution = self.variational_distribution(x)
        
        # 2. Sample z
        z = variational_distribution.rsample()

        # 3. p(x|z)
        likelihood_distribution = self.likelihood_distribution(z)

        return z, variational_distribution, likelihood_distribution

    def variational_distribution(self, x, ret_tree_params: bool = False) -> dist.Distribution:
        # Fix image dimensions
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)

        # A. Feature extraction
        h = self.encoder_backbone(x)
        
        # B. Marginals
        mu = self.head_mu(h)
        log_var = self.head_logvar(h)
        std = torch.nn.functional.softplus(log_var) + 1e-5 # Softplus for stability like in reference code
        
        # C. Copula parameters
        copula_pair_params_logits = self.head_copula_params(h)
        
        # Use factor 0.99 to avoid absolute 1 or -1 (TanH saturation)
        copula_pair_params = 0.99 * torch.tanh(copula_pair_params_logits)
        
        # Handle zeros
        is_zero = copula_pair_params == 0
        copula_pair_params[is_zero] = self._precision


        variational_negative_copula_entropy = -self._pair_copula_class(copula_pair_params).entropy()
        edge_logits = variational_negative_copula_entropy.clamp(min=1e-6).log()


        if self.training:
            soft_mwst, _ = sample_soft_tree(
                edge_logits=edge_logits,
                temperature=self.temperature,
                inject_gumbel_noise=self.inject_noise
            )
            hard_mwst, _ = sample_hard_tree(
                edge_logits=soft_mwst,
                inject_gumbel_noise=False
            )
        else:
            hard_mwst, _ = sample_hard_tree(
                edge_logits=edge_logits,
                inject_gumbel_noise=False
            )
            soft_mwst = hard_mwst

        # D. Build tree structure (Hard Tree for forward pass object)
        # Note: In training_step we do Soft Tree manually for KL calculation
        copula_tree_structure = CopulaTreeStructure.from_copula_pairs_param(
            copula_pair_params=copula_pair_params,
            pair_copula_class=self._pair_copula_class,
            backend=self._tree_structure_backend,
            given_trees=hard_mwst
        )
        
        variational_marginal_distributions = dist.Normal(loc=mu, scale=std)
        
        variational_copula_distribution = TreeCopula(
            copula_tree_structure=copula_tree_structure, 
            precision=self._precision
        )
        
        variational_distribution = MultivariateDistributionUsingCopula(
            marginals=variational_marginal_distributions,
            multivariate_copula=variational_copula_distribution,
            precision=self._precision
        )

        if ret_tree_params:
            return variational_distribution, soft_mwst, hard_mwst

        return variational_distribution

    def likelihood_distribution(self, z) -> dist.Distribution:
        return self.decoder.get_distribution(z)

    def prior_distribution(self):
        """
        Returns the prior distribution p(z).
        Base assumption: Standard Gaussian (Factorized Standard Gaussian).
        """
        # We create a distribution on the same device as z, with shape (Batch, latent_dim)
        if self.learn_prior_marginals:
            mu = self.raw_loc
            scale = torch.nn.functional.softplus(self.raw_scale) + 1e-4
        else:
            mu = torch.zeros(self.hparams.hidden_dim, device=self.device, dtype=self.dtype)
            scale = torch.ones_like(mu)

        prior_marginals = dist.Normal(loc=mu, scale=scale)

        if self.hparams.use_copula_prior:
            copula_pair_params = 0.99 * torch.tanh(self.raw_copula_params)

            # edge_logits = safe_edge_logits(self.raw_edge_logits.double())
            copula_full_graph_structure = CopulaFullGraphStructure(
                copula_pair_params=copula_pair_params,
                pair_copula_class=self.hparams.prior_pair_copula_class,
            )

            prior_copula = TreeAveragedCopula(
                copula_full_graph_structure=copula_full_graph_structure,
                tree_prior_pair_params_in_log=self.raw_edge_logits.to(device=self.device, dtype=self.dtype),
                precision=self._precision
            )
            p_z = MultivariateDistributionUsingCopula(
                marginals=prior_marginals,
                multivariate_copula=prior_copula,
                precision=self._precision
            )
        else:
            p_z = dist.Independent(prior_marginals, reinterpreted_batch_ndims=1)

        return p_z

    # --- Training Step with Soft Tree Logic ---
    def training_step(self, batch, batch_idx):
        stage = "train"
        x, _ = batch
        # x comes as [B, 1, 28, 28]

        # 1. Forward
        # We need the objects themselves to extract parameters
        variational_dist, soft_mwst, hard_mwst = self.variational_distribution(x, ret_tree_params=True)
        z = variational_dist.rsample()
        likelihood_dist = self.likelihood_distribution(z)
        
        p_z = self.prior_distribution()

        # 2. Reconstruction Loss
        # Note: Flattening the event shape for log_prob
        log_likelihood = likelihood_dist.log_prob(x.view(-1, *likelihood_dist.event_shape))

        # 3. KL Calculation (The Copula Way)

        # A. Extract parameters from variational distribution
        # We need to access raw parameters to calculate entropy
        q_copula = variational_dist.copula_distribution
        pair_params = q_copula.pair_params 
        
        # B. Compute edge weights (Negative Entropy)
        # This is essentially the MI of each variable pair
        variational_negative_copula_entropy = -self._pair_copula_class(pair_params).entropy()
        if self.use_soft_mi:
            variational_tree_MI = (variational_negative_copula_entropy * soft_mwst).sum(-1)
        else:
            variational_tree_MI = (variational_negative_copula_entropy * hard_mwst).sum(-1)

        # Note: If prior is not copula, CE is 0 with respect to dependencies.
        if self.hparams.use_copula_prior:
            # KL marginals (easy to compute since both are Gaussian)
            marginals_kl = dist.kl_divergence(
                variational_dist.marginal_distributions.base_dist, 
                p_z.marginal_distributions.base_dist
            ).sum(-1)
            # Here we would need to compute CE between variational copula and prior
            # But for simplicity, assume it's 0 for now
            v = p_z.marginal_distributions.cdf(z)
            CE = -p_z.copula_distribution.log_prob(v)
        else:
            # KL marginals (easy to compute since both are Gaussian)
            marginals_kl = dist.kl_divergence(
                variational_dist.marginal_distributions.base_dist, 
                p_z.base_dist
            ).sum(-1)
            CE = torch.zeros_like(variational_tree_MI)
        
        copula_kl = variational_tree_MI + CE
        kl_error = marginals_kl + copula_kl

        # 4. Total Loss
        loss = -(log_likelihood - self.hparams.kl_coeff * kl_error).mean()

        # Logging

        self.log(f"{stage}/{self.logging_names.loss}", loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/{self.logging_names.rec_error}", -log_likelihood.mean(), on_step=False, on_epoch=True)
        self.log(f"{stage}/{self.logging_names.kl_error}", kl_error.mean(), on_step=False, on_epoch=True)

        self.log(f"{stage}/{self.logging_names.kl_error_marginals}", marginals_kl.mean(), on_step=False, on_epoch=True)
        self.log(f"{stage}/{self.logging_names.kl_error_copula}", copula_kl.mean(), on_step=False, on_epoch=True)

        self.log(f"{stage}/variational_tree_MI", variational_tree_MI.mean(), on_step=False, on_epoch=True)
        self.log(f"{stage}/CE", CE.mean(), on_step=False, on_epoch=True)

        return {"loss": loss}

    # --- Validation & Test (IWAE) ---
    def validation_step(self, batch, batch_idx):
        x, _ = batch

        variational_dist = self.variational_distribution(x)
        if isinstance(variational_dist, MultivariateDistributionUsingCopula):
            q_copula = variational_dist.copula_distribution
            pair_params = q_copula.pair_params
            variational_negative_copula_entropy = -self._pair_copula_class(pair_params).entropy()
            # Note: This is calculated under model.eval() automatically
            edge_logits = variational_negative_copula_entropy.clamp(min=1e-6).log()
            soft_mwst, _ = sample_hard_tree(
                edge_logits=edge_logits,
                inject_gumbel_noise=False  # In validation we usually don't inject noise, but if we do it will be high
            )
            val_tree_mi = (variational_negative_copula_entropy * soft_mwst).sum(-1).mean()
            # Log this as a separate plot in WandB
            self.log("val/variational_tree_MI", val_tree_mi, on_step=False, on_epoch=True)

        results = self._shared_eval(x, K=self.hparams.K_eval)
        
        self.log(f"val/iwae@{self.hparams.K_eval}", results.iwae, prog_bar=True, on_epoch=True)
        self.log("val/elbo", results.elbo_val, on_epoch=True)
        self.log("val/rec_error", results.rec_error, on_epoch=True)
        self.log("val/kl", results.kl_error, on_epoch=True)
        return results

    def test_step(self, batch, batch_idx):
        x, _ = batch
        results = self._shared_eval(x, K=self.hparams.K_test)
        
        self.log(f"test/iwae@{self.hparams.K_test}", results.iwae, on_epoch=True)
        self.log("test/elbo", results.elbo_val, on_epoch=True)
        self.log("test/rec_error", results.rec_error, on_epoch=True)
        self.log("test/kl", results.kl_error, on_epoch=True)

    @torch.no_grad()
    def _shared_eval(self, x, K: int) -> EvaluationMetricResults:
        # 1. q(z|x)
        variational_dist = self.variational_distribution(x)
        
        # 2. Sample K: [K, B, d]
        z = variational_dist.rsample(torch.Size([K]))
        
        # 3. p(x|z)
        likelihood_dist = self.likelihood_distribution(z)
        prior_dist = self.prior_distribution()

        # 4. Calc Log Probs (ADAPTED)
        
        # Expand x: [K, B, C, H, W]
        x_expanded = x.unsqueeze(0).expand(K, *x.shape)
        
        # Handle shape mismatch for LowRank vs Independent
        target_shape = likelihood_dist.event_shape # [784] or [1, 28, 28]
        
        # Flatten K and B dims together AND reshape image to target shape
        # x_expanded shape is [K, B, 1, 28, 28] -> flatten to [K*B, *target_shape]
        flat_x_target = x_expanded.reshape(-1, *target_shape)
        
        # log_prob will return [K*B]
        log_likelihood_flat = likelihood_dist.log_prob(flat_x_target)
        
        # Reshape back to [K, B]
        log_likelihood = log_likelihood_flat.view(K, -1)

        log_prior = prior_dist.log_prob(z)
        log_variational = variational_dist.log_prob(z)

        # IWAE
        log_weights = log_likelihood + log_prior - log_variational
        iwae = get_iwae_from_log_weights(log_weights, K=K).mean()
        elbo = log_weights.mean()

        return EvaluationMetricResults(
            iwae=iwae.item(),
            elbo_val=elbo.item(),
            rec_error=-log_likelihood.mean().item(),
            kl_error=(log_variational - log_prior).mean().item(),
            beta_loss_val=-elbo.item()
        )
    

class MF_VAE(VTreeCopulaVAE2):
    def variational_distribution(self, x):
        # Fix image dimensions
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)

        # A. Feature extraction
        h = self.encoder_backbone(x)
        
        # B. Marginals
        mu = self.head_mu(h)
        log_var = self.head_logvar(h)
        std = torch.nn.functional.softplus(log_var) + 1e-5 # Softplus for stability like in the reference code

        variational_distribution = dist.Independent(dist.Normal(loc=mu, scale=std), reinterpreted_batch_ndims=1)
        
        
        return variational_distribution
    
        # --- Training Step with Soft Tree Logic ---
    def training_step(self, batch, batch_idx):
        stage = "train"
        x, _ = batch
        # x comes as [B, 1, 28, 28]

        # 1. Forward
        # We need the objects themselves to extract parameters
        variational_dist = self.variational_distribution(x)
        z = variational_dist.rsample()
        likelihood_dist = self.likelihood_distribution(z)
        
        p_z = self.prior_distribution()

        # 2. Reconstruction Loss
    # Note: Flatten the event shape for log_prob
        log_likelihood = likelihood_dist.log_prob(x.view(-1, *likelihood_dist.event_shape))

    # KL marginals (easy to compute because both are Gaussian)
        kl_error = dist.kl_divergence(
            variational_dist.base_dist, 
            p_z.base_dist
        ).sum(-1)

        # 4. Total Loss
        loss = -(log_likelihood - self.hparams.kl_coeff * kl_error).mean()

        # Logging

        self.log(f"{stage}/{self.logging_names.loss}", loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/{self.logging_names.rec_error}", -log_likelihood.mean(), on_step=False, on_epoch=True)
        self.log(f"{stage}/{self.logging_names.kl_error}", kl_error.mean(), on_step=False, on_epoch=True)


        return {"loss": loss}