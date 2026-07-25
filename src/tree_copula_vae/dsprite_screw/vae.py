import math
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torch.distributions as D
from torch.distributions import Independent, Normal, kl_divergence
import zuko

from tree_copula_vae.common.vae import EvaluationMetricResults, LoggingNames, get_iwae_from_log_weights
from tree_copula_vae.dsprite_screw.decoder import GaussianCopulaDecoder, GaussianMFDecoder
from tree_copula_vae.dsprite_screw.encoder import StrongEncoder
from tree_copula_vae.torch_copulas.base import PairCopula
from tree_copula_vae.torch_copulas.graphical_structures import E_TreeStructureBackEndTypes, CopulaFullGraphStructure, CopulaTreeStructure
from tree_copula_vae.torch_copulas.multivariate_copulas import TreeAveragedCopula, TreeCopula
from tree_copula_vae.torch_copulas.multivariate_distributions import MultivariateDistributionUsingCopula
from tree_copula_vae.torch_copulas.pair_copulas import BiVariateGaussianCopula
from tree_copula_vae.utils.graph_utils import sample_soft_tree



class CopulaVAE(pl.LightningModule):
    @property
    def logging_names(self) -> LoggingNames:
        return self._logging_names

    def __init__(
            self,
            latent_dim=3,
            lr=1e-3,
            kl_coeff=1.0,
            K_eval= 50,
            K_test=512,
            pair_copula_class: Type[PairCopula] = BiVariateGaussianCopula,
            use_nf_prior: bool = False,
            use_copula_prior: bool = False,
            start_temperature: float = 1.,
            inject_noise: bool = False,
            use_copula_decoder: bool = False,
            decoder_rank: int = 1
    ):
        super().__init__()
        self.save_hyperparameters()
        if use_nf_prior and use_copula_prior:
            raise ValueError

        if decoder_rank == 0 and use_copula_decoder:
            raise ValueError

        self.register_buffer("temperature", torch.tensor(start_temperature))

        # # # 1. Backbones (the "slim" architecture we selected)
        # # # hidden_dim=256 matches the output of LinearHeadEncoder
        # self.encoder_backbone = LinearHeadEncoder(input_channels=1, hidden_dim=256)
        # self.decoder = LinearHeadDecoder(latent_dim=latent_dim, output_channels=1)

        self.encoder_backbone = StrongEncoder(input_channels=1, hidden_dim=256)
        if use_copula_decoder:
            self.decoder = GaussianCopulaDecoder(latent_dim=latent_dim, output_channels=1, rank=decoder_rank)
        else:
            self.decoder = GaussianMFDecoder(latent_dim=latent_dim, output_channels=1)

        # 2. Projection Heads (Mean Field Baseline)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.n_edges = (latent_dim ** 2  - latent_dim) // 2
        self.fc_copula_params = nn.Linear(256, self.n_edges)

        self._precision = 1e-6  # Add the missing variable
        self._tree_structure_backend = E_TreeStructureBackEndTypes.dgl
        self._logging_names = LoggingNames()

        self._pair_copula_class = pair_copula_class
        self.inject_noise = inject_noise
        self.use_nf_prior = use_nf_prior
        self.use_copula_prior = use_copula_prior

        if use_nf_prior:
            # self.nf_prior = zuko.flows.MAF(
            #     features=latent_dim,
            #     transforms=3,
            #     hidden_features=(256, 256),
            # )
            self.nf_prior = zuko.flows.NSF(
                features=latent_dim,  # 3
                transforms=2,  # Two transformations are enough for this dimension
                hidden_features=(32, 32),  # A compact network, not 256 units
                bins=8  # Number of spline bins
            )
        elif use_copula_prior:
            self.raw_copula_params = nn.Parameter(torch.empty(self.n_edges))
            self.raw_edge_logits = nn.Parameter(torch.empty(self.n_edges))
            nn.init.normal_(self.raw_edge_logits, mean=0.0, std=1e-3)
            # Initialize with tiny noise (standard deviation 0.001).
            # This creates negligible correlations of approximately 0.0001 while breaking symmetry.
            nn.init.normal_(self.raw_copula_params, mean=0.0, std=1e-3)


    def variational_distribution(self, x):
        """
        Returns the q(z|x) distribution.
        The copula is applied here.
        """
        # 1. Extract image features
        h = self.encoder_backbone(x)

        # 2. Compute marginal parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = F.softplus(logvar) + 1e-5

        copula_pair_params_ = self.fc_copula_params(h)
        copula_pair_params = 0.99 * torch.tanh(copula_pair_params_)
        # Avoid copula_pair_params with 0 value since they can cause problems in the weighted calculations in validation step
        # e.g. 1 in the soft_mwst where copula_pair_params is zero can cause copula_pair_params*soft_mwst that don't form a valid tree
        is_zero_value_in_pair_params = copula_pair_params == 0
        copula_pair_params[is_zero_value_in_pair_params] = self._precision

        copula_tree_structure = CopulaTreeStructure.from_copula_pairs_param(
            copula_pair_params=copula_pair_params,
            pair_copula_class=self._pair_copula_class,
            backend=self._tree_structure_backend
        )

        variational_marginal_distributions = Normal(loc=mu, scale=std)
        # variational_marginal_distributions = Logistic(loc=mu, scale=std)

        variational_copula_distribution = TreeCopula(copula_tree_structure=copula_tree_structure, precision=self._precision)
        variational_distribution = MultivariateDistributionUsingCopula(
            marginals=variational_marginal_distributions,
            multivariate_copula=variational_copula_distribution,
            precision=self._precision
        )

        return variational_distribution

    def likelihood_distribution(self, z):
        """
        Returns the p(x|z) distribution.
        """
        return self.decoder.get_distribution(z)

    def prior_distribution(self):
        """
        Returns the p(z) distribution.
        """
        # Create a distribution on the same device as z, with shape (Batch, latent_dim).

        if self.use_nf_prior:
            # p_z = self.prior_module.get_distribution()
            p_z = self.nf_prior()
        elif self.use_copula_prior:
            copula_pair_params = 0.99 * torch.tanh(self.raw_copula_params)
            mu = torch.zeros(self.hparams.latent_dim, device=self.device, dtype=self.dtype)
            scale = torch.ones_like(mu)
            prior_marginals = Normal(loc=mu, scale=scale)
            edge_logits = self.safe_edge_logits(self.raw_edge_logits)
            copula_full_graph_structure = CopulaFullGraphStructure(
                copula_pair_params=copula_pair_params,
                pair_copula_class=self._pair_copula_class,
            )
            prior_copula = TreeAveragedCopula(
                copula_full_graph_structure=copula_full_graph_structure,
                tree_prior_pair_params_in_log=edge_logits,
                precision=self._precision
            )
            p_z = MultivariateDistributionUsingCopula(
                marginals=prior_marginals,
                multivariate_copula=prior_copula,
                precision=self._precision
            )
        else:
            mu = torch.zeros(self.hparams.latent_dim, device=self.device, dtype=self.dtype)
            scale = torch.ones_like(mu)

            p_z = Independent(Normal(loc=mu, scale=scale), reinterpreted_batch_ndims=1)
            # p_z = Independent(Logistic(loc=mu, scale=scale), reinterpreted_batch_ndims=1)

        return p_z

    def forward(self, x):
        # 1. q(z|x)
        q = self.variational_distribution(x)

        # 2. Sample z ~ q(z|x) using Reparameterization Trick
        z = q.rsample()

        # 3. p(x|z)
        p_x_z = self.likelihood_distribution(z)

        return p_x_z, q, z

    def safe_edge_logits(self, edge_logits: torch.Tensor):
        max_val = edge_logits.max(dim=-1, keepdim=True)[0]
        max_shifted_logits = edge_logits - max_val  # max = 0

        # r_min: minimal ratio w_min / w_max after temperature
        r_min = 1e-8
        log_r_min = math.log(r_min)  # < 0, scalar

        min_allowed = self.temperature * log_r_min  # scalar (negative)
        clamped_logits = torch.clamp(max_shifted_logits, min=min_allowed)

        return clamped_logits

    def training_step(self, batch, batch_idx):
        stage = "train"
        x, _ = batch  # Ignore the data latents

        # Forward pass
        res = self(x)
        p_x_z = res[0]
        q: MultivariateDistributionUsingCopula = res[1]
        z = res[2]

        p_z = self.prior_distribution()
        log_likelihood = p_x_z.log_prob(x.view(-1, *p_x_z.event_shape))

        variational_negative_copula_entropy = -self._pair_copula_class(q.copula_distribution.pair_params).entropy()
        edge_logits = variational_negative_copula_entropy.clamp(min=1e-6).log()
        clamped_logits = self.safe_edge_logits(edge_logits)

        soft_mwst, _ = sample_soft_tree(
            edge_logits=clamped_logits,
            temperature=self.temperature,
            num_nodes=self.hparams.latent_dim,
            inject_gumbel_noise=self.inject_noise
        )
        # variational_negative_copula_entropy = variational_negative_copula_entropy * variational_hard_tree
        variational_negative_copula_entropy = variational_negative_copula_entropy * soft_mwst # .detach()
        variational_tree_MI = variational_negative_copula_entropy.sum(-1)

        if self.use_nf_prior:
            marginals_kl = q.marginal_distributions.log_prob(z) - p_z.log_prob(z)
            CE = torch.zeros_like(variational_tree_MI)
        elif self.use_copula_prior:
            marginals_kl = kl_divergence(q.marginal_distributions.base_dist, p_z.marginal_distributions.base_dist).sum(-1)
            v = p_z.marginal_distributions.cdf(z)
            CE = -p_z.copula_distribution.log_prob(v)
        else:
            marginals_kl = kl_divergence(q.marginal_distributions.base_dist, p_z.base_dist).sum(-1)
            CE = torch.zeros_like(variational_tree_MI)

        # Independence copula assumption
        copula_kl = variational_tree_MI + CE
        kl_error = marginals_kl + copula_kl

        # 3. Total Loss (Negative ELBO)
        loss = -(log_likelihood - self.hparams.kl_coeff * kl_error).mean()

        self.log(f"{stage}/{self.logging_names.loss}", loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/{self.logging_names.rec_error}", -log_likelihood.mean(), on_step=False, on_epoch=True)
        self.log(f"{stage}/{self.logging_names.kl_error}", kl_error.mean(), on_step=False, on_epoch=True)

        self.log(f"{stage}/{self.logging_names.kl_error_marginals}", marginals_kl.mean(), on_step=False, on_epoch=True)
        self.log(f"{stage}/{self.logging_names.kl_error_copula}", copula_kl.mean(), on_step=False, on_epoch=True)

        self.log(f"{stage}/variational_tree_MI", variational_tree_MI.mean(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # K=50: a good accuracy-speed tradeoff during training
        results = self._shared_eval(x, K=self.hparams.K_eval)

        # Organized logs with the 'val/' prefix
        self.log(f"val/iwae@{self.hparams.K_eval}", results.iwae, prog_bar=True, on_epoch=True)
        self.log("val/elbo", results.elbo_val, on_epoch=True)
        self.log("val/rec_error", results.rec_error, on_epoch=True)
        self.log("val/kl", results.kl_error, on_epoch=True)

        return results  # Optional, for use by a callback if needed

    # ---------------------------------------------------------
    # TEST STEP
    # ---------------------------------------------------------
    def test_step(self, batch, batch_idx):
        x, _ = batch

        # K=512: maximum accuracy.
        # Mean Field is expected to perform much worse in IWAE here because of "wasted samples."
        results = self._shared_eval(x, K=self.hparams.K_test)

        self.log(f"test/iwae@{self.hparams.K_test}", results.iwae, on_epoch=True)
        self.log("test/elbo", results.elbo_val, on_epoch=True)
        self.log("test/rec_error", results.rec_error, on_epoch=True)
        self.log("test/kl", results.kl_error, on_epoch=True)


    @torch.no_grad()
    def _shared_eval(self, x, K: int) -> EvaluationMetricResults:
        # 1. q(z|x)
        variational_distribution = self.variational_distribution(x)  # Batch shape: [B]

        # 2. Sample z: [K, B, d]
        z = variational_distribution.rsample(torch.Size([K]))

        # 3. p(x|z) - Batch shape: [K, B]
        likelihood_distribution = self.likelihood_distribution(z)

        # 4. p(z) - Batch shape: [B] (Broadcasts automatically to [K, B])
        prior_distribution = self.prior_distribution()

        # --- SAFETY CHECK ---
        # Ensure x is duplicated to match K samples for every image.
        # x shape: [B, C, H, W] -> x_expanded: [K, B, C, H, W]
        x_expanded = x.unsqueeze(0).expand(K, *x.shape)

        # --- LOSS CALCULATION ---
        # All tensors now have shape [K, B].
        log_likelihood = likelihood_distribution.log_prob(
            x_expanded.reshape(*likelihood_distribution.batch_shape, *likelihood_distribution.event_shape)
        ).view(K, *variational_distribution.batch_shape)

        # KL: log q(z|x) - log p(z)
        # prior.log_prob(z) works because z is [K, B, d] and the prior is [B, d], so broadcasting is valid.
        log_prior = prior_distribution.log_prob(z)
        log_variational = variational_distribution.log_prob(z)
        kl_error = log_variational - log_prior

        # Monte Carlo Estimate of ELBO (averaging over K and B)
        elbo_val = (log_likelihood - kl_error).mean()

        # Beta loss (optional for eval, strictly for logging)
        beta_loss_val = -(log_likelihood - kl_error).mean()

        log_weights = log_likelihood + log_prior - log_variational

        # Assumption: this function applies logsumexp over dim=0 (the K dimension).
        iwae = get_iwae_from_log_weights(
            log_weights=log_weights,
            K=K
        ).mean()  # Average over B at the end

        return EvaluationMetricResults(
            iwae=iwae.item(),
            elbo_val=elbo_val.item(),
            beta_loss_val=beta_loss_val.item(),
            rec_error=-log_likelihood.mean().item(),
            kl_error=kl_error.mean().item()
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # ---- helper: freeze θ (decoder + prior) ----
    def freeze_generative(self):
        for p in self.decoder.parameters():
            p.requires_grad = False
        if self.use_nf_prior:
            for p in self.nf_prior.parameters():
                p.requires_grad = False

class MeanFieldVAE(CopulaVAE):
    def __init__(
            self,
            latent_dim=3,
            lr=1e-3,
            kl_coeff=1.0,
            K_eval=50,
            K_test=512,
            use_nf_prior: bool = False,
            use_copula_decoder: bool = False,
            decoder_rank: int = 1
    ):
        super().__init__(
            latent_dim, lr, kl_coeff, K_eval, K_test,
            use_nf_prior=use_nf_prior, use_copula_decoder=use_copula_decoder,
            decoder_rank=decoder_rank
        )


    def variational_distribution(self, x):
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # The same stability trick used for the copula
        std = F.softplus(logvar) + 1e-5

        # Independent(Normal) = Diagonal Gaussian = Mean Field
        # reinterpreted_batch_ndims=1 makes this a vector of independent variables
        # return Independent(Logistic(loc=mu, scale=std), reinterpreted_batch_ndims=1)
        return Independent(Normal(loc=mu, scale=std), reinterpreted_batch_ndims=1)

    def training_step(self, batch, batch_idx):
        stage = "train"
        x, _ = batch

        # 1. Forward
        # Calls our mean-field variational_distribution
        p_x_z, q_z_x, z = self(x)
        p_z = self.prior_distribution()

        # 2. Likelihood
        log_likelihood = p_x_z.log_prob(x.view(-1, *p_x_z.event_shape))

        # 3. KL Divergence (Analytical)
        # kl_error = q_z_x.log_prob(z) - p_z.log_prob(z)
        if self.use_nf_prior:
            kl_error = q_z_x.log_prob(z) - p_z.log_prob(z)
        else:
            kl_error = D.kl_divergence(q_z_x, p_z)

        # 4. Total Loss
        # In Mean Field there are no separate copula_kl and marginals_kl terms; everything is combined.
        loss = -(log_likelihood - self.hparams.kl_coeff * kl_error).mean()

        # Logging
        self.log(f"{stage}/{self.logging_names.loss}", loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/{self.logging_names.rec_error}", -log_likelihood.mean(), on_step=False, on_epoch=True)
        self.log(f"{stage}/{self.logging_names.kl_error}", kl_error.mean(), on_step=False, on_epoch=True)

        return loss