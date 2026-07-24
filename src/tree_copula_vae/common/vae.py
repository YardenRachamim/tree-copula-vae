import inspect
from abc import ABC, ABCMeta, abstractmethod
from typing import Union, Type, Dict
from torch.distributions import Bernoulli
import torch
import torch.nn as nn
from torch.optim import Optimizer
from dataclasses import dataclass
import math
from abc import ABC, abstractmethod

import wandb
import torch
import torch.distributions as dist
from torch import optim
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl

from tree_copula_vae.torch_copulas.multivariate_distributions import MultivariateDistributionUsingCopula
from tree_copula_vae.torch_copulas.multivariate_distributions import MultivariateDistributionUsingCopula

@dataclass
class LoggingNames():
    rec_error: str = "rec_error"
    elbo: str = "elbo"
    loss: str = "loss"
    kl_error: str = "kl_error"
    kl_error_marginals: str = "kl_error_marginals"
    kl_error_copula: str = "kl_error_copula"
    kl_error_z: str = 'kl_error_z'
    kl_error_trees: str = 'kl_error_trees'
    iwae: str = "iwae"
    beta_elbo: str = "beta_loss"
    beta: str = "beta"

def get_log_weight(
        x: torch.Tensor,
        z: torch.Tensor,
        likelihood_distribution: dist.Distribution,
        variational_distribution: dist.Distribution,
        prior_distribution: dist.Distribution
) -> torch.Tensor:
    log_likelihood = likelihood_distribution.log_prob(x)  # KxB
    log_variational = variational_distribution.log_prob(z)  # KxB
    log_prior = prior_distribution.log_prob(z)  # KxB
    
    # TODO: add error message
    assert log_likelihood.size() == log_variational.size()
    assert log_prior.size() == log_variational.size()

    log_weight = log_likelihood + log_prior - log_variational

    return log_weight


def get_iwae_from_log_weights(log_weights: torch.Tensor, K: int) -> torch.Tensor:
    if log_weights.ndim != 2:
        raise ValueError("'log_weight' must be 2 dimensional")

    return torch.logsumexp(log_weights, dim=0) - math.log(K)


@dataclass
class EvaluationMetricResults:
    iwae: float
    elbo_val: float
    rec_error: float
    kl_error: float
    beta_loss_val: float

class SharedVae(ABC, pl.LightningModule):
    @property
    def K_train(self) -> int:
        return self._K_train

    @property
    def K_eval(self) -> int:
        return self._K_eval

    @property
    def K_test(self) -> int:
        return self._K_test
    
    @property
    def input_dim(self) -> int:
        return self._input_dim
    
    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim
    
    @property
    def learning_rate(self) -> float:
        return self._learning_rate
    
    @property
    def precision(self) -> float:
        return self._precision

    @property
    def beta(self) -> float:
        return self._beta

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            learning_rate: float = 1e-3,
            K_train: int = 1,
            K_eval: int = 64,
            K_test: int = 1000,
            precision: float = 1e-6
        ):
        super().__init__()
        self._K_train = K_train
        self._K_eval = K_eval
        self._K_test = K_test
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._learning_rate = learning_rate
        self._precision = precision
        self._beta = 1.

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim)
        variational_distribution = self.variational_distribution(x)  # Bxd
        z = variational_distribution.rsample()
        logits = self.decoder(z)
        x_rec = dist.utils.clamp_probs(dist.utils.logits_to_probs(logits, is_binary=True))

        return x_rec

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, self.input_dim)
        eval_metrics_result = self._shared_eval(x=x, K=self.K_eval)

        self.log("val/iwae", eval_metrics_result.iwae)
        self._log_copula_params(batch_idx, x)

        return eval_metrics_result.__dict__

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, self.input_dim)
        eval_metrics_result = self._shared_eval(x=x, K=self.K_test)

        self.log("test/iwae", eval_metrics_result.iwae)
        self._log_copula_params(batch_idx, x)

        return eval_metrics_result.__dict__

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def _log_copula_params(self, batch_idx, x):
        if isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                d = {}
                variational_distribution = self.variational_distribution(x)
                if isinstance(variational_distribution, MultivariateDistributionUsingCopula):
                    variational_pair_params = variational_distribution.pair_params
                    d[f"Copula/Variational Copula Parameters"] = wandb.Histogram(
                        variational_pair_params.to("cpu").detach().numpy().T
                    )
                    self.logger.experiment.log(d)

    @abstractmethod
    def variational_distribution(self, x: torch.Tensor) -> dist.Distribution:
        pass

    def likelihood_distribution(self, z) -> dist.Distribution:
        logits = self.decoder(z)

        return dist.Independent(dist.ContinuousBernoulli(logits=logits), reinterpreted_batch_ndims=1)

    @abstractmethod
    def prior_distribution(self) -> dist.Distribution:
        pass

    @torch.no_grad()
    def _shared_eval(self, x, K: int) -> EvaluationMetricResults:
        variational_distribution = self.variational_distribution(x)  # Bxd
        z = variational_distribution.rsample(torch.Size([K]))  # Bxd
        likelihood_distribution = self.likelihood_distribution(z)
        prior_distribution = self.prior_distribution()

        err_msg = "Mismatch between variational and prior event shapes. variational: {}, prior: {}"
        assert variational_distribution.event_shape == prior_distribution.event_shape, err_msg.format(variational_distribution.event_shape, prior_distribution.event_shape)

        rec_error = -likelihood_distribution.log_prob(x)
        kl_error = variational_distribution.log_prob(z) - prior_distribution.log_prob(z)  # We tool random z from K evailable values
        elbo_val = -(rec_error + kl_error).mean()
        beta_loss_val = (rec_error + self.beta * kl_error).mean()

        log_weights = get_log_weight(
            x=x,
            z=z,
            variational_distribution=variational_distribution,
            likelihood_distribution=likelihood_distribution,
            prior_distribution=prior_distribution,
        )

        iwae = get_iwae_from_log_weights(
            log_weights=log_weights,
            K=K
        ).mean()

        return EvaluationMetricResults(
            iwae=iwae.item(),
            elbo_val=elbo_val.item(),
            beta_loss_val=beta_loss_val.item(),
            rec_error=rec_error.mean().item(),
            kl_error=kl_error.mean().item()
        )
