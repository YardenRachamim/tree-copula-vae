from typing import Any, Optional

import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import Distribution, Independent

from tree_copula_vae.torch_copulas.base import MultivariateCopula
from tree_copula_vae.utils.numerical_stability import safe_cdf, safe_icdf


class CopulaMarginals(Independent):
    def __init__(
        self,
        base_distribution: Distribution,
        reinterpreted_batch_ndims: int,
        precision: float = 1e-6,
    ):
        super().__init__(base_distribution, reinterpreted_batch_ndims)
        self._precision = precision

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        return safe_cdf(dist=self.base_dist, value=value)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        if ((value < 0) | (value > 1)).any():
            raise ValueError("icdf expects values in [0, 1]")
        return safe_icdf(
            dist=self.base_dist,
            value=value.clamp(self._precision, 1 - self._precision),
        )


class MultivariateDistributionUsingCopula(TorchDistribution):
    @property
    def marginal_distributions(self) -> CopulaMarginals:
        return self._marginals

    @property
    def copula_distribution(self) -> MultivariateCopula:
        return self._multivariate_copula

    @property
    def support(self) -> Optional[Any]:
        return self._marginals.support

    @property
    def pair_params(self) -> torch.Tensor:
        return self._multivariate_copula.pair_params

    @property
    def has_rsample(self) -> bool:
        return self._multivariate_copula.has_rsample and self._marginals.has_rsample

    def __init__(
        self,
        marginals: Distribution,
        multivariate_copula: MultivariateCopula,
        marginals_reinterpreted_batch_ndims: int = 1,
        precision: float = 1e-6,
    ):
        if marginals.event_shape != torch.Size([]):
            raise ValueError(
                "marginals must be univariate before reinterpretation; "
                f"got event_shape={marginals.event_shape}"
            )

        self._marginals = CopulaMarginals(
            base_distribution=marginals,
            reinterpreted_batch_ndims=marginals_reinterpreted_batch_ndims,
            precision=precision,
        )
        self._multivariate_copula = multivariate_copula
        self._cdf_eps_boundaries = precision

        if self._marginals.event_shape != multivariate_copula.event_shape:
            raise ValueError(
                "marginals and copula must have the same event shape; "
                f"got {self._marginals.event_shape} and {multivariate_copula.event_shape}"
            )
        if self._marginals.batch_shape != multivariate_copula.batch_shape:
            raise ValueError(
                "marginals and copula must have the same batch shape; "
                f"got {self._marginals.batch_shape} and {multivariate_copula.batch_shape}"
            )

        super().__init__(
            batch_shape=self._marginals.batch_shape,
            event_shape=self._marginals.event_shape,
            validate_args=False,
        )

    def rsample(self, sample_shape: torch.Size = torch.Size(), device=None) -> torch.Tensor:
        pseudo_samples = self._multivariate_copula.rsample(sample_shape=sample_shape, device=device)
        pseudo_samples = pseudo_samples.clamp(self._cdf_eps_boundaries, 1 - self._cdf_eps_boundaries)
        return safe_icdf(dist=self._marginals, value=pseudo_samples)

    def sample(self, sample_shape: torch.Size = torch.Size(), device=None) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, device=device)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if not self.support.check(value).all():
            raise ValueError(f"value contains entries outside support {self.support}")
        pseudo_observations = safe_cdf(dist=self._marginals, value=value)
        return self._marginals.log_prob(value) + self._multivariate_copula.log_prob(pseudo_observations)
