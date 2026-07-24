import networkx as nx
import numpy as np
import torch
from typing import List, Union, Optional, Any, Dict

import torch.nn as nn
import zuko.distributions
from torch.distributions import Distribution, constraints, MultivariateNormal, Normal, Independent, register_kl, kl_divergence, Categorical, Transform
from pyro.distributions import SpanningTree
from tree_copula_vae.torch_copulas.multivariate_copulas import MultivariateCopula
from tree_copula_vae.utils.batched_transformations import covariance_to_correlation
from tree_copula_vae.utils.graph_utils import make_complete_graph, make_numerical_stable_laplacian, sample_soft_tree, sample_hard_tree, get_edge_tensor

from pyro.distributions.torch_distribution import TorchDistribution

from tree_copula_vae.utils.numerical_stability import safe_icdf, safe_cdf


class CopulaMarginals(Independent):
    def __init__(self, base_distribution: Distribution, reinterpreted_batch_ndims: int, precision: float = 1e-6):
        super().__init__(base_distribution, reinterpreted_batch_ndims)
        self._eps = precision

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        return safe_cdf(dist=self.base_dist, value=value)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        if ((value < 0) | (value > 1)).any():
            raise ValueError(f"icdf expecting values in the interval [0, 1]")

        # If value is exactly 1 or exactly 0 it can yield an -inf value
        value = value.clamp(0 + self._eps, 1 - self._eps)

        return safe_icdf(dist=self.base_dist, value=value)

    def expand(self, batch_shape, _instance=None):
        super().expand(batch_shape=batch_shape, _instance=self)
        # self.base_dist = self.base_dist.expand(batch_shape=batch_shape)

        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(base_distribution={self.base_dist.__repr__()})"

class CopulaMarginalsWithNormalizingFlow(Distribution):
    @property
    def univariate_normalizing_flow(self) -> zuko.distributions.NormalizingFlow:
        return self._nf

    def __init__(self, univariate_normalizing_flow: zuko.distributions.NormalizingFlow, precision: float = 1e-6):
        if not isinstance(univariate_normalizing_flow.base, Independent):
            raise ValueError("Currently support only base distributions of type 'torch.distributions.Independent'")

        self._eps = precision
        self._nf = univariate_normalizing_flow

        batch_shape = self._nf.batch_shape
        event_shape = self._nf.event_shape
        validate_args = False
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args
        )

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        z = self._nf.transform(value)
        return safe_cdf(dist=self._nf.base.base_dist, value=z)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        if ((value < 0) | (value > 1)).any():
            raise ValueError(f"icdf expecting values in the interval [0, 1]")

        # If value is exactly 1 or exactly 0 it can yield an -inf value
        u = value.clamp(self._eps, 1 - self._eps)
        z = safe_icdf(dist=self._nf.base.base_dist, value=u)

        return self._nf.transform.inv(z)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._nf.log_prob(value)

    def expand(self, batch_shape, _instance=None):
        self._nf.expand(batch_shape=batch_shape)
        # self.base_dist = self.base_dist.expand(batch_shape=batch_shape)

        return self

class MultivariateDistributionUsingCopula(TorchDistribution):
    @property
    def marginal_distributions(self) -> CopulaMarginals:
        return self._marginals

    @property
    def copula_distribution(self) -> MultivariateCopula:
        return self._multivariate_copula

    @property
    def support(self) -> Optional[Any]:
        return self.marginal_distributions.support

    @property
    def pair_params(self) -> torch.Tensor:
        return self.copula_distribution.pair_params

    @property
    def has_rsample(self):
        return self.copula_distribution.has_rsample and self.marginal_distributions.has_rsample

    def __init__(self, marginals: Distribution,
                 multivariate_copula: MultivariateCopula,
                 marginals_reinterpreted_batch_ndims: int = 1,
                 precision: float = 1e-6):
        # Marginals allowed to be only distribution without an event shape
        is_marginals_valid_event_shape = marginals.event_shape == torch.Size([])
        if not is_marginals_valid_event_shape:
            raise ValueError(f"The event shape of the marginal distribution (before transformation) should be 0, got: {marginals.event_shape}")

        self._marginals = CopulaMarginals(
            base_distribution=marginals,
            reinterpreted_batch_ndims=marginals_reinterpreted_batch_ndims,
            precision=precision
        )

        is_event_shape_valid = self._marginals.event_shape == multivariate_copula.event_shape
        if not is_event_shape_valid:
            raise ValueError(f"marginals (after transformation) and copula must have the same event_shape,"
                             f" found: copula.event_shape={multivariate_copula.event_shape}, marginals.event_shape={self._marginals.event_shape}")
        self._event_shape = self._marginals.event_shape

        is_valid_batch_shape = self._marginals.batch_shape == multivariate_copula.batch_shape
        if not is_valid_batch_shape:
            raise ValueError(f"marginals (after transformation) and copula must have the same batch_shape,"
                             f" found: copula.batch_shape={multivariate_copula.batch_shape}, marginals.batch_shape={self._marginals.batch_shape}")

        self._batch_shape = self._marginals.batch_shape

        # Since we already get the distributions we don't perform arguments validation check.
        validate_args = False
        super().__init__(
            event_shape=self._event_shape,
            batch_shape=self._batch_shape,
            validate_args=validate_args
        )

        self._multivariate_copula = multivariate_copula
        self._arg_constraints = {k: v for k, v in self._marginals.arg_constraints.items()}
        self._arg_constraints.update({k: v for k, v in self._multivariate_copula.arg_constraints.items()})
        self._cdf_eps_boundaries = precision


    def rsample(self, sample_shape: torch.Size = torch.Size(), device=None) -> torch.Tensor:
        pseudo_samples = self.copula_distribution.rsample(sample_shape=sample_shape, device=device).to(device)
        # For numerical stability as icdf return nan/inf for value that are exactly 1 or 0
        pseudo_samples = pseudo_samples.clamp(0 + self._cdf_eps_boundaries, 1 - self._cdf_eps_boundaries)
        samples = safe_icdf(dist=self.marginal_distributions, value=pseudo_samples)
        # samples = self.marginal_distributions.icdf(pseudo_samples)

        return samples

    def sample(self, sample_shape: torch.Size = torch.Size(), device=None) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, device=device)


    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if not self.support.check(value).all():
            raise ValueError(f"value argument contains values not in support {self.support}")

        pseudo_observations = safe_cdf(dist=self.marginal_distributions, value=value)
        # pseudo_observations = self.marginal_distributions.cdf(value)
        log_probability = self.marginal_distributions.log_prob(value) + self.copula_distribution.log_prob(pseudo_observations)

        return log_probability

    def expand(self, batch_shape: torch.Size, _instance=None):
        self._marginals.expand(batch_shape)
        self._multivariate_copula.expand(batch_shape)
        self._batch_shape = self._marginals.batch_shape

        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(marginals={self._marginals.__repr__()}, multivariate_copula={self._multivariate_copula.__repr__()})"


class NormalizingFlowWithCopulaCoupling(TorchDistribution):
    @property
    def univariate_normalizing_flow(self) -> CopulaMarginalsWithNormalizingFlow:
        return self._nf

    @property
    def copula_distribution(self) -> MultivariateCopula:
        return self._multivariate_copula

    @property
    def pair_params(self) -> torch.Tensor:
        return self.copula_distribution.pair_params

    @property
    def support(self) -> Optional[Any]:
        return self._nf.support

    def __init__(
            self,
            univariate_normalizing_flow: zuko.distributions.NormalizingFlow,
            multivariate_copula: MultivariateCopula,
            precision: float = 1e-6
    ):
        if univariate_normalizing_flow.event_shape != multivariate_copula.event_shape:
            raise ValueError("expecting 'univariate_normalizing_flow.event_shape' to be equal to 'multivariate_copula.event_shape'")

        if univariate_normalizing_flow.batch_shape != multivariate_copula.batch_shape:
            raise ValueError("expecting 'univariate_normalizing_flow.batch_shape' to be equal to 'multivariate_copula.batch_shape'")

        self._nf = CopulaMarginalsWithNormalizingFlow(univariate_normalizing_flow)
        self._multivariate_copula = multivariate_copula
        self._cdf_eps_boundaries = precision

        # Since we already get the distributions we don't perform arguments validation check.
        validate_args = False
        self._event_shape = self._nf.event_shape
        self._batch_shape = self._nf.batch_shape
        super().__init__(
            event_shape=self._event_shape,
            batch_shape=self._batch_shape,
            validate_args=validate_args
        )

    def rsample(self, sample_shape: torch.Size = torch.Size(), device=None) -> torch.Tensor:
        pseudo_samples = self.copula_distribution.rsample(sample_shape=sample_shape, device=device).to(device)
        # For numerical stability as icdf return nan/inf for value that are exactly 1 or 0
        pseudo_samples = pseudo_samples.clamp(self._cdf_eps_boundaries, 1 - self._cdf_eps_boundaries)
        # samples = safe_icdf(dist=self.marginal_distributions, value=pseudo_samples)
        samples = self._nf.icdf(pseudo_samples)

        return samples

    def sample(self, sample_shape: torch.Size = torch.Size(), device=None) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, device=device)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # if not self.support.check(value).all():
        #     raise ValueError(f"value argument contains values not in support {self.support}")

        pseudo_observations = self._nf.cdf(value)
        log_probability = self._nf.log_prob(value) + self.copula_distribution.log_prob(pseudo_observations)

        return log_probability

    def expand(self, batch_shape: torch.Size, _instance=None):
        self._nf.expand(batch_shape)
        self._multivariate_copula.expand(batch_shape)
        self._batch_shape = self._nf.batch_shape

        return self

class AoTDistribution(Distribution):
    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return self._arg_constraints

    @property
    def support(self) -> Optional[Any]:
        return constraints.boolean

    @property
    def edge_logits(self) -> torch.Tensor:
        return self._edge_logits

    @property
    def has_rsample(self) -> bool:
        # In theory copula can always be resampled using 'Inverse Integral Trasnform'
        return True

    def __init__(self, edge_logits: torch.Tensor,
                 tau: float = 1.,
                 max_range: float = 15.,
                 validate_args: Optional[bool] = None):
        """
        A discrete tree distribution.
        Note! this is not a real tree distribution in sampling terms.
        we use this class only for code compatibility with the VAE framework
        Parameters
        ----------
        edge_logits :
        max_range :
        validate_args :
        """
        self._arg_constraints = {}

        batch_shape = torch.Size([edge_logits.size(0)])
        event_shape = torch.Size([edge_logits.size(1)])
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args
        )

        self._edge_logits = edge_logits
        self._max_range = max_range
        self._n_nodes = int(round(0.5 + (0.25 + 2 * edge_logits.size(-1)) ** 0.5))
        self._tau = tau

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if not self.support.check(value).all():
            raise Exception(f"value argument contains values not in support {self.support}")

        # Make edge_logits numerical stable
        # edge_logits = clip_range(self._edge_logits, self._max_range)
        edge_logits = self._edge_logits

        m = torch.max(edge_logits, dim=-1, keepdim=True)[0]
        edge_logits = edge_logits - m

        L_star = make_numerical_stable_laplacian(
            edge_weights_in_log=edge_logits,
            num_vertices=self._n_nodes,
            batch_shape=torch.Size([value.size(0)])
        )[..., :-1, :-1]
        normalizer = torch.slogdet(L_star).logabsdet
        # Return the max that was subtract from each edge_weights
        normalizer = normalizer + (self._n_nodes - 1) * m.flatten()
        edge_logits = edge_logits + m

        edge_logits = edge_logits.unsqueeze(0).expand(value.size(0), -1)
        relevant_edges = edge_logits * value

        log_prob = relevant_edges.sum(2) - normalizer
        # Sometimes we have values larger then 1 (log(p) = 1) due to numerical stability issues so we clip them
        log_prob = log_prob.clamp(max=0.)

        return log_prob

    def rsample(self,
                sample_shape: torch.Size = torch.Size(),
                hard: bool = False,
                device: Optional[torch.device] = None) -> torch.Tensor:
        # TODO: this is wrong!
        inject_gumbel_noise = True
        if hard:
            hard_mwst, edge_weights = sample_hard_tree(edge_logits=self._edge_logits)
            inject_gumbel_noise = False

        soft_mwst, edge_weights = sample_soft_tree(
            edge_logits=self._edge_logits,
            num_nodes=self._n_nodes,
            temperature=self._tau,
            inject_gumbel_noise=inject_gumbel_noise
        )

        if hard:
            mwst = soft_mwst + (hard_mwst - soft_mwst).detach()
        else:
            mwst = soft_mwst

        return mwst

    def sample(self, sample_shape: torch.Size = torch.Size()):
        # Can be not abstract since we assuming that has_rsample is always True
        with torch.no_grad():
            hard_mwst, edge_weights = sample_hard_tree(edge_logits=self._edge_logits)

        return hard_mwst

    @staticmethod
    def sample_tree_from_uniform_dist(n_samples: int, n_nodes: int, device: Union[str, torch.device]):
        # TODO: add transformation to eval if needed
        n_edges = (n_nodes ** 2 - n_nodes) // 2
        logits = torch.ones(n_nodes)
        tree_uni_dist = Categorical(logits=logits)
        prufer_seq = tree_uni_dist.sample(torch.Size([n_samples, n_nodes - 2]))

        trees = np.array([
            list(nx.from_prufer_sequence(list(seq.numpy())).edges()) for seq in prufer_seq
        ])
        graph = torch.zeros(n_samples, n_nodes, n_nodes, device=device)
        for i, tree in enumerate(trees):
            v1, v2 = tree.T
            graph[i, v1, v2] = 1
            graph[i, v2, v1] = 1

        edges = get_edge_tensor(graph)

        return edges

    def __repr__(self):
        # For string use and debugging
        return self.__class__.__name__
