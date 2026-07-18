import math
from collections import Counter
from typing import Optional, Any, Dict, Union, Type, List

import numpy as np
import pandas as pd
import networkx as nx
import dgl
import torch
from torch.distributions import (
    MultivariateNormal,
    Categorical,
    Normal,
    constraints
)
from torch.distributions import MultivariateNormal, Normal, Independent

from pyro.distributions import MultivariateStudentT
from tree_copula_vae.torch_copulas.base import MyStudentT as StudentT
from pyro.distributions import Uniform, SpanningTree

from tree_copula_vae.torch_copulas.base import PairCopula, MultivariateCopula
from tree_copula_vae.torch_copulas.graphical_structures import CopulaTreeStructure, CopulaFullGraphStructure, E_TreeStructureBackEndTypes
from tree_copula_vae.torch_copulas.pair_copulas import ConditionalPairCopula
from tree_copula_vae.torch_copulas import extended_constraints
from tree_copula_vae.utils.graph_utils import make_complete_graph, make_numerical_stable_laplacian, get_edge_tensor, maximum_weighted_spanning_tree, safe_edge_logits
from tree_copula_vae.utils.batched_transformations import covariance_to_correlation
from tree_copula_vae.utils.linalg import calculate_pd_matrix_log_det
from tree_copula_vae.utils.numerical_stability import safe_cdf, safe_icdf


class IndependenceCopula(MultivariateCopula):
    def __init__(self, event_shape: Union[int, torch.Size],
                 batch_shape: Union[int, torch.Size] = torch.Size(),
                 validate_args=None):
        self._event_shape = event_shape if isinstance(event_shape, torch.Size) else torch.Size([event_shape])
        if self._event_shape[0] < 2:
            raise Exception(f"Copula event_shape must be >= 2")
        self._batch_shape = batch_shape if isinstance(batch_shape, torch.Size) else torch.Size([batch_shape])
        self._validate_args = False
        super(IndependenceCopula, self).__init__(event_shape=self.event_shape,
                                                 batch_shape=self._batch_shape,
                                                 validate_args=validate_args)

    @property
    def pair_params(self):
        d = self.event_shape[0]
        n_pairs = (d ** 2 - d) // 2
        return torch.zeros(n_pairs)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(IndependenceCopula, _instance)
        batch_shape = torch.Size(batch_shape)
        super(IndependenceCopula, new).__init__(event_shape=self.event_shape,
                                                batch_shape=batch_shape,
                                                validate_args=False)
        return new

    def rsample(self, sample_shape=torch.Size(), device=None):
        uniform_dist = Uniform(
            low=torch.tensor(self._precision, device=device),
            high=torch.tensor(1. - self._precision, device=device)
        ).expand(torch.Size((self.batch_shape[0], self.event_shape[0])))
        samples = uniform_dist.rsample(sample_shape=sample_shape)
        return samples

    def log_prob(self, value):
        if not self.support.check(value):
            raise Exception(f"value argument contains values not in support {self.support}")
        return torch.zeros_like(value)[..., 0]

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return {}


class MultivariateGaussianCopula(MultivariateNormal, MultivariateCopula):
    def __init__(self, correlation_matrix, precision: float = 1e-6):
        if correlation_matrix.dim() == 2:
            correlation_matrix = correlation_matrix.unsqueeze(0)
        device = correlation_matrix.device
        loc = torch.diagonal(torch.zeros_like(correlation_matrix, device=device), offset=0, dim1=1, dim2=2)
        self.standard_normal_dist = Normal(loc=torch.tensor(0., device=device), scale=torch.tensor(1., device=device))
        super().__init__(loc=loc, covariance_matrix=correlation_matrix)

    @property
    def pair_params(self):
        m = self.covariance_matrix
        triu_indices = torch.triu_indices(self.event_shape[0], self.event_shape[0], 1)
        return m[..., triu_indices[0], triu_indices[1]]

    def log_prob(self, value):
        if not self.support.check(value).all():
            raise Exception(f"value argument contains values not in support {self.support}")
        value = self._change_value_to_be_numerical_stable(value)
        device = self.precision_matrix.device
        z = safe_icdf(self.standard_normal_dist, value)
        half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        eye = torch.eye(*self.event_shape, device=device).repeat(*self.batch_shape, 1, 1)
        p = self.precision_matrix - eye
        if z.ndim == 2:
            term = torch.einsum('bi, bij, bj -> b', z.double(), p.double(), z.double())
        elif z.ndim == 3:
            term = torch.einsum('sbi, bij, sbj -> sb', z.double(), p.double(), z.double())
        else:
            raise ValueError(f"Expecting value to be 2/3 dimensions got: {value.ndim}")
        log_density = - half_log_det - 0.5 * term
        return log_density

    def expand(self, batch_shape, _instance=None):
        return super().expand(batch_shape=batch_shape, _instance=self)

    @property
    def support(self) -> Optional[Any]:
        return extended_constraints.unit_cube

    @property
    def correlation_matrix(self) -> torch.Tensor:
        return self.covariance_matrix

    def rsample(self, sample_shape=torch.Size(), device=None):
        q = super(MultivariateGaussianCopula, self).rsample(sample_shape)
        u = safe_cdf(self.standard_normal_dist, q).clamp(self._precision, 1 - self._precision)
        return u

    def __repr__(self):
        return f"MultivariateGaussianCopula(covariance_matrix={self.covariance_matrix})"


class MultivariateStudentTCopula(MultivariateStudentT, MultivariateCopula):
    def __init__(self, correlation_matrix, df, precision: float = 1e-6):
        if correlation_matrix.dim() == 2:
            correlation_matrix = correlation_matrix.unsqueeze(0)
        if isinstance(df, float) or isinstance(df, int):
            df = torch.tensor(df, device=correlation_matrix.device, dtype=correlation_matrix.dtype)
        if df.dim() == 0:
            df = df.expand(correlation_matrix.shape[0])
        device = correlation_matrix.device
        loc = torch.zeros(correlation_matrix.shape[:-1], device=device)
        scale_tril = torch.linalg.cholesky(correlation_matrix)
        self.standard_t_dist = StudentT(df=df.unsqueeze(-1), loc=0., scale=1.)
        self._precision = precision
        super().__init__(df=df, loc=loc, scale_tril=scale_tril)

    @property
    def pair_params(self):
        L = self.scale_tril
        cov = L @ L.transpose(-1, -2)
        triu_indices = torch.triu_indices(self.event_shape[0], self.event_shape[0], 1)
        return cov[..., triu_indices[0], triu_indices[1]]

    def log_prob(self, value):
        if not self.support.check(value).all():
            raise Exception(f"value argument contains values not in support {self.support}")
        value = self._change_value_to_be_numerical_stable(value)
        x = safe_icdf(self.standard_t_dist, value)
        base_dist = MultivariateStudentT(self.df, self.loc, self.scale_tril)
        joint_log_prob = base_dist.log_prob(x)
        marginal_log_prob = self.standard_t_dist.log_prob(x).sum(-1)
        log_density = joint_log_prob - marginal_log_prob
        return log_density

    def rsample(self, sample_shape=torch.Size()):
        x = super().rsample(sample_shape)
        u = safe_cdf(self.standard_t_dist, x).clamp(self._precision, 1 - self._precision)
        return u

    @property
    def support(self):
        return constraints.interval(0., 1.)

    @property
    def correlation_matrix(self) -> torch.Tensor:
        L = self.scale_tril
        return L @ L.transpose(-1, -2)

    def __repr__(self):
        return f"MultivariateStudentTCopula(df={self.df}, correlation_matrix_shape={self.scale_tril.shape})"


class TreeCopula(MultivariateCopula):
    @property
    def pair_params(self) -> torch.Tensor:
        return self._copula_tree_structure.copula_pair_params

    @property
    def copula_tree_structure(self) -> CopulaTreeStructure:
        return self._copula_tree_structure

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return self._arg_constraints

    def __init__(self, copula_tree_structure: CopulaTreeStructure, precision: float = 1e-6):
        self._copula_tree_structure = copula_tree_structure
        self._pair_copula_class: Type[PairCopula] = copula_tree_structure.pair_copula_class
        event_shape = torch.Size([copula_tree_structure.number_of_nodes])
        batch_shape = copula_tree_structure.batch_shape
        self._arg_constraints = {}
        super().__init__(batch_shape=batch_shape, event_shape=event_shape, precision=precision)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if not self.support.check(value).all():
            raise Exception(f"value argument contains values not in support {self.support}")

        value = self._change_value_to_be_numerical_stable(value)

        distribution_batch_shape = self.batch_shape

        if value.size(-1) != self.event_shape[0]:
            raise ValueError(f"Value must match event shape in the last dimension")

        value_batch_shape = value.size()[:-1]
        copula_tree_structure = self._copula_tree_structure

        n_nodes_per_tree = copula_tree_structure.number_of_nodes
        edges = copula_tree_structure.edges
        edges_batch_idx = (edges // n_nodes_per_tree)
        edges_event_idx = (edges % n_nodes_per_tree)
        edge_location = CopulaFullGraphStructure.nodes_to_edge_location(edges_event_idx[:, 0], edges_event_idx[:, 1])

        pair_params = copula_tree_structure.copula_pair_params.unsqueeze(0)
        pair_copulas = copula_tree_structure.pair_copula_class(
            pair_params[..., edges_batch_idx[..., 0], edge_location],
            precision=self._precision
        )
        value = value.unsqueeze(0)

        if distribution_batch_shape == torch.Size([]) or distribution_batch_shape == torch.Size([1]):
            value_pairs = value[..., edges_event_idx]
        else:
            value_pairs = value[..., edges_batch_idx, edges_event_idx]

        log_prob = pair_copulas.log_prob(value_pairs)
        pre_sum_log_p = log_prob.view(*value_batch_shape, -1)
        log_prob = pre_sum_log_p.sum(-1)
        return log_prob

    def log_prob2(self, value):
        value = self._change_value_to_be_numerical_stable(value)
        mwst = maximum_weighted_spanning_tree(self.pair_params)
        log_probs = TreeAveragedCopula.calculate_AoT_log_prob(
            value=value,
            copula_pair_params=self.pair_params,
            tree_prior_params_in_log=mwst.log(),
            num_vertices=self.event_shape[0],
            copula_type=self._pair_copula_class,
            use_clip_range=False
        )
        return log_probs

    def _node_index_to_batch_index(self, node_indices: Union[List[int], torch.Tensor]):
        if isinstance(node_indices, list):
            node_indices = torch.tensor(node_indices)
        return node_indices // self._copula_tree_structure.number_of_nodes

    def _node_index_to_event_index(self, node_indices: Union[List[int], torch.Tensor]):
        if isinstance(node_indices, list):
            node_indices = torch.tensor(node_indices)
        return node_indices % self._copula_tree_structure.number_of_nodes

    def _get_edge_df(self, full_edge_list: pd.DataFrame, node_ids: List[int]) -> pd.DataFrame:
        is_edge_candidate = full_edge_list['source'].isin(node_ids) | full_edge_list['target'].isin(node_ids)
        edges_list_df = full_edge_list[is_edge_candidate].reset_index(drop=True)
        is_swap_needed = ~edges_list_df['source'].isin(node_ids)
        parent_src_cpy = edges_list_df['source'].copy()
        edges_list_df.loc[is_swap_needed, 'source'] = edges_list_df.loc[is_swap_needed, 'target']
        edges_list_df.loc[is_swap_needed, 'target'] = parent_src_cpy.loc[is_swap_needed]
        return edges_list_df

    def _dgl_sampling(self, samples: torch.Tensor, pair_params, sample_shape):
        device = samples.device
        dgl_graph = self._copula_tree_structure.dgl_graph.to(device)

        for e in dgl.bfs_edges_generator(dgl_graph, self._copula_tree_structure.root_id):
            dgl_src_node, dgl_dst_node = dgl_graph.find_edges(e.to(device))
            src_dst_in_pyro_order = torch.stack([dgl_src_node, dgl_dst_node]).T
            is_swap_needed = dgl_src_node > dgl_dst_node
            temp = src_dst_in_pyro_order[is_swap_needed, 0]
            src_dst_in_pyro_order[is_swap_needed, 0] = src_dst_in_pyro_order[is_swap_needed, 1]
            src_dst_in_pyro_order[is_swap_needed, 1] = temp

            pyro_node_relative_location = src_dst_in_pyro_order % self._copula_tree_structure.number_of_nodes
            batch_location = src_dst_in_pyro_order[..., 0] // self._copula_tree_structure.number_of_nodes
            edge_location_in_pair_params = CopulaFullGraphStructure.nodes_to_edge_location(
                pyro_node_relative_location[..., 0],
                pyro_node_relative_location[..., 1]
            )
            dgl_src_node_relative_location = dgl_src_node % self._copula_tree_structure.number_of_nodes
            dgl_dst_node_relative_location = dgl_dst_node % self._copula_tree_structure.number_of_nodes

            copula_param = pair_params[..., batch_location, edge_location_in_pair_params]
            assert torch.isclose(copula_param.float(), dgl_graph.edata['copula_param'][e].float()).all()
            parent_sample = samples[..., batch_location, dgl_src_node_relative_location]
            assert (parent_sample != 0).all()
            copula_dist = ConditionalPairCopula(self._pair_copula_class(copula_param), u=parent_sample)
            assert (samples[..., batch_location, dgl_dst_node_relative_location] == 0).all()
            samples[..., batch_location, dgl_dst_node_relative_location] = copula_dist.rsample(sample_shape=sample_shape)
            samples[..., batch_location, dgl_dst_node_relative_location] = samples[..., batch_location, dgl_dst_node_relative_location].clamp(
                min=self._precision, max=1 - self._precision
            )

        return samples

    def rsample(self, sample_shape: torch.Size = torch.Size(), device=None):
        if len(sample_shape) > 1:
            raise ValueError(f"Not supporting multiple dimension of sample_shape")

        device = self._copula_tree_structure.copula_pair_params.device
        dtype = self.pair_params.dtype
        batch_shape = self.batch_shape if len(self.batch_shape) > 0 else torch.Size([1])
        sample_shape_ = sample_shape if len(sample_shape) > 0 else torch.Size([1])
        pair_params = self.pair_params.view(*batch_shape, self.pair_params.size(-1))
        samples = torch.zeros((*sample_shape_, *batch_shape, *self.event_shape), device=device, dtype=dtype)

        batch_idx = self._node_index_to_batch_index(self._copula_tree_structure.root_id)
        event_index = self._node_index_to_event_index(self._copula_tree_structure.root_id)
        uniform_dist = Uniform(
            low=torch.tensor(self._precision, device=device, dtype=dtype),
            high=torch.tensor(1. - self._precision, device=device, dtype=dtype)
        ).expand(batch_shape)
        samples[..., batch_idx, event_index] = uniform_dist.sample(sample_shape=sample_shape)

        if self._copula_tree_structure.backend == E_TreeStructureBackEndTypes.dgl:
            samples = self._dgl_sampling(samples, pair_params, sample_shape)
        else:
            iteration_order = list(self._copula_tree_structure)
            prev_children_edges_list_df = None
            nx_mwst_tree = self._copula_tree_structure.nx_mwst_tree
            full_edge_list = nx.to_pandas_edgelist(nx_mwst_tree)

            for parents_node_ids, children_node_ids in zip(iteration_order[:-1], iteration_order[1:]):
                if prev_children_edges_list_df is None:
                    parents_edges_list_df = self._get_edge_df(full_edge_list=full_edge_list, node_ids=parents_node_ids)
                else:
                    parents_edges_list_df = prev_children_edges_list_df

                children_edges_list_df = self._get_edge_df(full_edge_list=full_edge_list, node_ids=children_node_ids)
                local_graph_df = pd.merge(
                    parents_edges_list_df, children_edges_list_df,
                    left_on=['source', 'target'],
                    right_on=['target', 'source'],
                    suffixes=('_parent', '_child')
                )
                assert local_graph_df['target_parent'].nunique() == local_graph_df['target_parent'].shape[0]

                local_graph_df['batch_idx'] = local_graph_df['source_parent'] // self._copula_tree_structure.number_of_nodes
                local_graph_df[['source_parent', 'target_parent', 'source_child', 'target_child']] %= self._copula_tree_structure.number_of_nodes
                for_edge_location_idx = np.sort(local_graph_df[['source_parent', 'target_parent']].values)
                local_graph_df['edge_location'] = CopulaFullGraphStructure.nodes_to_edge_location(for_edge_location_idx[:, 0], for_edge_location_idx[:, 1])
                copula_param = pair_params[..., local_graph_df['batch_idx'].values, local_graph_df['edge_location'].values]
                parent_sample = samples[..., local_graph_df['batch_idx'].values, local_graph_df['source_parent'].values]
                assert (local_graph_df.groupby("batch_idx").count().iloc[:, 0] == local_graph_df.groupby("batch_idx")['target_parent'].nunique()).all()
                assert (samples[..., local_graph_df['batch_idx'].values, local_graph_df['source_child'].values] == 0).all()

                copula_dist = ConditionalPairCopula(self._pair_copula_class(copula_param), u=parent_sample)
                samples[..., local_graph_df['batch_idx'].values, local_graph_df['source_child'].values] = copula_dist.rsample(sample_shape=sample_shape)
                prev_children_edges_list_df = children_edges_list_df

        samples = samples.view(*sample_shape, *self.batch_shape, *self.event_shape)
        return samples

    def expand(self, batch_shape: torch.Size, _instance=None):
        batch_shape = torch.Size(batch_shape)
        if batch_shape == self.batch_shape:
            return self
        try:
            torch.broadcast_shapes(batch_shape, self.batch_shape)
        except RuntimeError:
            raise ValueError(
                f"Incompatible batch_shape {batch_shape} for current {self.batch_shape}."
            )
        expanded_tree_structure = self._copula_tree_structure.expand(batch_shape)
        new = self._get_checked_instance(TreeCopula, _instance)
        TreeCopula.__init__(new, copula_tree_structure=expanded_tree_structure, precision=self._precision)
        return new

    def __repr__(self):
        return f"TreeCopula({self._copula_tree_structure})"


class TreeAveragedCopula(MultivariateCopula):
    @property
    def pair_params(self) -> torch.Tensor:
        return self._copula_pair_params

    @property
    def pair_copula_class(self) -> Type[PairCopula]:
        return self._pair_copula_class

    @property
    def copula_full_graph_structure(self) -> CopulaFullGraphStructure:
        return self._copula_full_graph_structure

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return self._arg_constraints

    @property
    def tree_prior_pair_params(self) -> torch.Tensor:
        return self._tree_prior_pair_params

    @tree_prior_pair_params.setter
    def tree_prior_pair_params(self, t: torch.Tensor):
        if t.ndim != self._tree_prior_pair_params.ndim:
            raise ValueError(f"new tree_prior_pair_params value must be of dim {self._tree_prior_pair_params.ndim} found {t.ndim}")
        if (t != 0).any():
            self._tree_prior_pair_params = t
            self._tree_prior_params_in_log = torch.log(self._tree_prior_pair_params)

    def __init__(self, copula_full_graph_structure: CopulaFullGraphStructure,
                 tree_prior_pair_params: torch.Tensor = None,
                 tree_prior_pair_params_in_log: torch.Tensor = None,
                 precision: float = 1e-6):
        if tree_prior_pair_params is not None and tree_prior_pair_params_in_log is not None:
            raise ValueError("tree_prior_pair_params and tree_prior_pair_params_in_log are mutually exclusive")
        event_shape = torch.Size([copula_full_graph_structure.number_of_nodes])
        self._copula_full_graph_structure = copula_full_graph_structure
        self._copula_pair_params = copula_full_graph_structure.copula_pair_params
        self._pair_copula_class = copula_full_graph_structure.pair_copula_class
        if tree_prior_pair_params is None and tree_prior_pair_params_in_log is None:
            self._tree_prior_pair_params = torch.ones_like(self._copula_pair_params, requires_grad=False)
            self._tree_prior_params_in_log = torch.zeros_like(self._tree_prior_pair_params, requires_grad=False)
            self._uniform_prior = True
        elif tree_prior_pair_params is not None:
            self._tree_prior_pair_params = tree_prior_pair_params
            self._tree_prior_params_in_log = torch.log(self.tree_prior_pair_params)
            self._uniform_prior = False
        elif tree_prior_pair_params_in_log is not None:
            self._tree_prior_pair_params = tree_prior_pair_params_in_log.exp()
            self._tree_prior_params_in_log = tree_prior_pair_params_in_log
            self._uniform_prior = False
        self._arg_constraints = {}
        super().__init__(event_shape=event_shape, precision=precision)

    def rsample(self, sample_shape: torch.Size = torch.Size(), device=None):
        return torch.zeros(*sample_shape, *self.event_shape)

    def sample(self, sample_shape: torch.Size = torch.Size(), device=None):
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, device=device)

    @staticmethod
    def calculate_AoT_log_prob(value: torch.Tensor,
                               copula_pair_params: torch.Tensor,
                               tree_prior_params_in_log: torch.Tensor,
                               num_vertices: int,
                               copula_type: Type[PairCopula],
                               use_clip_range: bool = False):
        return torch.zeros_like(value[..., 0])
