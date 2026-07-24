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

import math
import torch

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
        # We don't have arguments so we have nothing to validate
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
        # new._n_distributions_per_sample = batch_shape
        super(IndependenceCopula, new).__init__(event_shape=self.event_shape,
                                                batch_shape=batch_shape,
                                                validate_args=False)
        # new._validate_args = self._validate_args

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

        # In Independence copula the log prob is always zero
        return torch.zeros_like(value)[..., 0]

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return {}


class MultivariateGaussianCopula(MultivariateNormal, MultivariateCopula):
    def __init__(self, correlation_matrix, precision: float = 1e-6):

        # Make sure we always have batch_size of at least 1
        if correlation_matrix.dim() == 2:
            correlation_matrix = correlation_matrix.unsqueeze(0)

        device = correlation_matrix.device

        # The mu vector of the gaussian copula is always zero
        loc = torch.diagonal(
            torch.zeros_like(correlation_matrix, device=device),
            offset=0, dim1=1, dim2=2
        )
        # correlation_matrix = covariance_to_correlation(covariance_matrix, precision=precision)
        self.standard_normal_dist = Normal(
            loc=torch.tensor(0., device=device),
            scale=torch.tensor(1., device=device)
        )

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

        # Compute the inverse CDF of the standard normal distribution using PyTorch's Normal
        z =  safe_icdf(self.standard_normal_dist, value)
        # Same as calculate_pd_log_det
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
        # Since sampling from the gaussian copula is equivalent to  sample from normal distribution with the parameters that we already defined and
        # just convert them using the standard normal cdf. we can use rsample from the actual Multivariate distribution that we use and just
        # transfer it
        q = super(MultivariateGaussianCopula, self).rsample(sample_shape)
        u = safe_cdf(self.standard_normal_dist, q).clamp(self._precision, 1 - self._precision)

        return u

    def __repr__(self):
        return f"MultivariateGaussianCopula(covariance_matrix={self.covariance_matrix})"


class MultivariateStudentTCopula(MultivariateStudentT, MultivariateCopula):
    def __init__(self, correlation_matrix, df, precision: float = 1e-6):
        # 1. התאמת מימדים (Batch logic)
        if correlation_matrix.dim() == 2:
            correlation_matrix = correlation_matrix.unsqueeze(0)

        # 2. טיפול בדרגות חופש (df)
        if isinstance(df, float) or isinstance(df, int):
            df = torch.tensor(df, device=correlation_matrix.device, dtype=correlation_matrix.dtype)
        # אם df הוא סקלר, נרחיב אותו לגודל הבאץ' כדי ש-Pyro לא יתלונן
        if df.dim() == 0:
            df = df.expand(correlation_matrix.shape[0])

        device = correlation_matrix.device

        # 3. הגדרת המרכז והסקאלה
        # בקופולה, הממוצע של משתני הבסיס הוא תמיד 0
        loc = torch.zeros(correlation_matrix.shape[:-1], device=device)
        # Pyro מצפה ל-scale_tril (משולשית תחתונה - Cholesky)
        scale_tril = torch.linalg.cholesky(correlation_matrix)

        # 4. הגדרת התפלגות השוליים הסטנדרטית (Univariate Student-T)
        # נדרשת כדי להמיר בין המרחב הלטנטי (Real) למרחב הקופולה (Unit Cube)
        # loc=0, scale=1 הם הסטנדרט לבסיס הקופולה
        self.standard_t_dist = StudentT(df=df.unsqueeze(-1), loc=0., scale=1.)

        self._precision = precision

        # 5. אתחול מחלקת האב של Pyro
        super().__init__(df=df, loc=loc, scale_tril=scale_tril)

    @property
    def pair_params(self):
        # חילוץ הפרמטרים למקרה שצריך אותם לחישובי MI או לוגים אחרים
        # ב-Pyro, ה-scale_tril נשמר. נחשב ממנו את הקורלציה חזרה.
        L = self.scale_tril
        cov = L @ L.transpose(-1, -2)
        triu_indices = torch.triu_indices(self.event_shape[0], self.event_shape[0], 1)
        return cov[..., triu_indices[0], triu_indices[1]]

    def log_prob(self, value):
        # בדיקת תמיכה (Unit Cube)
        if not self.support.check(value).all():
            raise Exception(f"value argument contains values not in support {self.support}")

        value = self._change_value_to_be_numerical_stable(value)

        # 1. המרה מיוניפורם (u) למרחב ה-t (x)
        # שימוש ב-Inverse CDF של t חד-מימדי
        x = safe_icdf(self.standard_t_dist, value)

        # 2. חישוב הצפיפות המשותפת (Joint Log Prob) בעזרת Pyro
        # Pyro יודע לחשב log_prob של t רב-מימדי ביעילות
        base_dist = MultivariateStudentT(self.df, self.loc, self.scale_tril)
        joint_log_prob = base_dist.log_prob(x)

        # 3. חישוב סכום הצפיפויות השוליות (Marginal Log Probs)
        # log c(u) = log f_joint(x) - sum(log f_marginal(x_i))
        marginal_log_prob = self.standard_t_dist.log_prob(x).sum(-1)

        # צפיפות הקופולה
        log_density = joint_log_prob - marginal_log_prob

        return log_density

    def rsample(self, sample_shape=torch.Size()):
        # 1. דגימה מהתפלגות t רב-מימדית (Pyro)
        x = super().rsample(sample_shape)

        # 2. המרה ליוניפורם (Probability Integral Transform)
        # שימוש ב-CDF של t חד-מימדי
        u = safe_cdf(self.standard_t_dist, x).clamp(self._precision, 1 - self._precision)

        return u

    @property
    def support(self):
        # התיקון: שימוש ב-interval הסטנדרטי של PyTorch
        return constraints.interval(0., 1.)

    @property
    def correlation_matrix(self) -> torch.Tensor:
        # שחזור מטריצת הקורלציה מתוך ה-Cholesky שאוחסן ב-Pyro
        L = self.scale_tril
        return L @ L.transpose(-1, -2)

    def __repr__(self):
        return f"MultivariateStudentTCopula(df={self.df}, correlation_matrix_shape={self.scale_tril.shape})"


# Helper assuming you have safe_icdf/cdf defined elsewhere as in your snippet
# If not, simply map them to:
# safe_icdf = lambda dist, val: dist.icdf(val.clamp(1e-6, 1-1e-6))
# safe_cdf = lambda dist, val: dist.cdf(val)

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
        # TODO: add arg constrints
        self._copula_tree_structure = copula_tree_structure
        self._pair_copula_class: Type[PairCopula] = copula_tree_structure.pair_copula_class
        # self._nodes = copula_tree_structure.nodes
        # self._edges = copula_tree_structure.edges
        event_shape = torch.Size([copula_tree_structure.number_of_nodes])
        batch_shape = copula_tree_structure.batch_shape
        # TODO: consider to return pair_copula_class constraints
        self._arg_constraints = {}
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            precision=precision
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if not self.support.check(value).all():
            raise Exception(f"value argument contains values not in support {self.support}")

        # Clipping value for numerical stability
        value = self._change_value_to_be_numerical_stable(value)

        distribution_batch_shape = self.batch_shape

        if value.size(-1) != self.event_shape[0]:
            raise ValueError(f"Value must match event shape in the last dimension")

        # if value.ndim > 2 and distribution_batch_shape != torch.Size([]):
        #     raise ValueError(f"Currently support 3 dimensional input only if batch distribution is {torch.Size([])}. found: {distribution_batch_shape}")

        value_batch_shape = value.size()[:-1]
        # if value_batch_shape != distribution_batch_shape:
        #     size = torch.Size([torch.tensor(value_batch_shape).prod()])
        #     copula_tree_structure = self._copula_tree_structure.expand(batch_shape=size)
        #     value = value.view(*size, -1)
        # else:
        copula_tree_structure = self._copula_tree_structure

        # Extracting number of nodes per tree
        n_nodes_per_tree = copula_tree_structure.number_of_nodes
        # Extracting all edges in the entire forest
        edges = copula_tree_structure.edges
        # Get edges batch idx
        edges_batch_idx = (edges // n_nodes_per_tree)
        # Get edges relative event idx
        edges_event_idx = (edges % n_nodes_per_tree)
        # Extract edge location in pair params
        edge_location = CopulaFullGraphStructure.nodes_to_edge_location(edges_event_idx[:, 0], edges_event_idx[:, 1])
        # Constructing pair copula according to a pair of batch_idx and edge_location
        pair_params = copula_tree_structure.copula_pair_params.unsqueeze(0)
        pair_copulas = copula_tree_structure.pair_copula_class(pair_params[..., edges_batch_idx[..., 0], edge_location], precision=self._precision)
        # Extracting value tensor as pairs od nodes according to edges
        value = value.unsqueeze(0)

        # if (value.ndim > 1) and (distribution_batch_shape != value_batch_shape):
        #     edges_batch_idx = edges_batch_idx.unsqueeze(0).repeat(*value_batch_shape, 1, 1)
        #     edges_event_idx = edges_event_idx.unsqueeze(0).repeat(*value_batch_shape, 1, 1)

        # TODO: in case distribution_batch_shape == torch.Size([1]) then I need to change the reshaping of the log_prob
        if distribution_batch_shape == torch.Size([]) or distribution_batch_shape == torch.Size([1]):
            # We dont have batch_shape, so we don't need to use edges_batch_idx
            value_pairs = value[..., edges_event_idx]
        else:
            value_pairs = value[..., edges_batch_idx, edges_event_idx]

        # Calculating pairs log_prob
        log_prob = pair_copulas.log_prob(value_pairs)
        # Reshaping the log_prob to get each batch element in separate tensor, so we could sum the log_prob_on (-1 for the event shape)
        # TODO: in case distribution_batch_shape == torch.Size([1]) then I need to change the reshaping of the log_prob
        pre_sum_log_p = log_prob.view(*value_batch_shape, -1)
        log_prob = pre_sum_log_p.sum(-1)

        # ### test ################
        # log_prob = log_prob.clamp(min=-100)
        # ########################

        return log_prob

    def log_prob2(self, value):
        # TODO: a test!!!!!!!!!!!!!!!!!!!!
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
            # print(f"src_node: {dgl_src_node} dst_node: {dgl_dst_node} copula_param: {dgl_graph.edata['copula_param'][e]}")

            # Swap nodes order to maintain pyro order (bigger is always dest)
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

            # print(f"batch_index: {batch_location} edge_location_in_pair_params: {edge_location_in_pair_params}")
            copula_param = pair_params[..., batch_location, edge_location_in_pair_params]
            # print(f"copula_param: {copula_param}")
            # assert torch.isclose(copula_param.cpu().float(), dgl_graph.edata['copula_param'][e].cpu().float()).all()
            assert torch.isclose(copula_param.float(), dgl_graph.edata['copula_param'][e].float()).all()
            parent_sample = samples[..., batch_location, dgl_src_node_relative_location]
            assert (parent_sample != 0).all()
            copula_dist = ConditionalPairCopula(self._pair_copula_class(copula_param), u=parent_sample)
            assert (samples[..., batch_location, dgl_dst_node_relative_location] == 0).all()
            samples[..., batch_location, dgl_dst_node_relative_location] = copula_dist.rsample(sample_shape=sample_shape)
            samples[..., batch_location, dgl_dst_node_relative_location] = samples[..., batch_location, dgl_dst_node_relative_location].clamp(
                min=self._precision, max=1- self._precision
            )

            # print('-' * 50)

        return samples

    def rsample(self, sample_shape: torch.Size = torch.Size(), device=None):
        if len(sample_shape) > 1:
            raise ValueError(f"Not supporting multiple dimension of sample_shape")

        # Make sure all tensors on the same device
        device = self._copula_tree_structure.copula_pair_params.device
        dtype = self.pair_params.dtype
        # We always assume that we have batch_shape and sample_shape
        batch_shape = self.batch_shape if len(self.batch_shape) > 0 else torch.Size([1])
        sample_shape_ = sample_shape if len(sample_shape) > 0 else torch.Size([1])
        # Adapting pair_params to the existence of sample and batch shape
        pair_params = self.pair_params.view(*batch_shape, self.pair_params.size(-1))
        # placeholder for all samples that we will generate
        samples = torch.zeros((*sample_shape_, *batch_shape, *self.event_shape), device=device, dtype=dtype)
        assert pair_params.ndim == 2
        assert samples.ndim == 3

        # Insert root samples
        batch_idx = self._node_index_to_batch_index(self._copula_tree_structure.root_id)  # Extract batch position (should be 1 of each)
        event_index = self._node_index_to_event_index(self._copula_tree_structure.root_id)  # Extract dim position (should be 0 for all)
        uniform_dist = Uniform(
            low=torch.tensor(self._precision, device=device, dtype=dtype),
            high=torch.tensor(1. - self._precision, device=device, dtype=dtype)
        ).expand(batch_shape)
        samples[..., batch_idx, event_index] = uniform_dist.sample(sample_shape=sample_shape)

        if self._copula_tree_structure.backend == E_TreeStructureBackEndTypes.dgl:
            samples = self._dgl_sampling(samples, pair_params, sample_shape)
        else:
            # Iterate in bfs layers order
            iteration_order = list(self._copula_tree_structure)
            prev_children_edges_list_df = None

            # The block_diag nx batch of trees object
            nx_mwst_tree = self._copula_tree_structure.nx_mwst_tree
            full_edge_list = nx.to_pandas_edgelist(nx_mwst_tree)

            for parents_node_ids, children_node_ids in zip(iteration_order[:-1], iteration_order[1:]):
                # Get parents info in table - The children in the previous iteration are now the parents
                if prev_children_edges_list_df is None:
                    # parents_edges_list_df = nx.to_pandas_edgelist(nx_mwst_tree, nodelist=parents_node_ids)
                    parents_edges_list_df = self._get_edge_df(
                        full_edge_list=full_edge_list,
                        node_ids=parents_node_ids
                    )
                else:
                    parents_edges_list_df = prev_children_edges_list_df

                # Get children info in table
                # children_edges_list_df = nx.to_pandas_edgelist(nx_mwst_tree, nodelist=children_node_ids)
                children_edges_list_df = self._get_edge_df(
                    full_edge_list=full_edge_list,
                    node_ids=children_node_ids
                )

                # Merge them in order to stay with the local current graph - this operation
                local_graph_df = pd.merge(
                    parents_edges_list_df, children_edges_list_df,
                    left_on=['source', 'target'],
                    right_on=['target', 'source'],
                    suffixes=('_parent', '_child')
                )
                assert local_graph_df['target_parent'].nunique() == local_graph_df['target_parent'].shape[0]

                # Extract batch index of each row
                local_graph_df['batch_idx'] = local_graph_df['source_parent'] // self._copula_tree_structure.number_of_nodes
                # Extract relative dim position for each row
                local_graph_df[['source_parent', 'target_parent', 'source_child', 'target_child']] %= self._copula_tree_structure.number_of_nodes
                # For edge location we need each edge to be sorted (we don't assume direction here)
                for_edge_location_idx = np.sort(local_graph_df[['source_parent', 'target_parent']].values)
                local_graph_df['edge_location'] = CopulaFullGraphStructure.nodes_to_edge_location(for_edge_location_idx[:, 0],
                                                                                                  for_edge_location_idx[:, 1])
                # Get relevant edge parameters (here direction is important)
                copula_param = pair_params[..., local_graph_df['batch_idx'].values, local_graph_df['edge_location'].values]
                # Extract relevant parent samples to condition on
                parent_sample = samples[..., local_graph_df['batch_idx'].values, local_graph_df['source_parent'].values]

                assert (local_graph_df.groupby("batch_idx").count().iloc[:, 0] == local_graph_df.groupby("batch_idx")['target_parent'].nunique()).all()
                assert (samples[..., local_graph_df['batch_idx'].values, local_graph_df['source_child'].values] == 0).all()

                # Conditional sample according to copula parm in edge and parent sample
                copula_dist = ConditionalPairCopula(self._pair_copula_class(copula_param), u=parent_sample)
                samples[..., local_graph_df['batch_idx'].values, local_graph_df['source_child'].values] = copula_dist.rsample(sample_shape=sample_shape)
                # Update prev_children_edges_list_df for next iteration
                prev_children_edges_list_df = children_edges_list_df

        samples = samples.view(*sample_shape, *self.batch_shape, *self.event_shape)

        return samples

    def expand(self, batch_shape: torch.Size, _instance=None):
        """
        Returns a new ``TreeCopula`` whose batch dimensions are *viewed* as
        ``batch_shape``.  No data are copied – all underlying tensors are
        broadcast-expanded so gradients still flow to the original
        parameters.  This makes the distribution compatible with
        ``pyro.plate`` and tensor broadcasting.

        Args:
            batch_shape (torch.Size): desired batch shape.
            _instance: internal (used by torch.distributions); leave as None.

        Notes
        -----
        • The requested shape must broadcast with the current
          ``self.batch_shape``; otherwise a ``ValueError`` is raised.

        • The heavy-lifting is delegated to
          ``CopulaTreeStructure.expand`` which should return a *view* of the
          same tree topology with its ``copula_pair_params`` tensor
          expanded to the new batch dimensions.
        """
        batch_shape = torch.Size(batch_shape)

        # Fast path – already in the right shape.
        if batch_shape == self.batch_shape:
            return self

        # Sanity-check that broadcasting is possible.
        try:
            torch.broadcast_shapes(batch_shape, self.batch_shape)
        except RuntimeError:
            raise ValueError(
                f"Incompatible batch_shape {batch_shape} for current "
                f"{self.batch_shape}."
            )

        # 1. Expand the internal tree structure (pair-copula parameters, etc.).
        expanded_tree_structure = self._copula_tree_structure.expand(batch_shape)

        # 2. Create or reuse an instance shell as per Distribution.expand API.
        new = self._get_checked_instance(TreeCopula, _instance)

        # 3. Re-initialise the new object *in-place*.
        TreeCopula.__init__(
            new,
            copula_tree_structure=expanded_tree_structure,
            precision=self._precision,
        )
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

        # Event size is the number of nodes in the graph
        event_shape = torch.Size([copula_full_graph_structure.number_of_nodes])

        self._copula_full_graph_structure = copula_full_graph_structure
        self._copula_pair_params = copula_full_graph_structure.copula_pair_params
        self._pair_copula_class = copula_full_graph_structure.pair_copula_class

        if tree_prior_pair_params is None and tree_prior_pair_params_in_log is None:
            # 1 indicated uniform prior
            self._tree_prior_pair_params = torch.ones_like(self._copula_pair_params, requires_grad=False)
            self._tree_prior_params_in_log = torch.zeros_like(self._tree_prior_pair_params, requires_grad=False)
            self._uniform_prior = True
        elif tree_prior_pair_params is not None:
            self._tree_prior_pair_params = tree_prior_pair_params
            self._tree_prior_params_in_log = torch.log(self.tree_prior_pair_params)
            # TODO: this is not have to be false for example if someone pass me tree_prior_pair_params = 1
            self._uniform_prior = False
        elif tree_prior_pair_params_in_log is not None:
            self._tree_prior_pair_params = tree_prior_pair_params_in_log.exp()
            self._tree_prior_params_in_log = tree_prior_pair_params_in_log
            # TODO: this is not have to be false for example if someone pass me tree_prior_pair_params = 1
            self._uniform_prior = False

        # self._tree_prior_params_in_log = torch.log(self.tree_prior_pair_params)
        # self._spanning_tree_dist = SpanningTree(edge_logits=self._tree_prior_params_in_log.to("cpu"))
        self._arg_constraints = {}

        super().__init__(
            event_shape=event_shape, precision=precision
        )

    def rsample(self, sample_shape: torch.Size = torch.Size(), device=None):
        if len(sample_shape) > 1:
            raise ValueError("Invalid sample shape")

        samples = []
        n_samples = torch.tensor(sample_shape).prod().long().item()

        # TODO: consider use prufer_sequence to identify uniquelyy each tree and sample relatively easily
        with torch.no_grad():
            # params_in_MI_order = self._pair_copula_class(self.pair_params).params_in_MI_order()
            # edge_logits = ((self._tree_prior_pair_params * params_in_MI_order).log()).to('cpu')
            edge_logits = self._tree_prior_pair_params.log().to('cpu')
            spanning_tree_dist = SpanningTree(edge_logits=edge_logits)

        for i in range(n_samples):
            with torch.no_grad():
                u, v = spanning_tree_dist.sample().T
                edges_location = CopulaFullGraphStructure.nodes_to_edge_location(src_node_id=u, target_node_id=v)
                adj_matrix = np.zeros([self.event_shape[0], self.event_shape[0]])
                adj_matrix[u, v] = self.pair_params[edges_location].to("cpu").numpy()
                adj_matrix[v, u] = adj_matrix[u, v]

            copula_tree_structure = CopulaTreeStructure.from_numpy_array(
                adj_matrix=adj_matrix,
                copula_pair_params=self.pair_params,
                pair_copula_class=self._pair_copula_class
            )
            tree_copula_distribution = TreeCopula(copula_tree_structure=copula_tree_structure, precision=self._precision)

            sample = tree_copula_distribution.rsample().unsqueeze(0)
            samples.append(sample)

        # Assuming that we don't have a batch_shape
        sample = torch.cat(samples, dim=0).view(*sample_shape, *self.event_shape)

        return sample

    def sample(self, sample_shape: torch.Size = torch.Size(), device=None):
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, device=device)

    # def log_prob(self, value: torch.Tensor) -> torch.Tensor:
    #     if not self.support.check(value).all():
    #         raise Exception(f"value argument contains values not in support {self.support}")
    #     if not value.size(-1) == self.event_shape[-1]:
    #         raise ValueError(f"Value must match event shape in the last dimension got value.size()={value.size()}, event_shape={self.event_shape}")
    #
    #     # Clipping value for numerical stability
    #     value = self._change_value_to_be_numerical_stable(value)
    #
    #     # Construct the pair copulas using self._copula_pairs_params representation
    #     pair_copulas = self._pair_copula_class(self.pair_params)
    #     # Create value as pairs according to the graph expected representation
    #     i, j = make_complete_graph(num_vertices=self.event_shape[0])
    #     value_pairs = torch.stack((value[..., i], value[..., j]), dim=-1)
    #     value_batch_shape = value_pairs[..., 0, 0].size()
    #
    #     # Get log probabilities
    #     pair_log_probs = pair_copulas.log_prob(value_pairs)
    #     weighted_log_probs = self._tree_prior_params_in_log + pair_log_probs
    #     # self.clip_range(weighted_log_probs, max_range=15)  # TODO: 15 is arbitrary change
    #
    #     weighted_L_star = make_numerical_stable_laplacian(
    #         edge_weights_in_log=weighted_log_probs,
    #         num_vertices=self.event_shape[0],
    #         batch_shape=value_batch_shape
    #     )[..., :-1, :-1]
    #     only_weights_L_star = make_numerical_stable_laplacian(
    #         edge_weights_in_log=self._tree_prior_params_in_log,
    #         num_vertices=self.event_shape[0],
    #         batch_shape=value_batch_shape
    #     )[..., :-1, :-1]
    #
    #     log_probs = torch.linalg.slogdet(weighted_L_star).logabsdet - torch.linalg.slogdet(only_weights_L_star).logabsdet
    #     is_all_log_probs_finite = torch.isfinite(log_probs)
    #     if not is_all_log_probs_finite.all():
    #         is_inf = torch.isinf(log_probs) & (log_probs == float('inf'))
    #         is_neg_inf = torch.isinf(log_probs) & (log_probs == float('-inf'))
    #         log_probs[is_inf] = 50.
    #         log_probs[is_neg_inf] = torch.log(torch.tensor(1e-7))
    #
    #     # # Sometimes due to numerical instability the diagonal element is 0
    #     # weighted_L_star = self._check_pd_and_correct_if_needed(weighted_L_star)
    #     # only_weights_L_star = self._check_pd_and_correct_if_needed(only_weights_L_star)
    #     # Calculate log_probaility using cholsky  decomposition as both matrices are positive-definite
    #     # log_probs = calculate_pd_matrix_log_det(weighted_L_star) - calculate_pd_matrix_log_det(only_weights_L_star)
    #
    #     return log_probs
    @staticmethod
    def calculate_AoT_log_prob(
            value: torch.Tensor,
            copula_pair_params: torch.Tensor,
            tree_prior_params_in_log: torch.Tensor,
            num_vertices: int,
            copula_type: Type[PairCopula],
            use_clip_range: bool = False
    ):
        orig_dtype = copula_pair_params.dtype
        tree_prior_params_in_log = tree_prior_params_in_log.double()
        # Construct the pair copulas using self._copula_pairs_params representation
        pair_copulas = copula_type(copula_pair_params.double())
        # Create value as pairs according to the graph expected representation
        i, j = make_complete_graph(num_vertices=num_vertices)
        value_pairs = torch.stack((value[..., i], value[..., j]), dim=-1).double()
        value_batch_shape = value_pairs[..., 0, 0].size()

        # Get log probabilities
        pair_log_probs = pair_copulas.log_prob(value_pairs)
        weighted_log_probs = tree_prior_params_in_log + pair_log_probs

        m = torch.max(weighted_log_probs, dim=-1, keepdim=True)[0]
        weighted_log_probs = weighted_log_probs - m
        if use_clip_range:
            weighted_log_probs = torch.clamp(weighted_log_probs, math.log(1e-15))


        m_tree_prior = torch.max(tree_prior_params_in_log, dim=-1, keepdim=True)[0]
        tree_prior_params_in_log = tree_prior_params_in_log - m_tree_prior
        if use_clip_range:
            tree_prior_params_in_log = torch.clamp(tree_prior_params_in_log, min=math.log(1e-15))

        weighted_L_star = make_numerical_stable_laplacian(
            edge_weights_in_log=weighted_log_probs,
            num_vertices=num_vertices,
            batch_shape=value_batch_shape
        )[..., :-1, :-1]
        only_weights_L_star = make_numerical_stable_laplacian(
            edge_weights_in_log=tree_prior_params_in_log,
            num_vertices=num_vertices,
            batch_shape=value_batch_shape
        )[..., :-1, :-1]

        trees_log_prob = torch.linalg.slogdet(weighted_L_star).logabsdet
        trees_prior_log_prob = torch.linalg.slogdet(only_weights_L_star).logabsdet
        trees_log_prob = trees_log_prob + (num_vertices - 1) * m.view(value_batch_shape, -1)
        # TODO: this not support trees_prior_log_prob with sample shape (3 dim)
        trees_prior_log_prob = trees_prior_log_prob + (num_vertices - 1) * m_tree_prior.flatten()
        log_probs = trees_log_prob - trees_prior_log_prob

        is_all_log_probs_finite = torch.isfinite(log_probs)
        if not is_all_log_probs_finite.all():
            raise ValueError("Found invalid values in 'log_probs' calculation", log_probs[~is_all_log_probs_finite])

        return log_probs.to(orig_dtype)

    def log_prob(self, value):
        value = self._change_value_to_be_numerical_stable(value)

        log_probs = self.calculate_AoT_log_prob(
            value=value,
            copula_pair_params=self._copula_pair_params,
            tree_prior_params_in_log=self._tree_prior_params_in_log,
            num_vertices=self.event_shape[0],
            copula_type=self.pair_copula_class
        )

        return log_probs

    def _check_pd_and_correct_if_needed(self, pd_matrix: torch.Tensor) -> torch.Tensor:
        any_zero_value_in_diagonal = (torch.diagonal(pd_matrix, offset=0, dim1=-1, dim2=-2) == 0).any()
        if any_zero_value_in_diagonal:
            i = [i for i in range(self.event_shape[0] - 1)]
            pd_matrix[..., i, i] += self._precision

        return pd_matrix

    @staticmethod
    def make_laplacian(edge_weights: torch.Tensor, num_vertices: int):
        """

        Parameters
        ----------
        num_vertices :
        edge_weights :

        Returns
        -------

        """
        i, j = make_complete_graph(num_vertices=num_vertices)
        graph = torch.zeros(num_vertices, num_vertices)
        graph[i, j] = edge_weights
        graph[j, i] = edge_weights

        L = -graph
        i = [i for i in range(num_vertices)]
        L[i, i] = graph.sum(dim=-1)

        return L

    @staticmethod
    def sample_tree_from_uniform_dist(
            n_samples: int,
            n_nodes: int,
            device: str,
    ):
        # TODO: add transformation to eval if needed
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

    @staticmethod
    def clip_range(x, max_range=np.inf):
        m = torch.max(x, dim=-1, keepdim=True)[0]
        return torch.max(x, -1.0 * torch.tensor(max_range) * torch.ones_like(x) + m)

    # @staticmethod
    # def make_numerical_stable_laplacian(
    #         edge_weights_in_log: torch.Tensor,
    #         num_vertices: int,
    #         batch_shape: torch.Size,
    # ):
    #     v1, v2 = make_complete_graph(num_vertices=num_vertices)
    #     i = [i for i in range(num_vertices)]
    #     device = edge_weights_in_log.device
    #
    #     # Construct the full graph as Symmetric matrix
    #     graph = torch.zeros(*batch_shape, num_vertices, num_vertices, device=device)
    #     graph[..., v1, v2] = edge_weights_in_log
    #     graph[..., v2, v1] = edge_weights_in_log
    #
    #     # Assuming no direct edges i.e. the diagonal in the full graph is zero hence in log form it's -inf
    #     graph[..., i, i] = -torch.inf
    #
    #     log_L = graph.clone()
    #     # Setting the diagonal as sum of e^ln(x_ik)_i=j i.e. logsum exponent trick
    #     diagonal_elements = graph.logsumexp(dim=-1)
    #     log_L[..., i, i] = diagonal_elements
    #     # Get values back to non-log formation
    #     L = log_L.exp().clone()
    #     # The sign of the off-diagonals is always negative and the matrix is symmetric
    #     off_diagonal_elements = -L[..., v1, v2]
    #     L[..., v1, v2] = off_diagonal_elements
    #     L[..., v2, v1] = off_diagonal_elements
    #
    #     return L

    @staticmethod
    def edge_mean(num_vertices: int, edge_logits: torch.Tensor):
        """
        Computes marginal probabilities of each edge being active.

        .. note:: This is similar to other distributions' ``.mean()``
            method, but with a different shape because this distribution's
            values are not encoded as binary matrices.

        :returns: A symmetric square ``(V,V)``-shaped matrix with values
            in ``[0,1]`` denoting the marginal probability of each edge
            being in a sampled value.
        :rtype: Tensor
        """
        V = num_vertices
        v1, v2 = make_complete_graph(V).unbind(0)
        logits = edge_logits - edge_logits.max()
        w = edge_logits.new_zeros(V, V)
        w[v1, v2] = w[v2, v1] = logits.exp()
        laplacian = w.sum(-1).diag_embed() - w
        inv = (laplacian + 1 / V).pinverse()
        resistance = inv.diag() + inv.diag()[..., None] - 2 * inv
        from tree_copula_vae.utils.graph_utils import get_edge_tensor
        edge_mean = (resistance * w).unsqueeze(0)
        return get_edge_tensor(edge_mean)
    #
    # @staticmethod
    # def edge_mean_batched(num_vertices: int, edge_logits: torch.Tensor, batch_shape: torch.Size):
    #     """
    #     Computes marginal probabilities of each edge being active.
    #
    #     .. note:: This is similar to other distributions' ``.mean()``
    #         method, but with a different shape because this distribution's
    #         values are not encoded as binary matrices.
    #
    #     :returns: A symmetric square ``(V,V)``-shaped matrix with values
    #         in ``[0,1]`` denoting the marginal probability of each edge
    #         being in a sampled value.
    #     :rtype: Tensor
    #     """
    #     V = num_vertices
    #     v1, v2 = make_complete_graph(V).unbind(0)
    #     logits = edge_logits - edge_logits.max(axis=-1, keepdim=True)[0]
    #     w = torch.zeros(*batch_shape, V, V, device=edge_logits.device)
    #     w[..., v1, v2] = w[..., v2, v1] = logits.exp()
    #     # laplacian = w.sum(-1).diag_embed() - w
    #
    #     laplacian = TreeAveragedCopula.make_numerical_stable_laplacian(edge_weights_in_log=edge_logits, num_vertices=num_vertices, batch_shape=batch_shape)
    #     inv = (laplacian + 1 / V).pinverse()
    #     i = [i for i in range(V)]
    #     resistance = inv.diag() + inv.diag()[..., None] - 2 * inv
    #     from tree_copula_vae.utils.graph_utils import get_edge_tensor
    #     edge_mean = (resistance * w).unsqueeze(0)
    #     return get_edge_tensor(edge_mean)
