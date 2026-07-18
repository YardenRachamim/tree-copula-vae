from typing import Type, Union, List
from enum import Enum

import dgl
import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from networkx import Graph
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import block_diag
from tree_copula_vae.torch_copulas.base import PairCopula
from tree_copula_vae.torch_copulas.pair_copulas import BiVariateGaussianCopula
from tree_copula_vae.utils.graph_utils import make_complete_graph, split_sparse_block, get_edge_tensor


class E_TreeStructureBackEndTypes(str, Enum):
    nx = "nx"
    dgl = "dgl"


class CopulaFullGraphStructure(object):
    @property
    def copula_pair_params(self) -> torch.Tensor:
        return self._copula_pair_params

    @property
    def nx_graph(self) -> Graph:
        return self._copula_graph

    @property
    def copula_param_attribute_name(self) -> str:
        return self._edge_attributes[0]

    @property
    def copula_param_for_mwst_attribute_name(self) -> str:
        return self._edge_attributes[1]

    @property
    def pair_copula_class(self) -> Type[PairCopula]:
        return self._pair_copula_class

    @property
    def number_of_nodes(self) -> int:
        return self.nx_graph.number_of_nodes()

    @property
    def number_of_edges(self) -> int:
        return self.nx_graph.number_of_edges()

    @property
    def adjacency_matrix(self) -> np.ndarray:
        weight = self.copula_param_attribute_name
        A = nx.adjacency_matrix(
            G=self.nx_graph,
            weight=weight,
            nodelist=np.arange(0, self.number_of_nodes)
        )
        return A.todense()

    def __init__(self, copula_pair_params: torch.Tensor,
                 pair_copula_class: Type[PairCopula] = BiVariateGaussianCopula):
        self._copula_pair_params = self._set_copula_pair_params(copula_pair_params)
        self._pair_copula_class = self._set_pair_copula_class(pair_copula_class)
        self._copula_graph = self._set_copula_graph(copula_pair_params)
        self._edge_attributes = ["copula_param",
                                 "copula_params_in_MI_order"]

    @staticmethod
    def _validate_copula_pair_params_input(n_pairs: int):
        n_nodes = CopulaFullGraphStructure.get_n_nodes(n_edges=n_pairs)
        is_valid_pairs_amount = n_pairs == n_nodes * (n_nodes - 1) // 2
        return is_valid_pairs_amount

    def _set_copula_pair_params(self, copula_pair_params: torch.Tensor):
        n_pairs = self.get_n_pairs(copula_pair_params=copula_pair_params)
        is_valid_pairs_amount = self._validate_copula_pair_params_input(n_pairs)
        if not is_valid_pairs_amount:
            raise ValueError(f"Found illegal number of pairs: {n_pairs}")
        if copula_pair_params.ndim > 2:
            raise ValueError(f"Expecting 2 or less dimensions found copula_pair_params.ndim={copula_pair_params.ndim}")
        self._copula_pair_params = copula_pair_params
        return self._copula_pair_params

    def _set_copula_graph(self, copula_pair_params: torch.Tensor) -> Graph:
        n_pairs = self.get_n_pairs(copula_pair_params=copula_pair_params)
        n_nodes = self.get_n_nodes(n_edges=n_pairs)
        complete_graph = make_complete_graph(num_vertices=n_nodes)

        if copula_pair_params.ndim != 1:
            raise ValueError(f"copula_pair_params.ndim can't be bigger then 1 found: {copula_pair_params.ndim}")

        with torch.no_grad():
            adj_matrix = np.zeros((n_nodes, n_nodes))
            adj_matrix[complete_graph[0, :], complete_graph[1, :]] = copula_pair_params.to("cpu").numpy()
            adj_matrix.T[complete_graph[0, :], complete_graph[1, :]] = copula_pair_params.to("cpu").numpy()
            G_params = nx.from_numpy_array(adj_matrix)
            w = nx.get_edge_attributes(G_params, 'weight')
            nx.set_edge_attributes(G_params, w, 'copula_param')

            adj_matrix = np.zeros((n_nodes, n_nodes))
            copula_params_in_MI_order = self._pair_copula_class.params_in_MI_order(pair_params=copula_pair_params)
            adj_matrix[complete_graph[0, :], complete_graph[1, :]] = copula_params_in_MI_order.to("cpu").numpy()
            adj_matrix.T[complete_graph[0, :], complete_graph[1, :]] = copula_params_in_MI_order.to("cpu").numpy()
            G_order = nx.from_numpy_array(adj_matrix)
            w = nx.get_edge_attributes(G_order, 'weight')
            nx.set_edge_attributes(G_order, w, 'copula_params_in_MI_order')

            G = nx.compose(G_params, G_order)

        return G

    def _set_pair_copula_class(self, pair_copula_class):
        if not issubclass(pair_copula_class, PairCopula):
            raise TypeError(f"pair_copula_class must be a subclass of PairCopula found: {type(pair_copula_class)}")
        self._pair_copula_class = pair_copula_class
        return self._pair_copula_class

    @staticmethod
    def nodes_to_edge_location(src_node_id: Union[float, torch.Tensor], target_node_id: Union[float, torch.Tensor]):
        src_node_id = torch.tensor(src_node_id) if isinstance(src_node_id, float) else src_node_id
        target_node_id = torch.tensor(target_node_id) if isinstance(target_node_id, float) else target_node_id
        is_valid_ranked_nodes_input = (target_node_id - src_node_id) > 0
        if not is_valid_ranked_nodes_input.all():
            raise ValueError(f"Input must be ranked, i.e. src_node_id must be smaller then target_node_id at every index")
        return src_node_id + target_node_id * (target_node_id - 1) // 2

    @staticmethod
    def get_n_nodes(n_edges: int):
        return int(round(0.5 + (0.25 + 2 * n_edges) ** 0.5))

    @staticmethod
    def get_n_pairs(copula_pair_params: torch.Tensor) -> int:
        n_pairs = copula_pair_params.size(-1)
        return n_pairs


class CopulaTreeStructure(object):
    @property
    def nx_mwst_tree(self) -> Graph:
        if self._nx_mwst_tree is None:
            raise ValueError(f"backend is {self.backend} and it's not suppoerted nx trees")
        return self._nx_mwst_tree

    @property
    def nodes(self):
        if self.backend == E_TreeStructureBackEndTypes.nx:
            _nodes = self.nx_mwst_tree.nodes
        else:
            raise ValueError(f"'nodes' not supported for backend{self.backend} ")
        return _nodes

    @property
    def number_of_nodes(self):
        if self.backend == E_TreeStructureBackEndTypes.nx:
            _number_of_nodes = self.nx_mwst_tree.number_of_nodes() // self.batch_size
        else:
            _number_of_nodes = self.dgl_graph.num_nodes() // self.batch_size
        return _number_of_nodes

    @property
    def number_of_edges(self):
        return (self.number_of_nodes ** 2 - self.number_of_nodes) // 2

    @property
    def edges(self) -> torch.Tensor:
        if self.backend == E_TreeStructureBackEndTypes.nx:
            _edges = torch.tensor(list(self.nx_mwst_tree.edges()))
        else:
            _edges = torch.stack(self.dgl_graph.edges()).T
            is_valid_edge = _edges[:, 0] < _edges[:, 1]
            _edges = _edges[is_valid_edge]
        return _edges

    @property
    def adjacency_matrix(self) -> np.ndarray:
        if self._sparse_block_mwst is None:
            raise ValueError("'sparse_block_mwst' must be set")
        adj_matrix = split_sparse_block(
            arr=self._sparse_block_mwst,
            n_block=self.batch_size,
            n_nodes=self.number_of_nodes
        )
        return adj_matrix

    @property
    def pair_copula_class(self) -> Type[PairCopula]:
        return self._pair_copula_class

    @property
    def copula_param_attribute_name(self) -> str:
        return self._copula_param_attribute_name

    @property
    def root_id(self) -> Union[List[int], int]:
        return self._root_id

    @property
    def copula_pair_params(self) -> torch.Tensor:
        return self._copula_pair_params

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def batch_shape(self) -> torch.Size:
        return self._batch_shape

    @property
    def dgl_graph(self):
        return self._dgl_graph

    @property
    def backend(self) -> E_TreeStructureBackEndTypes:
        return self._backend

    def __init__(self, nx_tree: Graph = None,
                 root_node_source_id: Union[int, List[int]] = 0,
                 pair_copula_class: Type[PairCopula] = None,
                 copula_param_attribute_name: str = None,
                 copula_pair_params: torch.Tensor = None,
                 batch_size: int = None,
                 batch_shape: torch.Size = None,
                 dgl_graph=None,
                 sparse_block_mwst=None
                 ):
        self._nx_mwst_tree = nx_tree
        self._root_id = root_node_source_id
        self._pair_copula_class = pair_copula_class
        self._copula_param_attribute_name = copula_param_attribute_name
        self._copula_pair_params = copula_pair_params
        self._batch_size = batch_size
        self._batch_shape = batch_shape
        self._dgl_graph = dgl_graph
        self._sparse_block_mwst = sparse_block_mwst

        if dgl_graph is None and nx_tree is None:
            raise ValueError("Neither dgl nor nx_mwst_tree is provided")
        if dgl_graph is not None:
            self._backend = E_TreeStructureBackEndTypes.dgl
        else:
            self._backend = E_TreeStructureBackEndTypes.nx

    def expand(self, batch_shape: torch.Size):
        pair_params = self._copula_pair_params.expand(*batch_shape, *self._copula_pair_params.size())
        return self.from_copula_pairs_param(
            copula_pair_params=pair_params,
            pair_copula_class=self.pair_copula_class
        )

    @staticmethod
    def from_numpy_array(adj_matrix: np.ndarray,
                         copula_pair_params: torch.Tensor,
                         pair_copula_class: Type[PairCopula] = BiVariateGaussianCopula
                         ):
        if copula_pair_params.ndim > 1:
            raise ValueError(f"Not supporting construction by batch with from_numpy_array")

        with torch.no_grad():
            copula_param_attribute_name = "copula_param"
            nx_mwst_tree = nx.from_numpy_array(A=adj_matrix)
            w = nx.get_edge_attributes(nx_mwst_tree, 'weight')
            nx.set_edge_attributes(nx_mwst_tree, w, 'copula_params_in_MI_order')
            is_tree = nx.is_tree(G=nx_mwst_tree)
            is_directed = nx.is_directed(G=nx_mwst_tree)

        if not is_tree or is_directed:
            raise ValueError(f"adj_matrix must be undirected tree!")

        return CopulaTreeStructure(
            nx_tree=nx_mwst_tree,
            root_node_source_id=0,
            pair_copula_class=pair_copula_class,
            copula_param_attribute_name=copula_param_attribute_name,
            copula_pair_params=copula_pair_params,
            batch_shape=torch.Size([]),
            batch_size=1
        )

    @staticmethod
    def from_copula_pairs_param(copula_pair_params: torch.Tensor,
                                pair_copula_class: Type[PairCopula] = BiVariateGaussianCopula,
                                given_trees: torch.Tensor = None,
                                backend: E_TreeStructureBackEndTypes = E_TreeStructureBackEndTypes.dgl
                                ):
        if copula_pair_params.ndim > 2:
            raise ValueError(
                f"Currently support construction using 2 dimensions or less only (batch_shape, n_pairs) found: {copula_pair_params.ndim}")

        with torch.no_grad():
            n_pairs = CopulaFullGraphStructure.get_n_pairs(copula_pair_params=copula_pair_params)
            n_nodes = CopulaFullGraphStructure.get_n_nodes(n_edges=n_pairs)
            complete_graph = make_complete_graph(num_vertices=n_nodes)
            batch_size = 1 if copula_pair_params.ndim == 1 else copula_pair_params.size(0)
            batch_shape = torch.Size([]) if copula_pair_params.ndim == 1 else torch.Size([copula_pair_params.size(0)])

            adj_matrix_according_to_params_order = np.zeros((batch_size, n_nodes, n_nodes))

            if given_trees is None:
                copula_params_in_MI_order = -1 * pair_copula_class.params_in_MI_order(pair_params=copula_pair_params)
                adj_matrix_according_to_params_order[..., complete_graph[0, :], complete_graph[1, :]] = copula_params_in_MI_order.to("cpu").numpy()
                adj_matrix_according_to_params_order[..., complete_graph[1, :], complete_graph[0, :]] = copula_params_in_MI_order.to("cpu").numpy()
                sparse_block_according_to_params_order = block_diag(adj_matrix_according_to_params_order, format='csr')
                sparse_spanning_tree = minimum_spanning_tree(csgraph=sparse_block_according_to_params_order).astype(bool)
            else:
                adj_matrix_according_to_params_order[..., complete_graph[1, :], complete_graph[0, :]] = given_trees.to("cpu").numpy()
                adj_matrix_according_to_params_order[..., complete_graph[0, :], complete_graph[1, :]] = given_trees.to("cpu").numpy()
                sparse_spanning_tree = block_diag(adj_matrix_according_to_params_order, format='csr').astype(bool)

            real_params_adj_matrix = np.zeros((batch_size, n_nodes, n_nodes))
            real_params_adj_matrix[..., complete_graph[0, :], complete_graph[1, :]] = copula_pair_params.to("cpu").numpy()
            real_params_adj_matrix[..., complete_graph[1, :], complete_graph[0, :]] = copula_pair_params.to("cpu").numpy()
            sparse_block_for_real_params = block_diag(real_params_adj_matrix, format='csr')

            sparse_spanning_tree = sparse_spanning_tree + sparse_spanning_tree.T
            sparse_block_for_real_params_according_to_tree = sparse_block_for_real_params.multiply(sparse_spanning_tree)

            copula_param_attribute_name = "copula_param"

            nx_mwst_tree = None
            dgl_graph = None
            if backend == E_TreeStructureBackEndTypes.dgl:
                dgl_graph = dgl.from_scipy(
                    sp_mat=sparse_block_for_real_params_according_to_tree,
                    eweight_name=copula_param_attribute_name
                )
            elif backend == E_TreeStructureBackEndTypes.nx:
                nx_mwst_tree = CopulaTreeStructure.gpt_approach(
                    sparse_block_for_real_params_according_to_tree,
                    n_nodes * batch_size,
                    attr=copula_param_attribute_name
                )
            else:
                raise ValueError(f"Backend {backend} not supported")

            root_node_source_id = [i for i in range(0, n_nodes * batch_size, n_nodes)]

        return CopulaTreeStructure(
            nx_tree=nx_mwst_tree,
            root_node_source_id=root_node_source_id,
            pair_copula_class=pair_copula_class,
            copula_param_attribute_name=copula_param_attribute_name,
            copula_pair_params=copula_pair_params,
            batch_size=batch_size,
            batch_shape=batch_shape,
            dgl_graph=dgl_graph,
            sparse_block_mwst=sparse_block_for_real_params_according_to_tree
        )

    @staticmethod
    def _triples(A):
        nrows = A.shape[0]
        data, indices, indptr = A.data, A.indices, A.indptr
        for i in range(nrows):
            for j in range(indptr[i], indptr[i + 1]):
                yield i, indices[j], data[j]

    @staticmethod
    def claude_approach(A, edge_attribute="weight"):
        G = nx.Graph([(u, v) for u, v, w in CopulaTreeStructure._triples(A)])
        nx.set_edge_attributes(G, {(u, v): w for u, v, w in CopulaTreeStructure._triples(A)}, edge_attribute)
        return G

    @staticmethod
    def gpt_approach(A, n_nodes, attr="weight"):
        triples = CopulaTreeStructure._triples(A)
        adj = {u: {} for u in range(n_nodes)}
        for u, v, w in triples:
            d = {attr: w}
            adj[u][v] = d
            adj[v][u] = d
        G = nx.Graph()
        G._adj = adj
        G._node = {i: {} for i in range(n_nodes)}
        return G

    @staticmethod
    def from_weighted_copula_pairs_param(copula_pair_params: torch.Tensor,
                                         pair_params_weights: torch.Tensor,
                                         pair_copula_class: Type[PairCopula] = BiVariateGaussianCopula,
                                         backend: E_TreeStructureBackEndTypes = E_TreeStructureBackEndTypes.nx
                                         ):
        if copula_pair_params.ndim > 2:
            raise ValueError(
                f"Currently support construction using 2 dimensions or less only (batch_shape, n_pairs) found: {copula_pair_params.ndim}")

        with torch.no_grad():
            n_pairs = CopulaFullGraphStructure.get_n_pairs(copula_pair_params=copula_pair_params)
            n_nodes = CopulaFullGraphStructure.get_n_nodes(n_edges=n_pairs)
            complete_graph = make_complete_graph(num_vertices=n_nodes)
            batch_size = 1 if copula_pair_params.ndim == 1 else copula_pair_params.size(0)
            batch_shape = torch.Size([]) if copula_pair_params.ndim == 1 else torch.Size([copula_pair_params.size(0)])

            adj_matrix = np.zeros((batch_size, n_nodes, n_nodes))
            copula_params_in_MI_order = -1 * pair_copula_class.params_in_MI_order(pair_params=copula_pair_params * pair_params_weights)
            adj_matrix[..., complete_graph[0, :], complete_graph[1, :]] = copula_params_in_MI_order.to("cpu").numpy()
            adj_matrix[..., complete_graph[1, :], complete_graph[0, :]] = copula_params_in_MI_order.to("cpu").numpy()
            sA = block_diag(adj_matrix, format='csr')
            st = minimum_spanning_tree(csgraph=sA)
            adj_matrix[..., complete_graph[0, :], complete_graph[1, :]] = copula_pair_params.to("cpu").numpy()
            adj_matrix[..., complete_graph[1, :], complete_graph[0, :]] = copula_pair_params.to("cpu").numpy()
            sA = block_diag(adj_matrix, format='csr')
            sparse_spanning_tree = st + st.T
            mask = sparse_spanning_tree.astype(bool)
            sA = sA.multiply(mask)

            copula_param_attribute_name = "copula_param"

            nx_mwst_tree = None
            dgl_graph = None
            if backend == E_TreeStructureBackEndTypes.dgl:
                dgl_graph = dgl.from_scipy(
                    sp_mat=sA,
                    eweight_name=copula_param_attribute_name
                )
            elif backend == E_TreeStructureBackEndTypes.nx:
                nx_mwst_tree = CopulaTreeStructure.gpt_approach(
                    sA,
                    n_nodes * batch_size,
                    attr=copula_param_attribute_name
                )
            else:
                raise ValueError(f"Backend {backend} not supported")

            root_node_source_id = [i for i in range(0, n_nodes * batch_size, n_nodes)]

        return CopulaTreeStructure(
            nx_tree=nx_mwst_tree,
            root_node_source_id=root_node_source_id,
            pair_copula_class=pair_copula_class,
            copula_param_attribute_name=copula_param_attribute_name,
            copula_pair_params=copula_pair_params,
            batch_size=batch_size,
            batch_shape=batch_shape,
            dgl_graph=dgl_graph,
            sparse_block_mwst=sA
        )

    def plot_tree(self,
                  with_node_labels: bool = True,
                  with_edge_labels: bool = True
                  ) -> plt.Figure:
        fig, ax = plt.subplots()
        pos = nx.spring_layout(self.nx_mwst_tree)
        nx.draw(self.nx_mwst_tree, pos=pos, with_labels=with_node_labels, ax=ax)

        if with_edge_labels:
            edge_labels = nx.get_edge_attributes(
                G=self.nx_mwst_tree,
                name=self.copula_param_attribute_name
            )
            edge_labels = {e: f'{w:.2f}' for e, w in edge_labels.items()}
            nx.draw_networkx_edge_labels(
                self.nx_mwst_tree,
                pos=pos,
                edge_labels=edge_labels,
                ax=ax
            )

        return fig

    def __iter__(self):
        return nx.bfs_layers(G=self.nx_mwst_tree, sources=self._root_id)

    def __len__(self):
        return self.number_of_nodes

    def __repr__(self):
        return (f"CopulaTreeStructure("
                f"nx_tree={self.nx_mwst_tree},"
                f" root_node_source_id={self.root_id},"
                f"pair_copula_class={self._pair_copula_class},"
                f"copula_param_attribute_name={self.copula_param_attribute_name}")
