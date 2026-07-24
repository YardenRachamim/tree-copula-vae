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
    """
    An helper class that represents the graph using pair copula params. i.e. each edge in the graph contains the copula param
    of a single copula for 2 univariate RV.
    """

    # region Properties
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

    # endregion Properties

    # region Constructor
    def __init__(self, copula_pair_params: torch.Tensor,
                 pair_copula_class: Type[PairCopula] = BiVariateGaussianCopula):
        self._copula_pair_params = self._set_copula_pair_params(copula_pair_params)
        self._pair_copula_class = self._set_pair_copula_class(pair_copula_class)
        self._copula_graph = self._set_copula_graph(copula_pair_params)
        # TODO: add validations that all attribute exists in the self._copula_graph
        self._edge_attributes = ["copula_param",
                                 "copula_params_in_MI_order"]  # NOTE: do not change - the order matters for the relevant properties

    @staticmethod
    def _validate_copula_pair_params_input(n_pairs: int):
        n_nodes = CopulaFullGraphStructure.get_n_nodes(n_edges=n_pairs)
        is_valid_pairs_amount = n_pairs == n_nodes * (n_nodes - 1) // 2

        return is_valid_pairs_amount

    def _set_copula_pair_params(self, copula_pair_params: torch.Tensor):
        n_pairs = self.get_n_pairs(copula_pair_params=copula_pair_params)
        is_valid_pairs_amount = self._validate_copula_pair_params_input(n_pairs)

        if not is_valid_pairs_amount:
            error_msg = f"Found illegal number of pairs: {n_pairs}"
            raise ValueError(error_msg)

        if copula_pair_params.ndim > 2:
            error_msg = f"Expecting 2 or less dimensions found copula_pair_params.ndim={copula_pair_params.ndim}"
            raise ValueError(error_msg)

        self._copula_pair_params = copula_pair_params

        return self._copula_pair_params

    def _set_copula_graph(self, copula_pair_params: torch.Tensor) -> Graph:
        n_pairs = self.get_n_pairs(copula_pair_params=copula_pair_params)
        n_nodes = self.get_n_nodes(n_edges=n_pairs)
        complete_graph = make_complete_graph(num_vertices=n_nodes)

        if copula_pair_params.ndim != 1:
            raise ValueError(f"copula_pair_params.ndim can't be bigger then 1 found: {copula_pair_params.ndim}")

        with torch.no_grad():
            # Creating graph for copula_param
            adj_matrix = np.zeros((n_nodes, n_nodes))
            adj_matrix[complete_graph[0, :], complete_graph[1, :]] = copula_pair_params.to("cpu").numpy()
            adj_matrix.T[complete_graph[0, :], complete_graph[1, :]] = copula_pair_params.to("cpu").numpy()
            G_params = nx.from_numpy_array(adj_matrix)
            # 'rename' weight to 'copula_param'
            w = nx.get_edge_attributes(G_params, 'weight')
            nx.set_edge_attributes(G_params, w, 'copula_param')

            # Creating graph for copula_params_in_MI_order
            adj_matrix = np.zeros((n_nodes, n_nodes))
            copula_params_in_MI_order = self._pair_copula_class.params_in_MI_order(pair_params=copula_pair_params)
            adj_matrix[complete_graph[0, :], complete_graph[1, :]] = copula_params_in_MI_order.to("cpu").numpy()
            adj_matrix.T[complete_graph[0, :], complete_graph[1, :]] = copula_params_in_MI_order.to("cpu").numpy()
            G_order = nx.from_numpy_array(adj_matrix)
            # 'rename' weight to 'copula_param'
            w = nx.get_edge_attributes(G_order, 'weight')
            nx.set_edge_attributes(G_order, w, 'copula_params_in_MI_order')

            # Combine them
            G = nx.compose(G_params, G_order)

        return G

    def _set_pair_copula_class(self, pair_copula_class):
        if not issubclass(pair_copula_class, PairCopula):
            raise TypeError(f"pair_copula_class must be a subclass of PairCopula found: {type(pair_copula_class)}")

        self._pair_copula_class = pair_copula_class

        return self._pair_copula_class

    # endregion Constructor

    # region Static methods
    @staticmethod
    def nodes_to_edge_location(src_node_id: Union[float, torch.Tensor], target_node_id: Union[float, torch.Tensor]):
        src_node_id = torch.tensor(src_node_id) if isinstance(src_node_id, float) else src_node_id
        target_node_id = torch.tensor(target_node_id) if isinstance(target_node_id, float) else target_node_id
        # Input must be ranked edges to get the right result e.g. (1, 0) is not valid since 1 is bigger then 0
        is_valid_ranked_nodes_input = (target_node_id - src_node_id) > 0
        if not is_valid_ranked_nodes_input.all():
            raise ValueError(f"Input must be ranked, i.e. src_node_id must be smaller then target_node_id at every index")

        return src_node_id + target_node_id * (target_node_id - 1) // 2

    @staticmethod
    def get_n_nodes(n_edges: int):
        return int(round(0.5 + (0.25 + 2 * n_edges) ** 0.5))

    @staticmethod
    def get_n_pairs(copula_pair_params: torch.Tensor) -> int:
        # if copula_pair_params.ndim > 2:
        #     # NOTE: If I add support for pair_copulas with parameter that is more than 1 dim then it does make sense
        #     error_msg = f"copula_pair_params not make sense for ndim > 2 got: {copula_pair_params.ndim}"
        #     raise ValueError(error_msg)

        n_pairs = copula_pair_params.size(-1)

        return n_pairs

    # endregion Static methods


class CopulaTreeStructure(object):
    # region Properties
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
            # Always assume symmetry enabled
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

    # endregion Properties

    # region Constructor
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
        new_tree_structure = self.from_copula_pairs_param(
            copula_pair_params=pair_params,
            pair_copula_class=self.pair_copula_class
        )

        return new_tree_structure

    # @staticmethod
    # def from_copula_graph_structure(copula_graph_structure: CopulaFullGraphStructure,
    #                                 root_node_source_id: Union[int, List[int]] = 0,
    #                                 pair_copula_class: Type[PairCopula] = BiVariateGaussianCopula
    #                                 ):
    #
    #     nx_mwst_tree = nx.maximum_spanning_tree(
    #         G=copula_graph_structure.nx_graph,
    #         weight=copula_graph_structure.copula_param_for_mwst_attribute_name
    #     )
    #     copula_pair_params = copula_graph_structure.copula_pair_params
    #     copula_param_attribute_name = copula_graph_structure.copula_param_attribute_name
    #
    #     return CopulaTreeStructure(
    #         nx_tree=nx_mwst_tree,
    #         root_node_source_id=root_node_source_id,
    #         pair_copula_class=pair_copula_class,
    #         copula_param_attribute_name=copula_param_attribute_name,
    #         copula_pair_params=copula_pair_params
    #     )

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

        with ((torch.no_grad())):
            # Get number of edges of the full graph
            n_pairs = CopulaFullGraphStructure.get_n_pairs(copula_pair_params=copula_pair_params)
            # Get number of nodes of the full graph
            n_nodes = CopulaFullGraphStructure.get_n_nodes(n_edges=n_pairs)

            # Get edges indices in adj matrix for a complete graph
            complete_graph = make_complete_graph(num_vertices=n_nodes)
            # Assuming batch is one dimensional and is at the start
            batch_size = 1 if copula_pair_params.ndim == 1 else copula_pair_params.size(0)
            batch_shape = torch.Size([]) if copula_pair_params.ndim == 1 else torch.Size([copula_pair_params.size(0)])

            adj_matrix_according_to_params_order = np.zeros((batch_size, n_nodes, n_nodes))

            if given_trees is None:
                # preparing the weighted adj matrix for weights to the maximum spanning tree
                # We multiply by -1 since we want Maximum spanning tree while scipy support minimum only
                copula_params_in_MI_order = -1 * pair_copula_class.params_in_MI_order(pair_params=copula_pair_params)
                adj_matrix_according_to_params_order[..., complete_graph[0, :], complete_graph[1, :]] = copula_params_in_MI_order.to("cpu").numpy()
                adj_matrix_according_to_params_order[..., complete_graph[1, :], complete_graph[0, :]] = copula_params_in_MI_order.to("cpu").numpy()
                # Stack all matrices in block_diag sparse format
                sparse_block_according_to_params_order = block_diag(adj_matrix_according_to_params_order, format='csr')
                sparse_spanning_tree = minimum_spanning_tree(csgraph=sparse_block_according_to_params_order).astype(bool)
            else:
                # adj_matrix_according_to_params_order[..., complete_graph[0, :], complete_graph[1, :]] = given_trees.to("cpu").numpy()
                adj_matrix_according_to_params_order[..., complete_graph[1, :], complete_graph[0, :]] = given_trees.to("cpu").numpy()
                adj_matrix_according_to_params_order[..., complete_graph[0, :], complete_graph[1, :]] = given_trees.to("cpu").numpy()
                sparse_spanning_tree = block_diag(adj_matrix_according_to_params_order, format='csr').astype(bool)

            real_params_adj_matrix = np.zeros((batch_size, n_nodes, n_nodes))
            real_params_adj_matrix[..., complete_graph[0, :], complete_graph[1, :]] = copula_pair_params.to("cpu").numpy()
            real_params_adj_matrix[..., complete_graph[1, :], complete_graph[0, :]] = copula_pair_params.to("cpu").numpy()
            sparse_block_for_real_params = block_diag(real_params_adj_matrix, format='csr')

            # Element-wise multiply 'sA' with the mask
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

            # Each original 0 node is a root node since we have a forest
            root_node_source_id = [i for i in range(0, n_nodes * batch_size, n_nodes)]

            # assert len(root_node_source_id) == batch_size
            # assert nx_mwst_tree.number_of_nodes() == batch_size * n_nodes, \
            #     f"expecting #nodes={batch_size * n_nodes}, got {nx_mwst_tree.number_of_nodes()}"
            # assert nx_mwst_tree.number_of_edges() == batch_size * (n_nodes - 1), \
            #     f"expecting #edges={batch_size * (n_nodes - 1)}, got {nx_mwst_tree.number_of_edges()}"

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
        G = nx.Graph([(u, v) for u, v, w in CopulaTreeStructure._triples(A)])  # Create graph structure in one go
        nx.set_edge_attributes(G, {(u, v): w for u, v, w in CopulaTreeStructure._triples(A)}, edge_attribute)

        return G

    @staticmethod
    def gpt_approach(A, n_nodes, attr="weight"):
        triples = CopulaTreeStructure._triples(A)
        # Pre‑create the empty adjacency dicts so we pay O(1) per edge later
        adj = {u: {} for u in range(n_nodes)}
        for u, v, w in triples:  # still a Python loop, but very light
            d = {attr: w}
            adj[u][v] = d
            adj[v][u] = d  # drop this line for a DiGraph
        G = nx.Graph()
        G._adj = adj  # 👈 internal attribute – be careful
        G._node = {i: {} for i in range(n_nodes)}
        return G

    # @staticmethod
    # def gpt_approach2(A, num_nodes, edge_attr="weight"):
    #     triples = CopulaTreeStructure._triples(A)

    #     src, dst, w = map(np.asarray, zip(*triples))
    #     G_nk = nk.Graph(n=num_nodes, weighted=True, directed=False)
    #     if not hasattr(np, "ulong"):
    #         np.ulong = np.uint64
    #     G_nk.addEdges((w, (src, dst)))  # build in C++

    #     return nxadapter.nk2nx(G_nk)  # back to NetworkX

    @staticmethod
    def from_weighted_copula_pairs_param(copula_pair_params: torch.Tensor,
                                         pair_params_weights: torch.Tensor,
                                         pair_copula_class: Type[PairCopula] = BiVariateGaussianCopula,
                                         backend: E_TreeStructureBackEndTypes = E_TreeStructureBackEndTypes.nx
                                         ):
        """
        Creating copula-tree structure from weighted pair params
        This means that we decide the structure of the tree using 'pair_params_weights' and 'copula_pair_params'
         But, using this structure on the real 'copula_pair_params'
        """
        if copula_pair_params.ndim > 2:
            raise ValueError(
                f"Currently support construction using 2 dimensions or less only (batch_shape, n_pairs) found: {copula_pair_params.ndim}")

        with torch.no_grad():
            # Get number of edges of the full graph
            n_pairs = CopulaFullGraphStructure.get_n_pairs(copula_pair_params=copula_pair_params)
            # Get number of nodes of the full graph
            n_nodes = CopulaFullGraphStructure.get_n_nodes(n_edges=n_pairs)

            # Get edges indices in adj matrix for a complete graph
            complete_graph = make_complete_graph(num_vertices=n_nodes)
            # Assuming batch is one dimensional and is at the start
            batch_size = 1 if copula_pair_params.ndim == 1 else copula_pair_params.size(0)
            batch_shape = torch.Size([]) if copula_pair_params.ndim == 1 else torch.Size([copula_pair_params.size(0)])

            # preparing the weighted adj matrix for weights to the maximum spanning tree
            adj_matrix = np.zeros((batch_size, n_nodes, n_nodes))
            # We multiply by -1 since we want Maximum spanning tree while scipy support minimum only
            copula_params_in_MI_order = -1 * pair_copula_class.params_in_MI_order(pair_params=copula_pair_params * pair_params_weights)
            adj_matrix[..., complete_graph[0, :], complete_graph[1, :]] = copula_params_in_MI_order.to("cpu").numpy()
            adj_matrix[..., complete_graph[1, :], complete_graph[0, :]] = copula_params_in_MI_order.to("cpu").numpy()
            # Stack all matrices in block_diag sparse format
            sA = block_diag(adj_matrix, format='csr')
            st = minimum_spanning_tree(csgraph=sA)
            # Override the adj_matrix to save memory and since those are the actual copula params
            adj_matrix[..., complete_graph[0, :], complete_graph[1, :]] = copula_pair_params.to("cpu").numpy()
            adj_matrix[..., complete_graph[1, :], complete_graph[0, :]] = copula_pair_params.to("cpu").numpy()
            sA = block_diag(adj_matrix, format='csr')
            # Create the boolean mask where non-zeros in 'st' are 1, and zeros are 0 (we only interested in the topology)
            sparse_spanning_tree = st + st.T
            mask = sparse_spanning_tree.astype(bool)
            # Element-wise multiply 'sA' with the mask
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

            # Each original 0 node is a root node since we have a forest
            root_node_source_id = [i for i in range(0, n_nodes * batch_size, n_nodes)]

            # assert len(root_node_source_id) == batch_size
            # assert nx_mwst_tree.number_of_nodes() == batch_size * n_nodes, \
            #     f"expecting #nodes={batch_size * n_nodes}, got {nx_mwst_tree.number_of_nodes()}"
            # assert nx_mwst_tree.number_of_edges() == batch_size * (n_nodes - 1), \
            #     f"expecting #edges={batch_size * (n_nodes - 1)}, got {nx_mwst_tree.number_of_edges()}"

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

    # endregion Constructor

    def plot_tree(self,
                  with_node_labels: bool = True,
                  with_edge_labels: bool = True
                  ) -> plt.Figure:
        fig, ax = plt.subplots()
        # bfs_tree = nx.bfs_tree(G=self.nx_mwst_tree, source=self.root_id)
        pos = nx.spring_layout(self.nx_mwst_tree)
        nx.draw(self.nx_mwst_tree, pos=pos, with_labels=with_node_labels, ax=ax)

        if with_edge_labels:
            # Edge labels
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
        # return nx.edge_bfs(G=self.nx_mwst_tree, source=self._root_id)
        return nx.bfs_layers(G=self.nx_mwst_tree, sources=self._root_id)

    def __len__(self):
        return self.number_of_nodes

    def __repr__(self):
        return (f"CopulaTreeStructure("
                f"nx_tree={self.nx_mwst_tree},"
                f" root_node_source_id={self.root_id},"
                f"pair_copula_class={self._pair_copula_class},"
                f"copula_param_attribute_name={self.copula_param_attribute_name}")
