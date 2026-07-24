import math
import numpy as np
import torch
import networkx as nx

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import block_diag, spmatrix, isspmatrix_csr
from torch.distributions import Gumbel, Categorical


def split_block(arr: np.ndarray, n_block: int, n_nodes: int):
    a = []
    f = 0
    l = n_nodes

    for i in range(n_block):
        f = n_nodes * i
        l = n_nodes * i + n_nodes
        a.append(arr[f:l, f:l])

    return np.array(a)


def split_sparse_block(arr: spmatrix, n_block: int, n_nodes: int):
    a = []
    f = 0
    l = n_nodes

    for i in range(n_block):
        f = n_nodes * i
        l = n_nodes * i + n_nodes
        arr_coo = arr[f:l, f:l].tocoo()
        data = arr_coo.data
        idx = arr_coo.row, arr_coo.col
        t = np.zeros((n_nodes, n_nodes))
        t[idx] = data
        a.append(t)

    return np.array(a)


def make_complete_graph(num_vertices):
    # Adopted from pyro spanning tree code:
    # https: // docs.pyro.ai / en / dev / _modules / pyro / distributions / spanning_tree.html
    if num_vertices < 2:
        raise ValueError("PyTorch cannot handle zero-sized multidimensional tensors")
    V = num_vertices
    K = V * (V - 1) // 2
    v1 = torch.arange(V)
    v2 = torch.arange(V).unsqueeze(-1)
    v1, v2 = torch.broadcast_tensors(v1, v2)
    v1 = v1.contiguous().view(-1)
    v2 = v2.contiguous().view(-1)
    mask = v1 < v2
    grid = torch.stack((v1[mask], v2[mask]))

    return grid


def get_edge_tensor(adj_matrix: torch.Tensor):
    if adj_matrix.ndim != 3:
        raise ValueError(f"Currently support only batched input of dim 3 (batchXdimXdim): got {adj_matrix.size()}")
    if adj_matrix.size(1) != adj_matrix.size(2):
        # TODO: add symmetric test
        raise ValueError("Invalid adj matrix - not squared")

    num_vertices = adj_matrix.size(-1)
    v1, v2 = make_complete_graph(num_vertices=num_vertices)
    edges_indicator = adj_matrix[:, v1, v2]

    is_valid_edge_indicator = (edges_indicator.bool().int().sum(1) == num_vertices - 1)
    assert is_valid_edge_indicator.all(), f"Found invalid number of nodes for some edges at position {torch.argwhere(~is_valid_edge_indicator)}"

    return edges_indicator


def get_symmetric_adj_matrix(edges: torch.Tensor, num_nodes: int = None):
    """
    edges shape: (..., D*(D-1)//2) edge indicators like
    returns symmetric matrix (..., D, D) with zero diagonal.
    """
    batch_shape = edges.shape[:-1]
    if num_nodes is None:
        n_edges = edges.size(-1)
        num_nodes = int(round(0.5 + (0.25 + 2 * n_edges) ** 0.5))

    device = edges.device
    dtype = edges.dtype

    # Construct the full graph as Symmetric matrix
    v1, v2 = make_complete_graph(num_vertices=num_nodes)
    graph = torch.zeros(*batch_shape, num_nodes, num_nodes, device=device, dtype=dtype)
    graph[..., v1, v2] = edges
    graph[..., v2, v1] = graph[..., v1, v2]

    return graph


def make_numerical_stable_laplacian(
        edge_weights_in_log: torch.Tensor,
        num_vertices: int,
        batch_shape: torch.Size,
):
    v1, v2 = make_complete_graph(num_vertices=num_vertices)
    i = [i for i in range(num_vertices)]
    device = edge_weights_in_log.device
    dtype = edge_weights_in_log.dtype

    # Construct the full graph as Symmetric matrix
    graph = torch.zeros(*batch_shape, num_vertices, num_vertices, device=device, dtype=dtype)
    graph[..., v1, v2] = edge_weights_in_log
    graph[..., v2, v1] = graph[..., v1, v2]
    # Assuming no direct edges i.e. the diagonal in the full graph is zero hence in log form it's -inf
    graph[..., i, i] = -torch.inf

    log_L = graph.clone()  # Clone here since we doing inplace ops and autograd might complain
    # Setting the diagonal as sum of e^ln(x_ik)_i=j i.e. logsum exponent trick
    diagonal_elements = graph.logsumexp(dim=-1)
    log_L[..., i, i] = diagonal_elements
    # Get values back to non-log formation
    L = log_L.exp().clone()  # Clone here since we doing inplace ops and autograd might complain
    # The sign of the off-diagonals is always negative and the matrix is symmetric
    off_diagonal_elements = -L[..., v1, v2]
    L[..., v1, v2] = off_diagonal_elements
    L[..., v2, v1] = off_diagonal_elements

    return L

def project_to_pd_correlation(R: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    R: [B, d, d] symmetric (not necessarily PD)
    Returns: PD correlation matrix with diag=1.
    """
    # symmetrize
    R = 0.5 * (R + R.transpose(-1, -2))

    # eigen decomposition (batched)
    evals, evecs = torch.linalg.eigh(R)  # evals [B,d], evecs [B,d,d]

    # clamp eigenvalues to ensure PD
    evals_clamped = torch.clamp(evals, min=eps)

    # reconstruct
    R_pd = (evecs * evals_clamped.unsqueeze(-2)) @ evecs.transpose(-1, -2)

    # renormalize to correlation (diag=1)
    diag = torch.sqrt(torch.clamp(R_pd.diagonal(dim1=-2, dim2=-1), min=eps))
    R_corr = R_pd / (diag.unsqueeze(-1) * diag.unsqueeze(-2))

    # symmetrize again for numeric cleanliness
    R_corr = 0.5 * (R_corr + R_corr.transpose(-1, -2))

    return R_corr


def get_tree_precision_matrix(pair_params, soft_mwst=None, precision_epsilon=1e-6):
    """
    Constructs the Precision Matrix for a GMRF approximation.
    Supports arbitrary batch shapes (e.g., [B, S, D, D]).
    """
    if soft_mwst is None:
        soft_mwst = torch.ones_like(pair_params)

    # 1. חילוץ המימדים בצורה בטוחה
    # D הוא תמיד המימד האחרון
    D = pair_params.shape[-1]
    device = pair_params.device

    # 2. חישוב מקדמים (נשאר זהה, הפעולות הן element-wise ותומכות ב-Broadcasting)
    rho = torch.clamp(pair_params, -0.99, 0.99)
    rho_sq = rho.pow(2)
    denominator = 1 - rho_sq + precision_epsilon

    # יצירת מסכה לאלכסון (בגודל DxD, הבאץ' יסתדר לבד ב-Broadcasting)
    eye_d = torch.eye(D, device=device)  # [D, D]
    mask_off_diag = 1 - eye_d  # [D, D]

    # חישוב האיבר מחוץ לאלכסון
    # Broadcasting: [..., D, D] * [D, D] -> [..., D, D]
    off_diag_term = - (rho / denominator) * soft_mwst * mask_off_diag

    # חישוב האיבר לאלכסון
    diag_contrib = (rho_sq / denominator) * soft_mwst * mask_off_diag
    sum_diag_contrib = diag_contrib.sum(dim=-1)  # סכימה על השורה האחרונה -> [..., D]

    # 3. הרכבת המטריצה הסופית
    # Lambda = I + OffDiag + DiagSum
    # שימוש ב-diag_embed שמטפל נכון בבאץ' של וקטורים
    Lambda = eye_d + off_diag_term + torch.diag_embed(sum_diag_contrib)

    return Lambda

def tree_dist_log_partition_function(
        edge_weights_in_log: torch.Tensor,
        num_vertices: int,
        batch_shape: torch.Size
) -> torch.Tensor:
    # re-scaling by subtracting the max - for numerical stability
    # clipped_weights = clip_range(edge_weights_in_log, 15)
    m = torch.max(edge_weights_in_log, dim=-1, keepdim=True)[0]
    edge_weights_in_log = edge_weights_in_log - m

    L_star = make_numerical_stable_laplacian(
        edge_weights_in_log=edge_weights_in_log,
        num_vertices=num_vertices,
        batch_shape=batch_shape
    )[..., :-1, :-1]
    logzs = torch.linalg.slogdet(L_star).logabsdet

    # Return the max that was subtract from each weights
    logzs = logzs + (num_vertices - 1) * m.flatten()

    return logzs


# TODO: move to numrical stable utils
def clip_range(x, max_range=np.inf):
    m = torch.max(x, dim=-1, keepdim=True)[0]
    return torch.max(x, -1.0 * torch.tensor(max_range) * torch.ones_like(x) + m)


def calculate_edge_kirchoff_marginals_using_grad(weights: torch.Tensor, num_vertices: int):
    with torch.set_grad_enabled(True):
        if weights.ndim != 2:
            raise ValueError(f"Expecting output to be 2 dimensional (batch_size, n_edges) found: {weights.ndim} dims")

        is_grad_enabled = weights.requires_grad
        if not is_grad_enabled:
            weights = weights.clone().requires_grad_(True)

        logzs = tree_dist_log_partition_function(
            edge_weights_in_log=weights,
            num_vertices=num_vertices,
            batch_shape=torch.Size([weights.size(0)])
        )

        # Make sure that those calculations are part of the computation graph and that we can pass gradient in the backprop
        edges = torch.autograd.grad(logzs.sum(), weights, create_graph=True)[0]

        if not is_grad_enabled:
            edges = edges.detach().clone().requires_grad_(False)

    return edges

# --- Helper: Safe Edge Logits ---
def safe_edge_logits(edge_logits: torch.Tensor):
    """מונע קריסה נומרית בחישוב ההסתברויות של העצים"""
    max_val = edge_logits.max(dim=-1, keepdim=True)[0]
    max_shifted_logits = edge_logits - max_val 

    r_min = 1e-8
    min_allowed = math.log(r_min)
    
    clamped_logits = torch.clamp(max_shifted_logits, min=min_allowed)
    
    return clamped_logits

def sample_soft_tree(edge_logits: torch.Tensor,
                     temperature: float,
                     num_nodes: int = None,
                     inject_gumbel_noise: bool = True,
                     n_latent_samples: int = 1,
                     take_mean_tree: bool = True
                     ):
    if num_nodes is None:
        n_edges = edge_logits.size(-1)
        num_nodes = int(round(0.5 + (0.25 + 2 * n_edges) ** 0.5))
    # # Inject gumbel noise  for stochastic softmax trick (sst)
    if inject_gumbel_noise:
        gumbel_dist = Gumbel(edge_logits, scale=torch.ones_like(edge_logits))
        edge_weights = gumbel_dist.rsample(torch.Size([n_latent_samples]))
        if take_mean_tree:
            edge_weights = edge_weights.mean(0)
    else:
        edge_weights = edge_logits

    # from sst: exp_family_entropy relaxation
    weights = edge_weights.double() / temperature  # Making it double for numerical stability (I think it solves the problem of bigger then 1)
    clamped_logits = safe_edge_logits(weights)

    edges = calculate_edge_kirchoff_marginals_using_grad(
        weights=clamped_logits.view(-1, clamped_logits.size(-1)),
        num_vertices=num_nodes
    )
    edges = edges.view(*edge_weights.size())
    edges = edges.to(edge_weights.dtype)

    soft_mwst = edges.clamp(1e-6, 1)

    return soft_mwst, edge_weights


def sample_hard_tree(
        edge_logits: torch.Tensor,
        n_latent_samples: int = 1,
        inject_gumbel_noise: bool = True,
        take_mean_tree: bool = True
):
    if edge_logits.ndim != 2:
        raise ValueError(f"Expecting edge_logits to be 2 dimensional (batch_size, n_edges)")

    if inject_gumbel_noise:
        gumbel_dist = Gumbel(edge_logits, scale=torch.ones_like(edge_logits))
        edge_weights = gumbel_dist.rsample(torch.Size([n_latent_samples]))
        if take_mean_tree:
            edge_weights = edge_weights.mean(0)
    else:
        edge_weights = edge_logits

    # If edge weight is zero and this weight is part of the mwst we will get an error
    is_edge_weight_zero = edge_weights == 0
    edge_weights[is_edge_weight_zero] = 1e-6

    with torch.no_grad():
        mwst = maximum_weighted_spanning_tree(edge_logits=edge_weights.view(-1, edge_weights.size(-1)))
        mwst = mwst.view(*edge_weights.size())

    return mwst, edge_weights


@torch.no_grad()
def maximum_weighted_spanning_tree(edge_logits: torch.Tensor):
    if edge_logits.ndim > 2:
        raise ValueError(f"edge logits must be ar most 2 dims found: {edge_logits.ndim}")

    # Get number of edges of the full graph
    n_pairs = edge_logits.size(-1)
    # Get number of nodes of the full graph
    n_nodes = int(round(0.5 + (0.25 + 2 * n_pairs) ** 0.5))

    if n_nodes * (n_nodes - 1) // 2 != n_pairs:
        raise ValueError(
            f"edge_logits.size(-1)={n_pairs} is not n(n-1)/2 for any integer n; "
            f"inferred n_nodes={n_nodes}"
        )

    # Get edges indices in adj matrix for a complete graph
    edge_weights = edge_logits.double().cpu().numpy()
    if not np.isfinite(edge_weights).all():
        raise ValueError("edge_logits contain non-finite values (inf or NaN)")

    complete_graph = make_complete_graph(num_vertices=n_nodes)
    # Assuming batch is one dimensional and is at the start
    batch_size = 1 if edge_logits.ndim == 1 else edge_logits.size(0)
    # batch_shape = torch.Size([]) if edge_logits.ndim == 1 else torch.Size([edge_logits.size(0)])

    # preparing the weighted adj matrix for weights to the maximum spanning tree
    adj_matrix_for_ordered_params = np.zeros((batch_size, n_nodes, n_nodes))
    # We multiply by -1 since we want Maximum spanning tree while scipy support minimum only
    ordered_edge_weights = -edge_weights
    adj_matrix_for_ordered_params[..., complete_graph[0, :], complete_graph[1, :]] = ordered_edge_weights
    adj_matrix_for_ordered_params[..., complete_graph[1, :], complete_graph[0, :]] = ordered_edge_weights

    # Stack all matrices in block_diag sparse format
    sparse_block_array_for_ordered_params = block_diag(adj_matrix_for_ordered_params, format='csr')
    spanning_tree = minimum_spanning_tree(csgraph=sparse_block_array_for_ordered_params).astype(bool)

    adj_matrix_for_params = np.zeros((batch_size, n_nodes, n_nodes))
    adj_matrix_for_params[..., complete_graph[0, :], complete_graph[1, :]] = edge_weights
    adj_matrix_for_params[..., complete_graph[1, :], complete_graph[0, :]] = edge_weights
    sparse_block_array_for_params = block_diag(adj_matrix_for_params, format='csr')

    # Element-wise multiply 'sA' with the mask
    symmetric_tree = spanning_tree.maximum(spanning_tree.T)
    sparse_block_array_for_params_and_tree = sparse_block_array_for_params.multiply(symmetric_tree)
    adj_matrices = torch.from_numpy(split_sparse_block(sparse_block_array_for_params_and_tree, n_block=batch_size, n_nodes=n_nodes)).to(edge_logits.device)

    # Extract as edges (actual mwst)
    mwst = get_edge_tensor(adj_matrix=adj_matrices).bool().to(dtype=edge_logits.dtype)

    return mwst


def pyro_to_edge_indicator_trees_converter(edges: torch.Tensor, n_edges: int) -> torch.Tensor:
    # TODO: consider remove edge logits from inputs
    # TODO: validate
    v1 = edges[..., 0]
    v2 = edges[..., 1]
    single_sample = torch.zeros(n_edges, device=edges.device)
    k = v1 + v2 * (v2 - 1) // 2
    single_sample[k] = 1

    return single_sample


def sample_tree_from_uniform_dist(
        n_samples: int,
        n_nodes: int,
        device,
        dtype,
        as_edge_indicator: bool = True
):
    # TODO: add transformation to eval if needed
    logits = torch.ones(n_nodes)
    tree_uni_dist = Categorical(logits=logits)
    prufer_seq = tree_uni_dist.sample(torch.Size([n_samples, n_nodes - 2]))

    trees = np.array([
        list(nx.from_prufer_sequence(list(seq.numpy())).edges()) for seq in prufer_seq
    ])
    graph = torch.zeros(n_samples, n_nodes, n_nodes, device=device, dtype=dtype)
    for i, tree in enumerate(trees):
        v1, v2 = tree.T
        graph[i, v1, v2] = 1
        graph[i, v2, v1] = 1

    if as_edge_indicator:
        ret = get_edge_tensor(graph)
    else:
        ret = graph

    return ret
