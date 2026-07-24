import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import block_diag

def calculate_pd_matrix_log_det(pd_matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculating log determinant of positive definite matrix using knowledge adopted from:
            # https://math.stackexchange.com/questions/2001041/logarithm-of-the-determinant-of-a-positive-definite-matrix

    Parameters
    ----------
    pd_matrix : a tensor of shape (batch_size, d, d)

    Returns
    -------

    """
    # TODO: check for a more efficient way to compute just the diagonal of cholsky instead of the entire matrix
    # Derive cholsky form
    cholesky = torch.linalg.cholesky(pd_matrix)
    # Use the known relation between positive definite matrix, cholsky decomposition and calculation of log_determinant
    # log|A| = 2 * Trace(log(L)) if matrix is PD matrix A=LL^T
    log_det = 2 * torch.diagonal(cholesky, dim1=-1, dim2=-2).log().sum(-1)

    return log_det


def constant_val_estimation(uniform_samples: np.ndarray, alpha: float):
    assert alpha > 0. and alpha < 1.
    assert uniform_samples.ndim == 2

    d = uniform_samples.shape[1]
    n_samples = uniform_samples.shape[0]
    p = d * (1 - alpha)
    D_matrix = cdist(uniform_samples, uniform_samples, 'euclidean') ** p
    G = csr_matrix(D_matrix)
    Tcsr = minimum_spanning_tree(G)

    t2 = Tcsr.sum()
    t3 = (n_samples ** alpha)
    est = t2 / t3

    return est


def rego_entropy_estimation(samples: torch.Tensor, alpha: float, c: float = 1.):
    assert samples.ndim == 3
    assert alpha > 0. and alpha < 1.
    assert c > 0.

    n_blocks = samples.shape[0]
    d = samples.shape[-1]
    n_samples = samples.shape[-2]
    p = d * (1 - alpha)

    # D_matrix = cdist(samples, samples, 'euclidean') ** p
    # G = csr_matrix(D_matrix)
    D_matrix = torch.cdist(samples, samples, p=2) ** p
    G = block_diag(D_matrix.numpy(), format='csr')
    Tcsr = minimum_spanning_tree(G)

    t1 = 1 / (1 - alpha)
    indices = np.arange(0,  n_blocks * n_samples, n_samples)
    t2 = np.array([Tcsr[i:i + n_samples, i:i + n_samples].sum() for i in indices])
    # t2 = Tcsr.sum()
    t3 = (n_samples ** alpha) * c
    entropy_estimation = t1 * (np.log(t2) - np.log(t3))

    return entropy_estimation

def iterative_rego_entropy_estimation(samples: np.ndarray, alpha: float, c: float = 1.):
    assert samples.ndim == 3
    assert alpha > 0. and alpha < 1.
    assert c > 0.

    mis = []
    d = samples.shape[-1]
    n_samples = samples.shape[-2]
    for s in samples:
        p = d * (1 - alpha)

        D_matrix = cdist(s, s, 'euclidean') ** p
        G = csr_matrix(D_matrix)
        Tcsr = minimum_spanning_tree(G)

        t1 = 1 / (1 - alpha)
        t2 = Tcsr.sum()
        t3 = (n_samples ** alpha) * c
        entroyp_estimation = t1 * (np.log(t2) - np.log(t3))
        mis.append(entroyp_estimation)

    return mis

def estimate_MI(samples: np.ndarray, alpha: float, c: float = 1.):
    entropy = rego_entropy_estimation(samples=samples, alpha=alpha, c=c)
    MI = -1 * entropy

    return MI

def get_ranks(x: torch.Tensor, dim=-1) -> torch.Tensor:
    # argsort gives the indices that would sort the array along the specified axis
    sorted_indices = torch.argsort(x, dim=dim)
    # we need to invert the indices to get ranks: argsort the indices themselves along the columns
    ranks = torch.argsort(sorted_indices, dim=dim)

    return ranks
