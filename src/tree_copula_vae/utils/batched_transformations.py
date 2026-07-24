import torch


def precision_to_MI(precision_matrix_batch):
    # TODO: this function is not tested (may produce wrong results)
    B, N, _ = precision_matrix_batch.shape

    # Compute the diagonal elements for each matrix in the batch (BxN)
    diag_elements = torch.diagonal(precision_matrix_batch, dim1=-2, dim2=-1)

    # Compute the square root of the diagonal elements (BxN)
    sqrt_diag_elements = torch.sqrt(diag_elements)

    # Compute the outer product of the square root of diagonal elements (BxNxN)
    outer_sqrt_diag = sqrt_diag_elements.unsqueeze(-1) * sqrt_diag_elements.unsqueeze(-2)

    # Compute the correlation coefficients (BxNxN)
    rho = -precision_matrix_batch / outer_sqrt_diag

    # Set the diagonal of the correlation matrix to 0 (BxNxN)
    rho.diagonal(dim1=-2, dim2=-1).zero_()

    # Compute the mutual information (BxNxN)
    MI_matrix_batch = -0.5 * torch.log(1 - rho ** 2)

    return MI_matrix_batch


def precision_to_correlation(precision_matrix_batch):
    # TODO: this function is not tested (may produce wrong results)
    B, N, _ = precision_matrix_batch.shape

    # Compute the diagonal elements for each matrix in the batch (BxN)
    diag_elements = torch.diagonal(precision_matrix_batch, dim1=-2, dim2=-1)

    # Compute the square root of the diagonal elements (BxN)
    sqrt_diag_elements = torch.sqrt(diag_elements)

    # Compute the outer product of the square root of diagonal elements (BxNxN)
    outer_sqrt_diag = sqrt_diag_elements.unsqueeze(-1) * sqrt_diag_elements.unsqueeze(-2)

    # Compute the correlation coefficients (BxNxN)
    correlation_matrix_batch = -precision_matrix_batch / outer_sqrt_diag

    # Set the diagonal of the correlation matrix to 1 (BxNxN)
    correlation_matrix_batch.diagonal(dim1=-2, dim2=-1).fill_(1.0)

    return correlation_matrix_batch


def covariance_to_correlation(covariance_matrix, precision: float = 1e-6):
    stds = torch.sqrt(torch.diagonal(covariance_matrix, offset=0, dim1=-2, dim2=-1))
    inv_stds = 1.0 / stds
    inv_stds_mat = torch.diag_embed(inv_stds)
    correlation_matrix = torch.matmul(torch.matmul(inv_stds_mat, covariance_matrix), inv_stds_mat)
    correlation_matrix = correlation_matrix.clamp(-1 + precision, 1 - precision)

    return correlation_matrix
