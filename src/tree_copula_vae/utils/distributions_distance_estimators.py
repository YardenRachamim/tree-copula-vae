from typing import List, Union, Any, Dict

import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from abc import ABCMeta, abstractmethod
from torch.distributions import Distribution


class StatisticEstimator(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass


class MMDEstimator(StatisticEstimator):
    def __init__(self, bandwidth_ranges):
        """
        Calculating Maximum Mean discrepancy (MMD) unbiased estimation using sklearn and numpy methods
        currently only supports 'rbf' kernel

        bandwidth_ranges : For a mixture of gaussian kernels pass a list (of gamma parameters)
        """
        super().__init__()

        if isinstance(bandwidth_ranges, float):
            # Always consider gamma as a list
            bandwidth_ranges = [bandwidth_ranges]

        self.bandwidth_ranges = bandwidth_ranges
        self.metric = 'rbf'

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Empirical maximum mean discrepancy. The lower the result
        the more evidence that distributions are the same.

        Parameters
        ----------
        X : samples from distribution P
        Y : samples from distribution Q

        Returns
        -------
         mmd estimation for 2 datasets
        """
        m = X.shape[0]
        n = Y.shape[0]

        K_xx = np.zeros([m, m])
        K_yy = np.zeros([n, n])
        K_xy = np.zeros([m, n])

        for gamma in self.bandwidth_ranges:
            K_xx += pairwise_kernels(X, X, metric=self.metric, gamma=gamma)
            K_yy += pairwise_kernels(Y, Y, metric=self.metric, gamma=gamma)
            K_xy += pairwise_kernels(X, Y, metric=self.metric, gamma=gamma)

        w_x = 1 / (m * (m - 1))  # Corrected weight for K_xx
        w_y = 1 / (n * (n - 1))  # Corrected weight for K_yy
        w_xy = 2 / (m * n)  # Corrected weight for K_xy

        mmd = (w_x * (K_xx.sum() - K_xx.trace()) +
               w_y * (K_yy.sum() - K_yy.trace()) -
               w_xy * K_xy.sum())

        return mmd


class KLEstimator(StatisticEstimator):
    def __init__(self, low_variance: bool = True, no_bias: bool = True):
        """
        based on: http://joschu.net/blog/kl-approx.html

        Parameters
        ----------
        low_variance :
        no_bias :
        """
        super().__init__()
        self.low_variance = low_variance
        self.no_bias = no_bias

    def __call__(self, X: torch.Tensor, P: Distribution, Q: Distribution) -> torch.Tensor:
        # KL(P||Q)
        logr = Q.log_prob(X) - P.log_prob(X)

        if self.no_bias and self.low_variance:
            kl_estimation = (logr.exp() - 1) - logr
            # kl_estimation = kl_estimation.clip(low=0.)
        elif self.no_bias and not self.low_variance:
            kl_estimation = -logr
            # kl_estimation = kl_estimation.clip(low=0.)
        elif self.low_variance and not self.no_bias:
            kl_estimation = logr ** 2 / 2
        else:
            raise ValueError(f"Illegal combination of input arguments")

        return kl_estimation

# def calculate_mmd_torch(x, y, bandwidth_range: List[float]) -> float:
#     """Emprical maximum mean discrepancy. The lower the result
#        the more evidence that distributions are the same.
#
#     Args:
#         x: first sample, distribution P
#         y: second sample, distribution Q
#         kernel: kernel type such as "multiscale" or "rbf"
#     """
#     x = x.double()
#     y = y.double()
#     xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
#     rx = (xx.diag().unsqueeze(0).expand_as(xx))
#     ry = (yy.diag().unsqueeze(0).expand_as(yy))
#
#     dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
#     dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
#     dxy = rx.t() + ry - 2. * zz  # Used for C in (1)
#
#     XX, YY, XY = (torch.zeros(xx.shape),
#                   torch.zeros(xx.shape),
#                   torch.zeros(xx.shape))
#
#     # bandwidth_range = [1] #  [10, 15, 20, 50]
#     for a in bandwidth_range:
#         XX += torch.exp(-0.5 * dxx / a)
#         YY += torch.exp(-0.5 * dyy / a)
#         XY += torch.exp(-0.5 * dxy / a)
#
#     return torch.mean(XX + YY - 2. * XY).item()
