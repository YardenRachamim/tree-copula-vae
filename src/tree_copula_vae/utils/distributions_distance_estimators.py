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
        super().__init__()
        if isinstance(bandwidth_ranges, float):
            bandwidth_ranges = [bandwidth_ranges]
        self.bandwidth_ranges = bandwidth_ranges
        self.metric = 'rbf'

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> float:
        m = X.shape[0]
        n = Y.shape[0]
        K_xx = np.zeros([m, m])
        K_yy = np.zeros([n, n])
        K_xy = np.zeros([m, n])
        for gamma in self.bandwidth_ranges:
            K_xx += pairwise_kernels(X, X, metric=self.metric, gamma=gamma)
            K_yy += pairwise_kernels(Y, Y, metric=self.metric, gamma=gamma)
            K_xy += pairwise_kernels(X, Y, metric=self.metric, gamma=gamma)
        w_x = 1 / (m * (m - 1))
        w_y = 1 / (n * (n - 1))
        w_xy = 2 / (m * n)
        mmd = (w_x * (K_xx.sum() - K_xx.trace()) +
               w_y * (K_yy.sum() - K_yy.trace()) -
               w_xy * K_xy.sum())
        return mmd


class KLEstimator(StatisticEstimator):
    def __init__(self, low_variance: bool = True, no_bias: bool = True):
        super().__init__()
        self.low_variance = low_variance
        self.no_bias = no_bias

    def __call__(self, X: torch.Tensor, P: Distribution, Q: Distribution) -> torch.Tensor:
        logr = Q.log_prob(X) - P.log_prob(X)
        if self.no_bias and self.low_variance:
            kl_estimation = (logr.exp() - 1) - logr
        elif self.no_bias and not self.low_variance:
            kl_estimation = -logr
        elif self.low_variance and not self.no_bias:
            kl_estimation = logr ** 2 / 2
        else:
            raise ValueError(f"Illegal combination of input arguments")
        return kl_estimation
