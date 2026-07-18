from abc import ABCMeta, abstractmethod
from typing import Optional, Any, Union, Dict, Callable

from numbers import Number
import torch
import torch.nn.functional as F
from pyro.distributions.torch_distribution import TorchDistributionMixin
from torch.distributions import Distribution, constraints, Uniform, StudentT
from torch.distributions.utils import broadcast_all
from tree_copula_vae.torch_copulas import extended_constraints


class MultivariateCopula(Distribution, TorchDistributionMixin, metaclass=ABCMeta):
    def __init__(self, batch_shape: torch.Size = torch.Size(),
                 event_shape: torch.Size = torch.Size(),
                 validate_args: Optional[bool] = None,
                 precision: float = 1e-6):
        is_valid_event_shape = torch.tensor(event_shape).prod() >= 2
        if not is_valid_event_shape:
            raise ValueError(f"Copula event shape must be at least 2. Got: {event_shape}")

        super().__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)
        self._precision: float = precision

    def _change_value_to_be_numerical_stable(self, value):
        eps = self._precision
        lower_bound = eps
        upper_bound = 1 - eps
        numerical_stable_value = value.clamp(lower_bound, upper_bound)
        return numerical_stable_value

    @property
    def support(self) -> Optional[Any]:
        return extended_constraints.unit_cube

    @property
    def has_rsample(self) -> bool:
        return True

    @abstractmethod
    def rsample(self, sample_shape: torch.Size = torch.Size(), device: Optional[torch.device] = None) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def pair_params(self):
        pass

    def sample(self, sample_shape: torch.Size = torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape)

    def __repr__(self):
        return self.__class__.__name__


class PairCopula(MultivariateCopula, metaclass=ABCMeta):
    min_theta_val: float = float('-inf')
    max_theta_val: float = float('inf')

    def __init__(self, theta: Union[float, torch.Tensor] = None, validate_args=None, precision: float = 1e-6):
        self._event_shape = torch.Size([2])
        self._theta, = broadcast_all(theta)
        self._batch_shape = torch.Size() if isinstance(theta, Number) else self._theta.size()
        super(PairCopula, self).__init__(batch_shape=self._batch_shape,
                                         event_shape=self._event_shape,
                                         validate_args=validate_args, precision=precision)

    @property
    def pair_params(self):
        return self.theta

    @property
    def theta(self) -> torch.Tensor:
        return self._theta

    @abstractmethod
    def entropy(self):
        raise NotImplementedError

    @abstractmethod
    def inverse_conditional_distribution(self, u: torch.Tensor,
                                         t: torch.Tensor) -> torch.Tensor:
        pass

    def rsample(self, sample_shape=torch.Size(), device=None):
        if device is None:
            device = self.theta.device
        shape = self._extended_shape(sample_shape)
        uniform_sample = Uniform(
            low=torch.tensor(self._precision, device=device),
            high=torch.tensor(1. - self._precision, device=device)
        ).sample(sample_shape=shape)
        u = uniform_sample[..., 0]
        t = uniform_sample[..., 1]
        v = self.inverse_conditional_distribution(u, t)
        sample = torch.stack((u, v), dim=-1)
        return sample

    @staticmethod
    @abstractmethod
    def params_in_MI_order(pair_params: torch.Tensor):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}(theta={self.theta})"


import scipy.stats as stats
from torch.autograd import Function


class StudentTICDF(Function):
    @staticmethod
    def forward(ctx, u, df):
        u_np = u.detach().cpu().numpy()
        df_np = df.detach().cpu().numpy()
        x_np = stats.t.ppf(u_np, df_np)
        x = torch.from_numpy(x_np).to(device=u.device, dtype=u.dtype)
        ctx.save_for_backward(x, df, u)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, df, u = ctx.saved_tensors
        dist = StudentT(df=df, loc=0.0, scale=1.0)
        pdf = dist.log_prob(x).exp()
        grad_u = grad_output / (pdf + 1e-8)
        grad_df = None
        return grad_u, grad_df


class StudentTCDF(Function):
    @staticmethod
    def forward(ctx, x, df):
        x_np = x.detach().cpu().numpy()
        df_np = df.detach().cpu().numpy()
        cdf_np = stats.t.cdf(x_np, df_np)
        cdf = torch.from_numpy(cdf_np).to(device=x.device, dtype=x.dtype)
        ctx.save_for_backward(x, df)
        return cdf

    @staticmethod
    def backward(ctx, grad_output):
        x, df = ctx.saved_tensors
        dist = StudentT(df=df, loc=0.0, scale=1.0)
        pdf = dist.log_prob(x).exp()
        grad_x = grad_output * pdf
        grad_df = None
        return grad_x, grad_df


class MyStudentT(StudentT):
    def icdf(self, value) -> torch.Tensor:
        value = value.clamp(1e-6, 1 - 1e-6)
        return StudentTICDF.apply(value, self.df)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        return StudentTCDF.apply(value, self.df)
