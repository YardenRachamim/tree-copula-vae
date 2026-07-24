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
        # Actual lower bound is 0. but it's numerical stable
        lower_bound = eps
        # Actual lower bound is 1. but it's numerical stable
        upper_bound = 1 - eps

        # To make sure that gradients still propagate we use clamp
        numerical_stable_value = value.clamp(lower_bound, upper_bound)

        return numerical_stable_value

    @property
    def support(self) -> Optional[Any]:
        return extended_constraints.unit_cube

    @property
    def has_rsample(self) -> bool:
        # In theory copula can always be resampled using 'Inverse Integral Trasnform'
        return True

    @abstractmethod
    def rsample(self, sample_shape: torch.Size = torch.Size(), device: Optional[torch.device] = None) -> torch.Tensor:
        # To be implemented by inherit object
        pass

    @property
    @abstractmethod
    def pair_params(self):
        # To be implemented by inherit object
        pass

    def sample(self, sample_shape: torch.Size = torch.Size()):
        # Can be not abstract since we assuming that has_rsample is always True
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape)

    def __repr__(self):
        # For string use and debugging
        return self.__class__.__name__


class PairCopula(MultivariateCopula, metaclass=ABCMeta):
    min_theta_val: float = float('-inf')
    max_theta_val: float = float('inf')

    def __init__(self, theta: Union[float, torch.Tensor] = None, validate_args=None, precision: float = 1e-6):
        """
        Abstract representation of the bi-variate copula.

        bi-variate copulas are the basic copulas just as univariate is for regular distributions. This is the most studied types of copulas,
        and it used as a building blocks to many other Multivariate copulas such as Vine and Bayesian Copula Networks Parameters

        ----------
        theta : The bi-variate copulas parameter, usually it's a single float but we enable cases for more complex copulas parameters using Tensor
        validate_args : Any distribution need validate_args - see pytorch documentation of torch.distributions.Distribution
        """
        # In the bi-variate case it's always 2
        self._event_shape = torch.Size([2])
        # TODO: why I do that?
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
        """


        Parameters
        ----------
        u : The instance to be conditioned on
        t : the instance that we will transform given u in order to get the actual instance (most of the time will be represented as v)

        Returns
        -------

        """
        # Used to r-sample from the copula
        pass

    def rsample(self, sample_shape=torch.Size(), device=None):
        if device is None:
            device = self.theta.device

        # TODO: why I use it?
        shape = self._extended_shape(sample_shape)
        
        # We can use here rsample, However I use sample to emphasize that this is the base distribution as much as Standard-Normal is in the
        # gaussian case
        uniform_sample = Uniform(
            low=torch.tensor(self._precision, device=device),
            high=torch.tensor(1. - self._precision, device=device)
        ).sample(sample_shape=shape)
        # u is the first instance of the distribution (recall that, event_shape==2)
        u = uniform_sample[..., 0]
        t = uniform_sample[..., 1]
        v = self.inverse_conditional_distribution(u, t)

        # The actual copula sample
        sample = torch.stack((u, v), dim=-1)

        # # Pass the sample to device, If None stays in the same device i.e CPU
        # # TODO: must be a better way
        # sample = sample.to(device)

        return sample

    @staticmethod
    @abstractmethod
    def params_in_MI_order(pair_params: torch.Tensor):
        raise NotImplementedError()

    def __repr__(self):
        # For string use and debugging
        return f"{self.__class__.__name__}(theta={self.theta})"


import torch
import scipy.stats as stats
from torch.autograd import Function
from torch.distributions import StudentT


class StudentTICDF(Function):
    @staticmethod
    def forward(ctx, u, df):
        """
        Forward pass: משתמשים ב-SciPy כדי לחשב את ה-ICDF (ppf) במדויק.
        זה רץ על ה-CPU, אבל זה מאוד מדויק ויציב.
        """
        # המרה ל-Numpy עבור SciPy
        u_np = u.detach().cpu().numpy()
        df_np = df.detach().cpu().numpy()

        # חישוב הערך (Percent Point Function)
        x_np = stats.t.ppf(u_np, df_np)

        # החזרה ל-Tensor
        x = torch.from_numpy(x_np).to(device=u.device, dtype=u.dtype)

        # שמירת משתנים ל-Backward
        ctx.save_for_backward(x, df, u)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: הנגזרת של ICDF היא פשוט 1 חלקי ה-PDF.
        """
        x, df, u = ctx.saved_tensors

        # הנגזרת לפי u:
        # d(ICDF(u))/du = 1 / PDF(ICDF(u))
        # אנחנו משתמשים במימוש ה-PDF של PyTorch שהוא יציב ומהיר
        dist = StudentT(df=df, loc=0.0, scale=1.0)
        pdf = dist.log_prob(x).exp()

        # הגנה מחלוקה באפס (בקצוות)
        grad_u = grad_output / (pdf + 1e-8)

        # הנגזרת לפי df:
        # זה מסובך מתמטית ולא נתמך ב-SciPy בקלות.
        # אם אנחנו לא לומדים את ה-df (הוא קבוע), אפשר להחזיר None.
        # אם חייבים ללמוד את df, זה המקום שבו השיטה הזו נופלת.
        grad_df = None

        return grad_u, grad_df


class StudentTCDF(Function):
    @staticmethod
    def forward(ctx, x, df):
        """
        Forward: חישוב מדויק באמצעות SciPy.
        """
        x_np = x.detach().cpu().numpy()
        df_np = df.detach().cpu().numpy()

        cdf_np = stats.t.cdf(x_np, df_np)

        cdf = torch.from_numpy(cdf_np).to(device=x.device, dtype=x.dtype)

        # שומרים ל-backward
        ctx.save_for_backward(x, df)
        return cdf

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: הנגזרת של CDF היא פשוט ה-PDF!
        d(CDF(x))/dx = PDF(x)
        """
        x, df = ctx.saved_tensors

        # משתמשים במימוש היעיל והיציב של PyTorch ל-PDF
        dist = StudentT(df=df, loc=0.0, scale=1.0)
        pdf = dist.log_prob(x).exp()

        # כלל השרשרת
        grad_x = grad_output * pdf

        # נגזרת לפי df: (מורכבת, נחזיר None כי ה-df קבוע)
        grad_df = None

        return grad_x, grad_df


class MyStudentT(StudentT):
    def icdf(self, value) -> torch.Tensor:
        value = value.clamp(1e-6, 1 - 1e-6)

        return StudentTICDF.apply(value, self.df)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        return StudentTCDF.apply(value, self.df)


import torch
from torch.distributions import Distribution, Uniform, constraints
from torch.distributions.utils import broadcast_all


class Logistic(Distribution):
    """
    Logistic distribution with location `loc` and scale `scale`.
    API is compatible with torch.distributions.Normal
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Logistic, self).__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterization trick:
        z = mu + s * log(u / (1-u))
        """
        shape = self._extended_shape(sample_shape)
        # 1. Sample Uniform(0, 1)
        u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)

        # 2. Numerical stability clamp (prevent log(0))
        eps = torch.finfo(u.dtype).eps
        u = u.clamp(min=eps, max=1 - eps)

        # 3. Logit function (Inverse CDF of Logistic is Logit)
        logit_u = torch.log(u) - torch.log1p(-u)

        # 4. Scale and Shift
        return self.loc + self.scale * logit_u

    def log_prob(self, value):
        """
        PDF: f(x) = (1/s) * exp(-(x-m)/s) / (1 + exp(-(x-m)/s))^2
        Log PDF computation requires care for stability.
        """
        if self._validate_args:
            self._validate_sample(value)

        # z = (x - mu) / s
        z = (value - self.loc) / self.scale

        # log(f(x)) = -log(s) - z - 2*log(1 + exp(-z))
        # We use softplus for stable log(1 + exp(-z)) -> softplus(-z)
        return -torch.log(self.scale) - z - 2. * F.softplus(-z)

    def cdf(self, value):
        """
        CDF: F(x) = 1 / (1 + exp(-(x-m)/s)) = Sigmoid((x-m)/s)
        """
        if self._validate_args:
            self._validate_sample(value)
        return torch.sigmoid((value - self.loc) / self.scale)

    def icdf(self, value):
        """
        Inverse CDF (Quantile function): mu + s * log(p / (1-p))
        """
        # Numerical stability
        eps = torch.finfo(value.dtype).eps
        value = value.clamp(min=eps, max=1 - eps)

        logit_v = torch.log(value) - torch.log1p(-value)
        return self.loc + self.scale * logit_v


# # Helper imports needed for the class above
# from torch.distributions import constraints
# from numbers import Number
#