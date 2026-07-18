from typing import Dict
import math

import torch
from torch.distributions import constraints, Normal, Uniform
from tree_copula_vae.torch_copulas.base import PairCopula, MyStudentT
from tree_copula_vae.utils.numerical_stability import safe_icdf, safe_cdf
import torch.distributions.constraints as constraints


def safe_lgamma(x):
    return torch.lgamma(x)


def safe_digamma(x):
    return torch.digamma(x)


class ConditionalPairCopula(PairCopula):
    def __init__(self, pair_copula: PairCopula, u: torch.Tensor):
        self.pair_copula = pair_copula
        self.u = u
        super().__init__(theta=self.pair_copula.theta, validate_args=False)

    def entropy(self):
        self.pair_copula.entropy()

    def inverse_conditional_distribution(self, u: torch.FloatTensor, t: torch.Tensor) -> torch.Tensor:
        return super().inverse_conditional_distribution(u=u, t=t)

    def rsample(self, sample_shape=torch.Size(), device=None):
        sample_shape = self.u.size()
        t = Uniform(low=torch.tensor(self._precision, device=self.u.device), high=torch.tensor(1. - self._precision, device=self.u.device)).rsample(sample_shape=sample_shape).type(torch.get_default_dtype())
        return self.pair_copula.inverse_conditional_distribution(u=self.u, t=t)

    @staticmethod
    def params_in_MI_order(pair_params: torch.Tensor):
        pass

    def expand(self, batch_shape: torch.Size, _instance=None):
        new = self._get_checked_instance(ConditionalPairCopula, _instance)
        new.pair_copula = self.pair_copula.expand(batch_shape=batch_shape)
        new._pseudo_inputs = self.u.expand(batch_shape)
        return new

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return self.pair_copula.arg_constraints


class BiVariateGaussianCopula(PairCopula):
    eps = 1e-6
    min_theta_val = -1 + eps
    max_theta_val = 1 - eps
    arg_constraints = {'theta': constraints.interval(min_theta_val, max_theta_val)}

    def entropy(self):
        return 0.5 * torch.log(1 - self.theta ** 2)

    def inverse_conditional_distribution(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u = u.clamp(self._precision, 1 - self._precision).to(self.theta.device)
        t = t.clamp(self._precision, 1 - self._precision).to(self.theta.device)
        a = torch.sqrt(1 - self.theta ** 2 + self._precision)
        b = self.theta
        norm = Normal(loc=torch.tensor(0., device=b.device), scale=torch.tensor(1., device=b.device)).expand(self.batch_shape)
        i_cdf_t = safe_icdf(dist=norm, value=t).clamp(-10, 10)
        i_cdf_u = safe_icdf(dist=norm, value=u).clamp(-10, 10)
        v = safe_cdf(dist=norm, value=a * i_cdf_t + b * i_cdf_u)
        return v

    @staticmethod
    def params_in_MI_order(pair_params: torch.Tensor):
        return pair_params.abs()

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(BiVariateGaussianCopula, _instance)
        theta = self.theta.expand(batch_shape)
        super(BiVariateGaussianCopula, new).__init__(theta=theta, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def mode(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    def log_prob(self, value):
        if not self.support.check(value).all():
            raise Exception(f"There are value not in support {self.support}")
        value = self._change_value_to_be_numerical_stable(value)
        device = value.device
        standard_normal = Normal(loc=torch.tensor(0., device=device), scale=torch.tensor(1., device=device)).expand(batch_shape=self.batch_shape)
        theta_square = self.theta ** 2
        inverse_u = safe_icdf(dist=standard_normal, value=value[..., 0]).clamp(-10, 10)
        inverse_v = safe_icdf(dist=standard_normal, value=value[..., 1]).clamp(-10, 10)
        one_minus_theta_square = 1 - theta_square + self._precision
        a = 0.5 * torch.log(one_minus_theta_square)
        n1 = theta_square * inverse_u ** 2
        n2 = theta_square * inverse_v ** 2
        n3 = 2 * self.theta * inverse_v * inverse_u
        d = 2 * one_minus_theta_square
        log_prob = -a - ((n1 + n2 - n3) / d)
        return log_prob

    def cdf(self, value: torch.Tensor):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        raise NotImplementedError

    def spearman_corr(self) -> torch.Tensor:
        return torch.arcsin(self.theta / 2) * 6 / torch.pi


class BiVariateStudentTCopula(PairCopula):
    eps = 1e-6
    min_theta_val = -1 + eps
    max_theta_val = 1 - eps
    min_df_val = 2.0 + eps
    arg_constraints = {'theta': constraints.interval(min_theta_val, max_theta_val), 'df': constraints.greater_than(min_df_val)}

    def __init__(self, theta, df=3., validate_args=None, precision=1e-6):
        if not torch.is_tensor(df):
            df = torch.tensor(float(df))
        self.df = df.to(device=theta.device, dtype=theta.dtype)
        super(BiVariateStudentTCopula, self).__init__(theta=theta, validate_args=validate_args, precision=precision)

    @classmethod
    def get_class_with_fixed_df(cls, fixed_df_value):
        def init_fixed(instance, theta, validate_args=None):
            cls.__init__(instance, theta=theta, df=fixed_df_value, validate_args=validate_args)

        DynamicClass = type(f'{cls.__name__}_FixedDF_{fixed_df_value}', (cls,), {'__init__': init_fixed})
        return DynamicClass

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(BiVariateStudentTCopula, _instance)
        theta = self.theta.expand(batch_shape)
        df = self.df.expand(batch_shape)
        super(BiVariateStudentTCopula, new).__init__(theta=theta, validate_args=False)
        new.df = df
        new._validate_args = self._validate_args
        return new

    @staticmethod
    def params_in_MI_order(pair_params: torch.Tensor):
        return pair_params.abs()

    def _get_inv_t_values(self, value):
        marginal_dist = MyStudentT(df=self.df, loc=0., scale=1.)
        u = value[..., 0]
        v = value[..., 1]
        x = safe_icdf(marginal_dist, u).clamp(-50, 50)
        y = safe_icdf(marginal_dist, v).clamp(-50, 50)
        return x, y

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = self._change_value_to_be_numerical_stable(value)
        x, y = self._get_inv_t_values(value)
        rho = self.theta
        df = self.df
        rho_sq = rho ** 2
        one_minus_rho_sq = 1 - rho_sq + self._precision
        numerator = x ** 2 + y ** 2 - 2 * rho * x * y
        denominator = df * one_minus_rho_sq
        z = numerator / denominator
        log_det = torch.log(one_minus_rho_sq)
        log_C_mv = safe_lgamma((df + 2) / 2) - safe_lgamma(df / 2) - torch.log(df * math.pi) - 0.5 * log_det
        log_prob_mv = log_C_mv - ((df + 2) / 2) * torch.log(1 + z)
        log_C_uni = safe_lgamma((df + 1) / 2) - safe_lgamma(df / 2) - 0.5 * torch.log(df * math.pi)
        log_prob_x = log_C_uni - ((df + 1) / 2) * torch.log(1 + (x ** 2) / df)
        log_prob_y = log_C_uni - ((df + 1) / 2) * torch.log(1 + (y ** 2) / df)
        log_copula = log_prob_mv - log_prob_x - log_prob_y
        return log_copula

    def inverse_conditional_distribution(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u = u.clamp(self._precision, 1 - self._precision).to(self.theta.device)
        t = t.clamp(self._precision, 1 - self._precision).to(self.theta.device)
        rho = self.theta
        df = self.df
        marginal_dist = MyStudentT(df=df, loc=0., scale=1.)
        x = safe_icdf(marginal_dist, u).clamp(-50, 50)
        df_cond = df + 1.0
        scale_factor = torch.sqrt((df + x ** 2) / df_cond * (1 - rho ** 2 + self._precision))
        cond_dist = MyStudentT(df=df_cond, loc=0., scale=1.)
        quantile_t = safe_icdf(cond_dist, t).clamp(-50, 50)
        y = (rho * x) + (scale_factor * quantile_t)
        v = safe_cdf(marginal_dist, y)
        return v

    def entropy(self):
        rho = self.theta
        nu = self.df
        d = 2.0
        return 0.0
