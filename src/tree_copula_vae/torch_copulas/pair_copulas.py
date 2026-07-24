from typing import Dict
import math


import torch
from torch.distributions import constraints, Normal, Uniform
from tree_copula_vae.torch_copulas.base import PairCopula, MyStudentT
from tree_copula_vae.utils.numerical_stability import safe_icdf, safe_cdf
import torch.distributions.constraints as constraints

# פונקציות עזר לחישובים מתמטיים
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
        t = Uniform(
            low=torch.tensor(self._precision, device=self.u.device),
            high=torch.tensor(1. - self._precision, device=self.u.device)
        ).rsample(sample_shape=sample_shape).type(torch.get_default_dtype())

        return self.pair_copula.inverse_conditional_distribution(u=self.u, t=t)

    @staticmethod
    def params_in_MI_order(pair_params: torch.Tensor):
        pass
        # return pair_copula.params_in_MI_order(pair_params=pair_params)

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

    def inverse_conditional_distribution(self, u: torch.Tensor,
                                         t: torch.Tensor) -> torch.Tensor:
        # First make sure we don't have extreme values (0, 1) in the input
        u = u.clamp(self._precision, 1 - self._precision).to(self.theta.device)
        t = t.clamp(self._precision, 1 - self._precision).to(self.theta.device)

        a = torch.sqrt(1 - self.theta ** 2 + self._precision)
        b = self.theta
        norm = Normal(
            loc=torch.tensor(0., device=b.device),
            scale=torch.tensor(1., device=b.device)
        ).expand(self.batch_shape)
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
        super(BiVariateGaussianCopula, new).__init__(theta=theta,
                                                     validate_args=False)
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

        standard_normal = Normal(
            loc=torch.tensor(0., device=device),
            scale=torch.tensor(1., device=device)
        ).expand(batch_shape=self.batch_shape)
        theta_square = self.theta ** 2

        inverse_u = safe_icdf(dist=standard_normal, value=value[..., 0]).clamp(-10, 10)
        inverse_v = safe_icdf(dist=standard_normal, value=value[..., 1]).clamp(-10, 10)
        # This goes into ln and as a denominator hence we add some precision factor for numerical stability
        one_minus_theta_square = 1 - theta_square + self._precision

        a = 0.5 * torch.log(one_minus_theta_square)
        n1 = theta_square * inverse_u ** 2
        n2 = theta_square * inverse_v ** 2
        n3 = 2 * self.theta * inverse_v * inverse_u
        d = 2 * one_minus_theta_square

        log_prob = -a - ((n1 + n2 - n3) / d)  # TODO: might cause instabilities

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

    # Student-t needs DF > 0. usually DF > 2 for variance, but > 0 is enough for density
    min_df_val = 2.0 + eps

    arg_constraints = {
        'theta': constraints.interval(min_theta_val, max_theta_val),
        'df': constraints.greater_than(min_df_val)
    }

    def __init__(self, theta, df=3., validate_args=None, precision=1e-6):
        if not torch.is_tensor(df):
            df = torch.tensor(float(df))

        self.df = df.to(device=theta.device, dtype=theta.dtype)
        super(BiVariateStudentTCopula, self).__init__(theta=theta, validate_args=validate_args, precision=precision)

    @classmethod
    def get_class_with_fixed_df(cls, fixed_df_value):
        """
        Factory method: מחזירה מחלקה חדשה (יורשת) שבה ה-df מקובע.
        זה שומר על אנקפסולציה - העולם החיצון לא יודע איך יוצרים את המחלקה.
        """

        # זו הפונקציה שתשמש כ-__init__ למחלקה החדשה
        def init_fixed(instance, theta, validate_args=None):
            # instance הוא ה-self
            # אנחנו קוראים ל-init המקורי (cls.__init__) ומעבירים לו את ה-df הקבוע
            # הלוגיקה של המרה לטנזור/device כבר קורית ב-init המקורי למעלה!
            cls.__init__(instance, theta=theta, df=fixed_df_value, validate_args=validate_args)

        # יצירת המחלקה הדינמית
        # אנחנו יורשים מ-cls (שהוא BiVariateStudentTCopula)
        DynamicClass = type(
            f'{cls.__name__}_FixedDF_{fixed_df_value}',
            (cls,),
            {'__init__': init_fixed}
        )

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
        # For sorting edges in MST, theta is still the dominant factor for correlation strength
        # df affects tail dependence but less so the global magnitude of MI
        return pair_params.abs()

    def _get_inv_t_values(self, value):
        """Helper to transform u,v -> x,y using Inverse CDF of Student-T"""
        # Note: creating distribution objects in loop might be slow,
        # but StudentT icdf implementation is complex.
        # Assuming safe_icdf wrapper exists as in your code.

        # We use independent StudentT for the marginals
        marginal_dist = MyStudentT(df=self.df, loc=0., scale=1.)

        # value is [..., 2] (u and v)
        u = value[..., 0]
        v = value[..., 1]

        # Transform to domain of t-distribution
        x = safe_icdf(marginal_dist, u).clamp(-50, 50)  # Clamp for stability
        y = safe_icdf(marginal_dist, v).clamp(-50, 50)

        return x, y

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        value = self._change_value_to_be_numerical_stable(value)

        # 1. Transform u,v to x,y (quantiles)
        x, y = self._get_inv_t_values(value)

        rho = self.theta
        df = self.df

        # 2. Calculate Log Density using the Copula Density Formula:
        # log c(u,v) = log_prob_multivariate_t(x,y) - log_prob_univ_t(x) - log_prob_univ_t(y)

        # --- Multivariate Part ---
        # Mahalanobis-like term
        # z = (x^2 + y^2 - 2*rho*x*y) / (df * (1-rho^2))
        rho_sq = rho ** 2
        one_minus_rho_sq = 1 - rho_sq + self._precision

        numerator = x ** 2 + y ** 2 - 2 * rho * x * y
        denominator = df * one_minus_rho_sq
        z = numerator / denominator

        log_det = torch.log(one_minus_rho_sq)

        # Log constant for multivariate t (d=2)
        # C_mv = lgamma((df+2)/2) - lgamma(df/2) - log(df*pi) - 0.5*log_det
        log_C_mv = safe_lgamma((df + 2) / 2) - safe_lgamma(df / 2) - torch.log(df * math.pi) - 0.5 * log_det
        log_prob_mv = log_C_mv - ((df + 2) / 2) * torch.log(1 + z)

        # --- Univariate Part (Marginals) ---
        # log p(x) = lgamma((df+1)/2) - lgamma(df/2) - 0.5*log(df*pi) - (df+1)/2 * log(1 + x^2/df)
        log_C_uni = safe_lgamma((df + 1) / 2) - safe_lgamma(df / 2) - 0.5 * torch.log(df * math.pi)

        log_prob_x = log_C_uni - ((df + 1) / 2) * torch.log(1 + (x ** 2) / df)
        log_prob_y = log_C_uni - ((df + 1) / 2) * torch.log(1 + (y ** 2) / df)

        # --- Combine ---
        log_copula = log_prob_mv - log_prob_x - log_prob_y

        return log_copula

    def inverse_conditional_distribution(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Rosenblatt transform for Student-t.
        Computes v = C^{-1}(t | u).
        """
        # Clamping for safety
        u = u.clamp(self._precision, 1 - self._precision).to(self.theta.device)
        t = t.clamp(self._precision, 1 - self._precision).to(self.theta.device)

        rho = self.theta
        df = self.df

        # 1. Get x = F^{-1}(u) using univariate t distribution with nu dof
        marginal_dist = MyStudentT(df=df, loc=0., scale=1.)
        x = safe_icdf(marginal_dist, u).clamp(-50, 50)

        # 2. The conditional distribution of Y|X=x is a scaled t-distribution
        # with df_new = df + 1
        df_cond = df + 1.0

        # The conditional mean and scale
        # mu_cond = rho * x
        # scale_cond = sqrt( (df + x^2) / (df + 1) * (1 - rho^2) )
        scale_factor = torch.sqrt((df + x ** 2) / df_cond * (1 - rho ** 2 + self._precision))

        # 3. We need to solve: t = T_{df+1} ( (y - mu_cond) / scale_cond )
        # So: y = mu_cond + scale_cond * T^{-1}_{df+1}(t)

        cond_dist = MyStudentT(df=df_cond, loc=0., scale=1.)
        quantile_t = safe_icdf(cond_dist, t).clamp(-50, 50)  # T^{-1}_{df+1}(t)

        y = (rho * x) + (scale_factor * quantile_t)

        # 4. Convert back to v = F(y) using univariate t with nu dof
        v = safe_cdf(marginal_dist, y)

        return v

    def entropy(self):
        """
        Analytical Entropy of the Bivariate Student-t Copula.
        MI = -Entropy.
        """
        rho = self.theta
        nu = self.df
        d = 2.0  # Bivariate

        # 1. Log Determinant of Correlation Matrix (1 - rho^2)
        log_det = torch.log(1 - rho ** 2 + self._precision)

        # 2. Helper Gamma / Digamma terms
        lgamma_nu_2 = safe_lgamma(nu / 2)
        lgamma_nu_d_2 = safe_lgamma((nu + d) / 2)
        digamma_nu_2 = safe_digamma(nu / 2)
        digamma_nu_d_2 = safe_digamma((nu + d) / 2)

        # 3. Multivariate t Entropy (H_multi)
        # H_multi = 0.5*log_det + d/2*log(nu*pi) + lgamma(nu/2) - lgamma((nu+d)/2) + ...
        term1 = 0.5 * log_det
        term2 = (d / 2) * torch.log(nu * math.pi)
        term3 = lgamma_nu_2 - lgamma_nu_d_2
        term4 = ((nu + d) / 2) * (digamma_nu_d_2 - digamma_nu_2)

        H_multi = term1 + term2 + term3 + term4

        # 4. Univariate t Entropy (H_uni)
        # For d=1, log_det=0
        lgamma_nu_1_2 = safe_lgamma((nu + 1) / 2)
        digamma_nu_1_2 = safe_digamma((nu + 1) / 2)

        h_uni_term2 = 0.5 * torch.log(nu * math.pi)
        h_uni_term3 = lgamma_nu_2 - lgamma_nu_1_2
        h_uni_term4 = ((nu + 1) / 2) * (digamma_nu_1_2 - digamma_nu_2)

        H_uni = h_uni_term2 + h_uni_term3 + h_uni_term4

        # Copula Entropy = H_multi - sum(H_uni)
        # For pairs: H_multi - 2 * H_uni
        return H_multi - d * H_uni

    def spearman_corr(self) -> torch.Tensor:
        # For Student-t, Spearman's rho is the same as Gaussian
        return torch.arcsin(self.theta / 2) * 6 / math.pi


class FGMCopula(PairCopula):
    arg_constraints = {'theta': constraints.interval(-1, 1)}

    def inverse_conditional_distribution(self, u: torch.Tensor,
                                         t: torch.Tensor) -> torch.Tensor:
        a = self.theta * (2. * u - 1)
        zero_a_idx = a == 0
        # If a is zero then u is .5 and we need to return v = u
        # However, if a will stay zero then we can't do the calculations of b and c
        # We will give a some default value and make sure to return other one later
        a[zero_a_idx] = 1.

        b = 0.5 - (1 / (2 * a))
        c = torch.sqrt((b ** 2) + (t / a))

        v = torch.zeros_like(a)
        pos_a_idx = a > 0
        neg_a_idx = a < 0
        v[pos_a_idx] = b[pos_a_idx] + c[pos_a_idx]
        v[neg_a_idx] = b[neg_a_idx] - c[neg_a_idx]
        # Here we set the values for all a=0 from before
        v[zero_a_idx] = u[zero_a_idx]

        return v

    def params_in_MI_order(self):
        return self.theta.abs()

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(FGMCopula, _instance)
        theta = self.theta.expand(batch_shape)
        super(FGMCopula, new).__init__(theta=theta,
                                       validate_args=False)
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
        if not self.support.check(value):
            raise Exception(f"There are value not in support {self.support}")

        value = self._change_value_to_be_numerical_stable(value)
        u = value[..., 0]
        v = value[..., 1]
        a = 1 - 2 * u
        b = 1 - 2 * v
        log_prob = torch.log(1 + self.theta * a * b)

        return log_prob

    def cdf(self, value: torch.FloatTensor):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

class ClaytonCopula(PairCopula):
    min_theta_val = 1e-6
    # max_theta_val = 1 - 1e-6 # TODO: consider add maximum value to avoid instabilities
    arg_constraints = {'theta': constraints.greater_than(0.)}

    def inverse_conditional_distribution(self, u: torch.FloatTensor,
                                         t: torch.FloatTensor) -> torch.FloatTensor:
        # a = u ** (-self.theta)
        # b = t ** (-self.theta / (self.theta + 1))
        # v = (1 + a * (b - 1)) ** (-1 / self.theta)

        theta = self.theta

        # Numerical stable version (Genreates with ChatGPT)
        # Compute E = -theta * ln(u)
        E = -theta * torch.log(u)
        # Compute F = -[theta / (theta + 1)] * ln(t)
        theta_plus_one = theta + 1
        F = - (theta / theta_plus_one) * torch.log(t)
        # Compute ln(exp(F) - 1) in a numerically stable way
        # Handle small F values to avoid log(0) issues
        # ln(exp(F) - 1) = F + ln(1 - exp(-F))
        # When F is small, exp(-F) ~ 1, so we use torch.log1p for stability
        ln_expF_minus1 = F + torch.log1p(-torch.exp(-F))
        # Compute ln(exp(E) * (exp(F) - 1)) = E + ln_expF_minus1
        ln_product = E + ln_expF_minus1
        # Compute ln_S = ln(1 + exp(ln_product)) in a numerically stable way
        ln_S = torch.logaddexp(torch.tensor(0.0, dtype=torch.float64, device=u.device), ln_product)
        # Compute the exponent
        exponent = - (1 / theta) * ln_S
        # Compute the final result
        v = torch.exp(exponent)

        return v

    @staticmethod
    def params_in_MI_order(pair_params: torch.Tensor):
        return pair_params

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ClaytonCopula, _instance)
        theta = self.theta.expand(batch_shape)
        super(ClaytonCopula, new).__init__(theta=theta,
                                           validate_args=False)
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

        u = value[..., 0]
        v = value[..., 1]

        a = torch.log(self.theta + 1)
        b = (self.theta + 1) * torch.log(torch.mul(v, u))
        c = (2 * self. theta + 1) / self.theta
        d = torch.log(u ** (-self.theta) + v ** (-self.theta) - 1)

        log_prob = a -b - (c * d)

        return log_prob

    def cdf(self, value: torch.FloatTensor):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError
