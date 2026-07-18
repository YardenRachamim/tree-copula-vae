import unittest
import torch
import numpy as np
from statsmodels.distributions.copula.api import GaussianCopula
from torch.distributions import Uniform
import itertools
from tqdm import tqdm

from tree_copula_vae.torch_copulas.graphical_structures import CopulaTreeStructure
from tree_copula_vae.torch_copulas.multivariate_copulas import MultivariateGaussianCopula, TreeCopula, MultivariateStudentTCopula
from tree_copula_vae.torch_copulas.pair_copulas import BiVariateGaussianCopula
from tree_copula_vae.utils.distributions_distance_estimators import MMDEstimator
from tree_copula_vae.utils.statistic_tests import permutation_test
from mixedvines.copula import GaussianCopula
from statsmodels.distributions.copula.api import StudentTCopula
from mixedvines.mixedvine import MixedVine
from scipy.stats import uniform


class TestTorchCopulas(unittest.TestCase):

    def setUp(self) -> None:
        self.eps = 1e-3
        self.standard_uniform_dist = Uniform(self.eps, torch.tensor(1.) - self.eps)
        self.n_samples = 1000
        self.pseudo_samples = self.standard_uniform_dist.sample(torch.Size([self.n_samples, 2])).clip(self.eps, 1 - self.eps)
        self.correlations_array = np.linspace(-0.98, 0.98)
        self.n_permutations = 1000
        self.alpha = 0.05
        self.bandwidth_ranges = [10, 15, 20, 50]

    def test_bivariate_gaussian_copula_log_prob(self):
        for correlation in self.correlations_array:
            actual_copula = BiVariateGaussianCopula(theta=torch.tensor(correlation))
            expected_copula = GaussianCopula(correlation)
            actual_log_prob = actual_copula.log_prob(self.pseudo_samples)
            expected_log_prob = expected_copula.logpdf(self.pseudo_samples.numpy())
            assert np.isclose(actual_log_prob, expected_log_prob, rtol=1e-3, atol=1e-3).all(), expected_log_prob.min()

    def test_bivariate_gaussian_copula_sampling(self):
        correlation = np.random.choice(self.correlations_array)
        actual_copula = BiVariateGaussianCopula(theta=torch.tensor(correlation))
        expected_copula = GaussianCopula(correlation)
        actual_samples = actual_copula.sample(torch.Size([self.n_samples])).numpy()
        expected_samples = expected_copula.rvs(self.n_samples)
        statistic = MMDEstimator(bandwidth_ranges=self.bandwidth_ranges)
        p_value = permutation_test(X=expected_samples, Y=actual_samples, statistic=statistic)
        error_msg = f"For correlation={correlation} found p_value={p_value} for n_permutations={self.n_permutations}"
        assert p_value > self.alpha, error_msg

    def test_multivariate_gaussian_copula_log_prob(self):
        for correlation in self.correlations_array:
            correlation_matrix = torch.tensor([[1, correlation], [correlation, 1]])
            actual_copula = MultivariateGaussianCopula(correlation_matrix)
            expected_copula = GaussianCopula(correlation)
            actual_log_prob = actual_copula.log_prob(self.pseudo_samples)
            expected_log_prob = expected_copula.logpdf(self.pseudo_samples.numpy())
            assert np.isclose(actual_log_prob, expected_log_prob, rtol=1e-3, atol=1e-3).all()

    def test_multivariate_studentT_copula_log_prob(self):
        df = 4
        for correlation in self.correlations_array:
            correlation_matrix = torch.tensor([[1, correlation], [correlation, 1]])
            actual_copula = MultivariateStudentTCopula(correlation_matrix, df=df)
            expected_copula = StudentTCopula(corr=correlation, df=df)
            actual_log_prob = actual_copula.log_prob(self.pseudo_samples)
            expected_log_prob = expected_copula.logpdf(self.pseudo_samples.numpy())
            assert np.isclose(actual_log_prob, expected_log_prob, rtol=1e-3, atol=1e-3).all()

    def test_multivariate_studentT_copula_sampling(self):
        df = 4
        correlation = np.random.choice(self.correlations_array)
        correlation_matrix = torch.tensor([[1, correlation], [correlation, 1]])
        actual_copula = MultivariateStudentTCopula(correlation_matrix, df=df)
        expected_copula = StudentTCopula(corr=correlation, df=df)
        expected_samples = expected_copula.rvs(self.n_samples)
        actual_samples = actual_copula.sample(torch.Size([self.n_samples])).view(*expected_samples.shape).numpy()
        statistic = MMDEstimator(bandwidth_ranges=self.bandwidth_ranges)
        p_value = permutation_test(X=expected_samples, Y=actual_samples, statistic=statistic)
        error_msg = f"For correlation={correlation} found p_value={p_value} for n_permutations={self.n_permutations}"
        assert p_value > self.alpha, error_msg

    def test_multivariate_gaussian_copula_sampling(self):
        correlation = np.random.choice(self.correlations_array)
        correlation_matrix = torch.tensor([[1, correlation], [correlation, 1]])
        actual_copula = MultivariateGaussianCopula(correlation_matrix)
        expected_copula = GaussianCopula(correlation)
        expected_samples = expected_copula.rvs(self.n_samples)
        actual_samples = actual_copula.sample(torch.Size([self.n_samples])).view(*expected_samples.shape).numpy()
        statistic = MMDEstimator(bandwidth_ranges=self.bandwidth_ranges)
        p_value = permutation_test(X=expected_samples, Y=actual_samples, statistic=statistic)
        error_msg = f"For correlation={correlation} found p_value={p_value} for n_permutations={self.n_permutations}"
        assert p_value > self.alpha, error_msg

    def test_tree_copula_log_prob_statsmodels(self):
        for c01, c02 in tqdm(np.array(list(itertools.combinations(self.correlations_array, 2)))):
            correlations = (c01, c02, 0.)
            pairs = [GaussianCopula(c01), GaussianCopula(c02), GaussianCopula(0.)]
            pseudo_samples = self.standard_uniform_dist.sample(torch.Size([self.n_samples, 3])).clip(self.eps, 1 - self.eps)
            pseudo_samples_numpy = pseudo_samples.numpy()
            T = CopulaTreeStructure.from_copula_pairs_param(copula_pair_params=torch.Tensor(correlations))
            tree_copula = TreeCopula(T)
            expected_tree_log_prob = pairs[0].logpdf(pseudo_samples_numpy[:, [0, 1]]) + pairs[1].logpdf(pseudo_samples_numpy[:, [0, 2]]) + pairs[2].logpdf(pseudo_samples_numpy[:, [1, 2]])
            actual_log_prob = tree_copula.log_prob(pseudo_samples)
            diff = expected_tree_log_prob - actual_log_prob.numpy()
            is_close = np.isclose(actual_log_prob.numpy(), expected_tree_log_prob, rtol=1e-3, atol=1e-3)
            error_msg = f"Correlations: {correlations}. Diff is: {diff[~is_close]}"
            assert is_close.all(), expected_tree_log_prob.min()
            assert is_close.all(), error_msg

    def test_tree_copula_log_prob_mixedvines(self):
        dim = 3
        vine = MixedVine(dim)
        vine.set_marginal(0, uniform(0, 1))
        vine.set_marginal(1, uniform(0, 1))
        vine.set_marginal(2, uniform(0, 1))
        vine.set_copula(2, 0, GaussianCopula(0.))
        for c01, c02 in tqdm(np.array(list(itertools.combinations(self.correlations_array, 2)))):
            correlations = (c01, c02, 0.)
            pseudo_samples = self.standard_uniform_dist.sample(torch.Size([self.n_samples, 3])).clip(self.eps, 1 - self.eps)
            pseudo_samples_numpy = pseudo_samples.numpy()
            T = CopulaTreeStructure.from_copula_pairs_param(copula_pair_params=torch.Tensor(correlations))
            tree_copula = TreeCopula(T)
            vine.set_copula(1, 0, GaussianCopula(c01))
            vine.set_copula(1, 1, GaussianCopula(c02))
            expected_tree_log_prob = vine.logpdf(pseudo_samples_numpy)
            actual_log_prob = tree_copula.log_prob(pseudo_samples)
            is_finite = torch.isfinite(torch.tensor(expected_tree_log_prob)).numpy()
            diff = expected_tree_log_prob[is_finite] - actual_log_prob.numpy()[is_finite]
            is_close = np.isclose(actual_log_prob.numpy()[is_finite], expected_tree_log_prob[is_finite], rtol=1e-3, atol=1e-3)
            error_msg = f"Correlations: {correlations}. Diff is: {diff[~is_close]}"
            assert is_close.all(), error_msg

    def test_tree_copula_log_prob_statsmodels_batch(self):
        for c01, c02 in tqdm(np.array(list(itertools.combinations(self.correlations_array, 2)))):
            pseudo_samples = self.standard_uniform_dist.sample(torch.Size([self.n_samples, 2, 3])).clip(self.eps, 1 - self.eps)
            pseudo_samples_numpy = pseudo_samples.numpy()
            correlations1 = (c01, c02, 0.)
            correlations2 = (c01, 0., c02)
            pairs1 = [GaussianCopula(c01), GaussianCopula(c02), GaussianCopula(0.)]
            pairs2 = [GaussianCopula(c01), GaussianCopula(0.), GaussianCopula(c02)]
            copula_pair_params = torch.stack([torch.Tensor(correlations1), torch.Tensor(correlations2)], 0)
            T = CopulaTreeStructure.from_copula_pairs_param(copula_pair_params=copula_pair_params)
            tree_copula = TreeCopula(T)
            expected_tree_log_prob1 = pairs1[0].logpdf(pseudo_samples_numpy[:, 0, :][:, [0, 1]]) + pairs1[1].logpdf(pseudo_samples_numpy[:, 0, :][:, [0, 2]]) + pairs1[2].logpdf(pseudo_samples_numpy[:, 0, :][:, [1, 2]])
            expected_tree_log_prob2 = pairs2[0].logpdf(pseudo_samples_numpy[:, 1, :][:, [0, 1]]) + pairs2[1].logpdf(pseudo_samples_numpy[:, 1, :][:, [0, 2]]) + pairs2[2].logpdf(pseudo_samples_numpy[:, 1, :][:, [1, 2]])
            actual_log_prob = tree_copula.log_prob(pseudo_samples)
            actual_log_prob1 = actual_log_prob[:, 0]
            actual_log_prob2 = actual_log_prob[:, 1]
            is_close = np.isclose(actual_log_prob1.numpy(), expected_tree_log_prob1, rtol=1e-3, atol=1e-3)
            is_close &= np.isclose(actual_log_prob2.numpy(), expected_tree_log_prob2, rtol=1e-3, atol=1e-3)
            assert is_close.all()

    def test_tree_copula_sampling(self):
        uniform_dist = Uniform(torch.tensor(-1.), torch.tensor(1.))
        c01 = uniform_dist.sample().clip(-1 + self.eps, 1 - self.eps).item()
        c02 = uniform_dist.sample().clip(-1 + self.eps, 1 - self.eps).item()
        correlations = (c01, c02, 0.)
        dim = 3
        vine = MixedVine(dim)
        vine.set_marginal(0, uniform(0, 1))
        vine.set_marginal(1, uniform(0, 1))
        vine.set_marginal(2, uniform(0, 1))
        vine.set_copula(2, 0, GaussianCopula(0.))
        vine.set_copula(1, 0, GaussianCopula(c01))
        vine.set_copula(1, 1, GaussianCopula(c02))
        T = CopulaTreeStructure.from_copula_pairs_param(copula_pair_params=torch.Tensor(correlations))
        tree_copula = TreeCopula(T)
        expected_samples = vine.rvs(self.n_samples)
        actual_samples = tree_copula.sample(torch.Size([self.n_samples])).view(*expected_samples.shape).numpy()
        statistic = MMDEstimator(bandwidth_ranges=self.bandwidth_ranges)
        p_value = permutation_test(
            X=expected_samples,
            Y=actual_samples, 
            statistic=statistic, 
            n_permutations=self.n_permutations
        )
        error_msg = f"For correlation={correlations} found p_value={p_value} for n_permutations={self.n_permutations}"
        assert p_value > self.alpha, error_msg
