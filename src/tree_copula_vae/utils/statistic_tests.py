import numpy as np
from tqdm import tqdm
from tree_copula_vae.utils.distributions_distance_estimators import StatisticEstimator


def _calculate_statistic(x1, x2, statistic, permutation_id):
    return statistic(x1, x2), permutation_id


def permutation_test(
        X: np.ndarray,
        Y: np.ndarray,
        statistic: StatisticEstimator,
        n_permutations: int = 1000,
        verbose: int = 1
):
    m = X.shape[0]
    n = Y.shape[0]
    permutations = []
    combined_data = np.concatenate([X, Y], axis=0)
    observed_statistic = statistic(X, Y)
    for i in tqdm(range(n_permutations)):
        np.random.shuffle(combined_data)
        X_permute = combined_data[:m]
        Y_permute = combined_data[m:]
        mmd = statistic(X_permute, Y_permute)
        permutations.append(mmd)
    permutations = np.array(permutations)
    p_value = (permutations > observed_statistic).mean()
    return p_value
