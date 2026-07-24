import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from tree_copula_vae.utils.distributions_distance_estimators import StatisticEstimator


def _calculate_statistic(x1, x2, statistic, permutation_id):
    # print(f"Start working at permutation: {permutation_id}")
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

    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     future_tasks = []
    #     for i in range(n_permutations):
    #         np.random.shuffle(combined_data)
    #         X_permute = combined_data[:m]
    #         Y_permute = combined_data[m:]
    #         future_task = executor.submit(_calculate_statistic, X_permute, Y_permute, statistic, i)
    #
    #         future_tasks.append(future_task)
    #
    #     i = 0
    #     for completed_task in as_completed(future_tasks):
    #         mmd, permutation_id = completed_task.result()
    #         # print(f"Finished working at permutation: {permutation_id}", flush=True)
    #         permutations.append(mmd)
    #         i += 1
    #         if i +1 % 10 == 0:
    #             print(f"Finished {i} permutation test")

    for i in tqdm(range(n_permutations)):
        if i + 1 % 10 == 0:
            print(f"Finished the {i+1}th permutation")

        np.random.shuffle(combined_data)

        X_permute = combined_data[:m]
        Y_permute = combined_data[m:]
        mmd = statistic(X_permute, Y_permute)
        permutations.append(mmd)
    permutations = np.array(permutations)
    p_value = (permutations > observed_statistic).mean()

    return p_value
