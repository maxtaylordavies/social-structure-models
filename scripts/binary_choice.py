from collections import Counter
from itertools import product
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import betabinom
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.distributions import crp
from src.utils import generate_all_partitions, posterior_mean, error

sns.set_theme()


def generate_observations(M, N, z, theta):
    return (np.random.random(size=(M, N)) < theta[z].reshape((-1, 1))).astype(int)


def likelihood(O, z, a, b):
    totals = Counter(z)
    return np.prod(
        [np.prod(betabinom.pmf(np.sum(O[z == k], axis=0), totals[k], a, b)) for k in totals]
    )


def posterior(O, partitions, a, b, c):
    likelihoods = np.array([likelihood(O, z, a, b) for z in partitions])
    z_priors = np.array([crp(z, c) for z in partitions])
    probabilities = np.multiply(likelihoods, z_priors)
    return probabilities / np.sum(probabilities)


def evaluate_binary_choice_model():
    start = time.time()

    M = 10  # number of agents
    z_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # ground truth group assignments
    theta_true = np.array([0.1, 0.9])  # ground truth latent group parameters

    partitions = generate_all_partitions(M)
    print(f"generated {len(partitions)} partitions")

    a_vals, N_vals, c_vals = (
        [0.01, 0.1, 0.3, 0.5, 1.0],
        [1, 5, 10, 20, 50, 100],
        [0.1, 1.0, 10.0],
    )
    repeats, results = 5, pd.DataFrame({"alpha": [], "N": [], "c": [], "MSE": []})

    obs = [generate_observations(M, N_vals[-1], z_true, theta_true) for _ in range(repeats)]

    # evaluate model for different values of alpha, N, c
    param_grid = list(product(a_vals, N_vals, c_vals))
    print(f"evaluating model for {len(param_grid) * repeats} parameter combinations")
    for alpha, N, c in tqdm(param_grid):
        for rep in range(repeats):
            probs = posterior(obs[rep][:, :N], partitions, alpha, alpha, c)
            z = posterior_mean(partitions, probs)
            results = results.append(
                {"alpha": alpha, "N": N, "c": c, "MSE": error(z, z_true)},
                ignore_index=True,
            )

    print(f"total time taken: {time.time() - start} seconds")

    # save results to file
    results.to_pickle("binary_choice.pkl")

    # plot results
    pallete = sns.color_palette("cubehelix", len(a_vals))
    fig, axes = plt.subplots(1, len(c_vals), figsize=(12, 4), sharex=True, sharey=True)
    for i, c in enumerate(c_vals):
        ax = sns.lineplot(
            data=results[results["c"] == c],
            x="N",
            y="MSE",
            hue="alpha",
            errorbar="se",
            palette=pallete,
            ax=axes[i],
            legend=i == len(c_vals) - 1,
        )
        ax.set(
            xlabel="Length of observation history",
            ylabel="Mean squared error",
            title=f"c = {c}",
        )
        if i == len(c_vals) - 1:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    fig.suptitle("Performance of binary choice model")
    plt.savefig("binary_choice.png", bbox_inches="tight")


def main():
    evaluate_binary_choice_model()


if __name__ == "__main__":
    main()
