from collections import Counter
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, bernoulli, binom, betabinom
from scipy.special import gamma
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from utils import generate_all_partitions, posterior_mean, error


def generate_observations(M, N, z, theta):
    O = np.zeros((M, N))
    for i in range(M):
        O[i] = (np.random.random(size=(N)) < theta[z[i]]).astype(int)
    return O


def crp(z, c):
    counts = Counter(z)
    prod = np.prod([gamma(counts[k]) for k in counts])
    coeff = ((c ** len(counts)) * gamma(c)) / gamma(c + len(z))
    return coeff * prod


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


def main():
    start = time.time()

    M = 10  # number of agents
    a, b, c = 0.1, 0.1, 1  #  parameters for beta prior and CRP
    z_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # ground truth group assignments
    theta_true = np.array([0.1, 0.9])  # ground truth latent group parameters

    print("generating partitions...")
    partitions = generate_all_partitions(M)
    print(f"generated {len(partitions)} partitions")

    N_vals, means, errors, repeats = [1, 3, 5, 10, 15, 20], [], [], 5

    for i in tqdm(range(len(N_vals))):
        tmp = np.zeros((repeats, M))
        for j in range(repeats):
            probs = posterior(
                generate_observations(M, N_vals[i], z_true, theta_true), partitions, a, b, c
            )
            tmp[j] = posterior_mean(partitions, probs)
            errors.append(error(tmp[j], z_true))
        means.append(np.mean(tmp, axis=0))

    print(f"total time taken: {time.time() - start} seconds")

    sns.relplot(
        data=pd.DataFrame(
            {"N": N_vals, "error": np.array([error(z, z_true) for z in means])}
        ),
        x="N",
        y="error",
        kind="line",
    )

    plt.show()

    sns.relplot(
        data=pd.DataFrame({"N": np.repeat(N_vals, repeats), "error": errors}),
        x="N",
        y="error",
        kind="line",
    )

    plt.show()


if __name__ == "__main__":
    main()
