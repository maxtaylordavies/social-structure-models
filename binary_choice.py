from collections import Counter

import numpy as np
from scipy.stats import beta, bernoulli, binom, betabinom
from scipy.special import gamma
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import generate_all_partitions, error


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
    M = 10  # number of agents
    a, b, c = 0.1, 0.1, 1  #  parameters for beta prior and CRP
    z_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # ground truth group assignments
    theta_true = np.array([0.1, 0.9])  # ground truth latent group parameters

    print("generating partitions...")
    partitions = generate_all_partitions(M)
    print(f"generated {len(partitions)} partitions")

    N_vals, means, repeats = [1, 3, 5, 10, 15, 20], [], 3
    errors = np.zeros((repeats, len(N_vals)))

    for N in tqdm(N_vals):
        for i in range(repeats):
            probs = posterior(
                generate_observations(M, N, z_true, theta_true), partitions, a, b, c
            )
            _, e = error(partitions, probs, z_true)
            errors[i, N_vals.index(N)] = e

    plt.plot(N_vals, np.mean(errors, axis=0))
    plt.show()


if __name__ == "__main__":
    main()
