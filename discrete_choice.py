from collections import Counter

# import jax.numpy as np
# from jax import random
import numpy as np
from scipy.special import gamma
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from utils import (
    boltzmann,
    random_partition,
    generate_all_partitions,
    posterior_mean,
    map_estimate,
    error,
)

sns.set_theme()

# key = random.PRNGKey(0)


def generate_observations(config, N):
    theta = boltzmann(config["r"], config["beta"])

    cum_p = np.tile(np.cumsum(theta[config["z"]], axis=-1), (N, 1, 1)).transpose([1, 0, 2])
    return np.argmax(np.random.uniform(size=(config["M"], N, 1)) < cum_p, axis=-1)


def prior(z, c):
    counts = Counter(z)
    prod = np.prod([gamma(counts[k]) for k in counts])
    coeff = ((c ** len(counts)) * gamma(c)) / gamma(c + len(z))
    return coeff * prod


def dirichlet_multinomial(counts, T, L, g, log=True):
    coeff = (gamma(L * g) * gamma(T + 1)) / gamma(T + (L * g))
    prods = np.prod(gamma(counts + g) / (gamma(g) * gamma(counts + 1)), axis=1)

    tmp = coeff * prods
    return np.sum(np.log(tmp)) if log else np.prod(tmp)


def likelihood(O, z, L, g, log=True):
    # likelihood of each group is given by multinomial-dirichlet distribution
    def group_likelihood(k, T_k):
        return dirichlet_multinomial(
            np.array([np.sum(O[z == k] == c, axis=0) for c in range(L)]), T_k, L, g, log=log
        )

    # likelihood of partition is product of likelihoods of each group
    totals = Counter(z)
    likelihoods = [group_likelihood(k, totals[k]) for k in totals]

    return -np.sum(likelihoods) if log else np.prod(likelihoods)


def posterior(O, partitions, config):
    likelihoods = np.array(
        [likelihood(O, z, config["L"], config["g"], log=config["log"]) for z in partitions]
    )
    z_priors = np.array([prior(z, config["c"]) for z in partitions])

    post = (
        likelihoods + np.log(z_priors)
        if config["log"]
        else np.multiply(likelihoods, z_priors)
    )

    if not config["log"]:
        return post / np.sum(post)

    post = np.exp(post)
    return post / np.sum(post)


def evaluate_discrete_choice_model(config):
    results = {"beta": [], "N": [], "mean": [], "map": [], "err_mean": [], "err_map": []}

    if config["sample"]:
        partitions = [random_partition(3, config["M"]) for _ in range(1000)]
        partitions = np.array(partitions)
    else:
        partitions = generate_all_partitions(config["M"])

    config["r"] = np.eye(config["L"])
    N_vals, beta_vals = config["N"], config["betas"]

    for _ in tqdm(range(config["repeats"])):
        for beta in beta_vals:
            config["beta"] = beta
            config["z"] = random_partition(3, config["M"])

            if config["sample"]:
                partitions[-1] = config["z"]

            O = generate_observations(config, N_vals[-1])

            for N in N_vals:
                post = posterior(O[:, :N], partitions, config)
                mean, _map = posterior_mean(partitions, post), map_estimate(partitions, post)

                results["beta"].append(beta)
                results["N"].append(N)
                results["mean"].append(mean)
                results["map"].append(_map)
                results["err_mean"].append(error(mean, config["z"]))
                results["err_map"].append(error(_map, config["z"]))

    return pd.DataFrame(results)


def main():
    config = {
        "M": 10,  # number of agents
        "N": [0, 1, 2, 3, 5, 10, 20, 50],  # number of observations per agent
        "L": 3,  # choice size,
        "betas": [0.01, 0.1, 0.2, 0.3, 0.5, 1.0],  # boltzmann temperature
        "g": 2,  # parameter for dirichlet prior over theta
        "c": 1,  # concentration parameter for CRP prior
        "sample": False,  # whether to randomly sample the set of partitions used for evaluation
        "log": True,  # whether to use logs for intermediate probability computations
        "repeats": 5,  # how many different true partitions to evaluate the model over
    }

    results = evaluate_discrete_choice_model(config)

    ax = sns.lineplot(
        data=results,
        x="N",
        y="err_mean",
        hue="beta",
        palette=sns.color_palette("viridis", len(config["betas"])),
    )
    ax.set(
        xlabel="Length of observation history",
        ylabel="Mean squared error",
        title="Performance of Dirichlet-Multinomial model",
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    for fmt in ["png", "svg", "pdf"]:
        plt.savefig(f"discrete_choice.{fmt}")
    plt.show()


if __name__ == "__main__":
    main()
