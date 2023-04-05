from collections import Counter

import numpy as np
from scipy.special import gamma
import seaborn as sns
import matplotlib.pyplot as plt

from src.distributions import boltzmann2d, crp, dirichlet_multinomial
from src.modelling import evaluate_model
from src.utils import create_reward_functions

sns.set_theme()

# key = random.PRNGKey(0)


def generate_observations(config):
    r = create_reward_functions(config["Ltot"], config["ratio"])[config["z"]]

    trial_probs = []
    for _ in range(config["N"]):
        tmp = r[:, np.random.choice(config["Ltot"], config["L"], replace=False)]
        tmp = boltzmann2d(tmp / np.sum(tmp, axis=1, keepdims=True), config["beta"])
        trial_probs.append(np.cumsum(tmp, axis=-1))
    cum_p = np.stack(trial_probs).transpose([1, 0, 2])  # (M, N, L)

    return np.argmax(np.random.uniform(size=(config["M"], config["N"], 1)) < cum_p, axis=-1)


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


def posterior(O, partitions, config, priors=None):
    likelihoods = np.array(
        [likelihood(O, z, config["L"], config["g"], log=config["log"]) for z in partitions]
    )

    if priors is None:
        priors = np.array([crp(z, config["c"]) for z in partitions])

    post = likelihoods + np.log(priors) if config["log"] else np.multiply(likelihoods, priors)

    if not config["log"]:
        return post / np.sum(post)

    post = np.exp(post)
    return post / np.sum(post)


def main():
    config = {
        "M": 10,  # number of agents
        "N": 50,  # number of observations per agent
        "K": 2,  # number of true groups
        "Ltot": 4,  # total number of options available
        "L": 3,  # number of options available per trial
        "betas": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 10],  # boltzmann2d temperature
        "ratio": 10,  # choice preference ratio
        "g": 2,  # parameter for dirichlet prior over theta
        "c": 1,  # concentration parameter for CRP prior
        "sample": True,  # whether to randomly sample the set of partitions used for evaluation
        "log": True,  # whether to use logs for intermediate probability computations
        "update_batch_size": 5,  # batch size (num observations) for updating the posterior
        "repeats": 100,  # how many different true partitions to evaluate the model over
    }

    results = evaluate_model(config, generate_observations, posterior)

    ax = sns.lineplot(
        data=results,
        x="N",
        y="err_mean",
        hue="beta",
        palette=sns.color_palette("viridis", len(config["betas"])),
    )
    ax.set(
        xlim=(0, config["N"]),
        xlabel="Length of observation history",
        ylabel="MSE (posterior mean)",
        title="Performance of Dirichlet-Multinomial model",
    )

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # for fmt in ["png", "svg"]:
    #     plt.savefig(f"discrete_choice_3.{fmt}")
    plt.show()


if __name__ == "__main__":
    main()
