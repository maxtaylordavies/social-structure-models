import time

import numpy as np
import numpy.ma as ma
from scipy.stats import mode
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src.distributions import dirichlet_multinomial, crp
from src.gridworld import Gridworld
from src.agent import Population
from src.utils import (
    create_reward_functions,
    random_partition,
    posterior_mean,
    map_estimate,
    error,
    log,
)
from scripts.discrete_choice import likelihood as dc_likelihood


def generate_observations(config):
    R = config["R"][: config["K"], :3]
    population = Population(
        assignments=config["z"],
        group_rewards=R,
    )
    O = np.zeros((config["N"], config["M"], config["T_max"], 2))

    for n in range(config["N"]):
        O[n, :, :] = population.generate_trajectories(
            world=config["world"],
            beta=config["beta"],
            start_pos=np.array([5, 5]),
            max_T=config["T_max"],
        )

    return O


def make_buffer(O, z):
    buffer = {}
    N, M, T, _ = O.shape
    for n in range(N):
        for m in range(M):
            for t in range(T):
                s, a = O[n, m, t]

                if s == -1:
                    break

                if (s, z[m]) not in buffer:
                    buffer[(s, z[m])] = []
                buffer[(s, z[m])].append(a)

    return buffer


def likelihood_1(O, z, config):
    if np.sum(z) == 0:
        return -np.inf if config["log"] else 0

    actions = ma.masked_equal(O[:, :, :, 1], -1)

    # print(O.shape)
    # print(actions.shape)

    O_hat = mode(actions, axis=2)[0].squeeze(axis=2).T

    return dc_likelihood(O_hat, z, 4, config["g"], config["log"])


def likelihood_2(O, z, config):
    if np.sum(z) == 0:
        return -np.inf if config["log"] else 0

    buffer = make_buffer(O, z)

    def l(s, k):
        counts = np.array([buffer[(s, k)].count(a) for a in range(4)]).reshape((4, 1))
        return dirichlet_multinomial(
            counts, np.sum(counts), 4, config["g"], log=config["log"]
        )

    res = 0 if config["log"] else 1
    for (s, k) in buffer:
        tmp = l(s, k)
        res = res + tmp if config["log"] else res * tmp

    return res


def posterior(O, partitions, config, l_func, priors=None):
    if priors is None:
        priors = np.array([crp(z, config["c"]) for z in partitions])

    if len(O) == 0:
        return priors / np.sum(priors)

    likelihoods = np.array([l_func(O, z, config) for z in partitions])
    post = likelihoods + np.log(priors) if config["log"] else np.multiply(likelihoods, priors)

    if config["log"]:
        post = np.exp(post - np.max(post))

    return post / np.sum(post)


def compare_models(
    config,
    partitions,
    true_partitions,
):
    results = {"beta": [], "N": [], "err_mean": [], "err_map": [], "model": []}

    for rep in tqdm(range(config["repeats"])):
        config["z"] = true_partitions[rep]
        if config["sample"]:
            partitions[-1] = config["z"]

        for beta in config["betas"]:
            config["beta"] = beta

            log("generating observations...", with_tqdm=True)
            O = generate_observations(config)

            for model, l_func in zip(
                ["trajectory", "action"], [likelihood_1, likelihood_2]
            ):
                log(f"evaluating model {model}...", with_tqdm=True)
                priors = None
                for n in np.arange(0, config["N"] + 1, config["update_batch_size"]):
                    log(n, with_tqdm=True)
                    post = posterior(
                        O[max(n - config["update_batch_size"], 0) : n],
                        partitions,
                        config,
                        priors=priors,
                        l_func=l_func,
                    )
                    priors = post

                    mean = posterior_mean(partitions, post)
                    _map = map_estimate(partitions, post)

                    # tqdm.write(f"{model}: mean = {str(mean)}")
                    # tqdm.write(f"{model}: map = {str(_map)}")

                    results["beta"].append(beta)
                    results["N"].append(n)
                    results["err_mean"].append(error(mean, config["z"]))
                    results["err_map"].append(error(_map, config["z"]))
                    results["model"].append(model)

    return pd.DataFrame(results)


def main():
    config = {
        "M": 10,  # number of agents
        "N": 20,  # number of observations per agent
        "T_max": 8,  # maximum number of time steps per trajectory
        "K": 2,  # number of true groups
        "Ltot": 4,  # total number of options available
        "L": 3,  # number of options available per trial
        # "betas": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 10],  # boltzmann temperature
        "betas": [0.01, 0.05, 0.1, 0.5, 1],
        "ratio": 10,  # choice preference ratio
        "g": 2,  # parameter for dirichlet prior over theta
        "c": 1,  # concentration parameter for CRP prior
        "sample": True,  # whether to randomly sample the set of partitions used for evaluation
        "log": True,  # whether to use logs for intermediate probability computations
        "update_batch_size": 1,  # batch size (num observations) for updating the posterior
        "repeats": 25,  # how many different true partitions to evaluate the model over
    }

    config["R"] = create_reward_functions(config["Ltot"], config["ratio"])
    config["world"] = Gridworld(
        height=6,
        width=11,
        goals=[(5, 0), (0, 5), (5, 10)],
        move_cost=-1,
        gamma=0.9,
    )

    partitions_sample = [random_partition(config["K"], config["M"]) for _ in range(1000)]
    true_partitions = [
        random_partition(config["K"], config["M"]) for _ in range(config["repeats"])
    ]

    results = compare_models(config, partitions_sample, true_partitions)

    ax = sns.lineplot(
        data=results,
        x="N",
        y="err_mean",
        hue="beta",
        style="model",
        palette=sns.color_palette("viridis", len(config["betas"])),
    )
    ax.set(
        xlim=(0, config["N"]),
        xlabel="Trajectories observed per agent",
        ylabel="MSE (posterior mean)",
        title="Performance of Dirichlet-Multinomial model",
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()


if __name__ == "__main__":
    main()
