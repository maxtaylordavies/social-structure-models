from collections import defaultdict
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
    avg_pairwise_distance,
    normalise,
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
    for k in np.unique(z):
        O_k = O[:, z == k].reshape(-1, 2)
        buffer.update({(s, k): list(O_k[O_k[:, 0] == s, 1]) for s in np.unique(O_k[:, 0])})
    return buffer


# computes the weighted average of the states visited by a group
def get_group_centroid(buffer, config, k):
    idxs = [[s] * len(buffer[s, k_]) for s, k_ in buffer if k_ == k]
    idxs = np.array([s for sl in idxs for s in sl])
    return config["world"].idx_to_state(idxs).mean(axis=1)


def group_distance(buffer, config):
    centroids = [get_group_centroid(buffer, config, k_) for k_ in range(config["K"])]
    return avg_pairwise_distance(centroids)


def likelihood_1(O, z, config):
    if np.sum(z) == 0:
        return -np.inf if config["log"] else 0

    O[O == -1] = np.nan
    O_hat = mode(O[:, :, :, 1], axis=2, nan_policy="omit")[0].squeeze(axis=2).T

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

    sim_term = 0 if config["log"] else 1
    if config["w1"] > 0:
        for (s, k) in buffer:
            tmp = l(s, k)
            sim_term = sim_term + tmp if config["log"] else sim_term * tmp

    diff_term = group_distance(buffer, config) if config["w2"] > 0 else 0

    return (config["w1"] * sim_term) + (config["w2"] * diff_term)


def posterior(O, partitions, config, l_func, priors=None):
    if priors is None:
        priors = np.array([crp(z, config["c"]) for z in partitions])
        priors = priors / np.sum(priors)

    if len(O) == 0:
        return priors

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

    def record(beta, n, probs, model):
        results["beta"].append(beta)
        results["N"].append(n)
        results["err_mean"].append(error(posterior_mean(partitions, probs), config["z"]))
        results["err_map"].append(error(map_estimate(partitions, probs), config["z"]))
        results["model"].append(model)

    for rep in tqdm(range(config["repeats"])):
        config["z"] = true_partitions[rep]
        if config["sample"]:
            partitions[-1] = config["z"]

        for beta in config["betas"]:
            config["beta"] = beta

            O = generate_observations(config)
            for model, l_func in zip(["action"], [likelihood_2]):
                probs = None
                for n in np.arange(0, config["N"] + 1, config["update_batch_size"]):
                    probs = posterior(
                        O[max(n - config["update_batch_size"], 0) : n],
                        partitions,
                        config,
                        l_func=l_func,
                        priors=probs,
                    )
                    record(beta, n, probs, model)

    return pd.DataFrame(results)


def save_results(results, filename):
    results.to_pickle(f"../data/{filename}.pkl")


def plot_results(results, config, show, filename):
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
    fig.set_tight_layout(True)

    # sns.lineplot(
    #     data=results[results["model"] == "trajectory"],
    #     x="N",
    #     y="err_mean",
    #     hue="beta",
    #     palette=sns.color_palette("viridis", len(config["betas"])),
    #     ax=axs[0],
    #     legend=False,
    # )
    # axs[0].set(
    #     xlim=(0, config["N"]),
    #     xlabel="Trajectories observed per agent",
    #     ylabel="MSE (posterior mean)",
    #     title="Trajectory-level",
    # )
    sns.lineplot(
        # data=results[results["model"] == "action"],
        data=results,
        x="N",
        y="err_mean",
        hue="beta",
        palette=sns.color_palette("viridis", len(config["betas"])),
        ax=axs[0],
        legend=False,
    )
    axs[0].set(
        xlim=(0, config["N"]),
        xlabel="Trajectories observed per agent",
        ylabel="MSE (posterior mean)",
        # title="Action-level",
    )

    sns.lineplot(
        # data=results[results["model"] == "action"],
        data=results,
        x="N",
        y="err_map",
        hue="beta",
        palette=sns.color_palette("viridis", len(config["betas"])),
        ax=axs[1],
    )
    axs[1].set(
        xlim=(0, config["N"]),
        xlabel="Trajectories observed per agent",
        ylabel="MSE (MAP estimate)",
        # title="Action-level",
    )

    sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Comparison of DirMult gridworld models")

    if show:
        plt.show()

    if filename:
        fig.savefig(f"../figures/{filename}.svg", bbox_inches="tight")


def main():
    config = {
        "M": 10,  # number of agents
        "N": 50,  # number of observations per agent
        "T_max": 10,  # maximum number of time steps per trajectory
        "K": 2,  # number of true groups
        "Ltot": 4,  # total number of options available
        "L": 3,  # number of options available per trial
        # "betas": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 10],  # boltzmann temperature
        "betas": [0.01, 0.025, 0.05, 0.1],
        "ratio": 10,  # choice preference ratio
        "g": 2,  # parameter for dirichlet prior over theta
        "c": 1,  # concentration parameter for CRP prior
        "w1": 1,  # weight for action similarity term in likelihood
        "w2": 0,  # weight for group state distance term in likelihood
        "sample": True,  # whether to randomly sample the set of partitions used for evaluation
        "log": True,  # whether to use logs for intermediate probability computations
        "update_batch_size": 5,  # batch size (num observations) for updating the posterior
        "repeats": 10,  # how many different true partitions to evaluate the model over
    }

    config["R"] = create_reward_functions(config["Ltot"], config["ratio"])
    config["world"] = Gridworld(
        height=6,
        width=11,
        goals=[(5, 0), (0, 5), (5, 10)],
        move_cost=-1,
        gamma=0.9,
    )

    # config["z"] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # config["beta"] = 0.07

    # O = generate_observations(config)

    # reps = 100
    # partitions = [random_partition(config["K"], config["M"]) for _ in range(reps)]
    # partitions[0] = config["z"]

    # probs = posterior(O, partitions, config, l_func=likelihood_2)

    # print("------------------------------------")
    # print(posterior_mean(partitions, probs))
    # print(map_estimate(partitions, probs))

    partitions_sample = [random_partition(config["K"], config["M"]) for _ in range(100)]
    true_partitions = [
        random_partition(config["K"], config["M"]) for _ in range(config["repeats"])
    ]

    results = compare_models(config, partitions_sample, true_partitions)
    plot_results(results, config, show=True, filename=None)

    # for ubs in [1, 2, 5, 10, 25]:
    #     config["update_batch_size"] = ubs
    #     results = compare_models(config, partitions_sample, true_partitions)
    #     save_results(results, f"gridworlds_4/{ubs}")
    #     plot_results(results, config, f"gridworlds_4/{ubs}")


if __name__ == "__main__":
    main()
