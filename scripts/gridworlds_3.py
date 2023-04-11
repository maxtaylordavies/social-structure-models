from collections import Counter

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.distributions import dirichlet_multinomial
from src.gridworld import Gridworld
from src.agent import Agent, Population
from src.modelling import evaluate_model
from src.utils import create_reward_functions, random_partition
from scripts.discrete_choice import posterior as _posterior
from scripts.gridworlds_2 import generate_observations as generate_observations_mode


def generate_observations(config):
    O = np.zeros((config["N"], config["M"], config["T"]))

    for n in range(config["N"]):
        R = config["R"][
            : config["K"], np.random.choice(config["Ltot"], size=config["L"], replace=False)
        ]
        population = Population(
            assignments=config["z"],
            group_rewards=R,
        )
        O[n, :, :] = population.generate_trajectories(
            world=config["world"],
            beta=config["beta"],
            start_pos=np.array([5, 5]),
            max_T=config["T"],
        )  # shape (M, T)

    return np.transpose(O, (1, 0, 2))  # shape (M, N, T)


def likelihood(O, z, L, g, log=True):
    # likelihood of each group is given by multinomial-dirichlet distribution
    def group_likelihood(k, T_k):
        group_obs = O.reshape(O.shape[0], -1)[z == k]
        counts = np.array([np.sum(group_obs == a, axis=0) for a in range(4)])
        return dirichlet_multinomial(counts, T_k, 4, g, log=log)

    # likelihood of partition is product of likelihoods of each group
    totals = Counter(z)
    likelihoods = [group_likelihood(k, totals[k]) for k in totals]

    return -np.sum(likelihoods) if log else np.prod(likelihoods)


def posterior(O, partitions, config, priors):
    return _posterior(O, partitions, config, priors, l_func=likelihood)


def main():
    config = {
        "M": 10,  # number of agents
        "N": 5,  # number of observations per agent
        "T": 10,  # number of time steps per trajectory
        "K": 2,  # number of true groups
        "Ltot": 4,  # total number of options available
        "L": 3,  # number of options available per trial
        # "betas": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 10],  # boltzmann temperature
        "betas": [0.01, 0.1, 0.5, 1],
        "ratio": 10,  # choice preference ratio
        "g": 2,  # parameter for dirichlet prior over theta
        "c": 1,  # concentration parameter for CRP prior
        "sample": True,  # whether to randomly sample the set of partitions used for evaluation
        "log": True,  # whether to use logs for intermediate probability computations
        "update_batch_size": 1,  # batch size (num observations) for updating the posterior
        "repeats": 1,  # how many different true partitions to evaluate the model over
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

    traj_results = evaluate_model(
        config=config,
        gen_obs_func=generate_observations_mode,
        posterior_func=_posterior,
        partitions=partitions_sample,
        true_partitions=true_partitions,
    )
    action_results = evaluate_model(
        config=config,
        gen_obs_func=generate_observations,
        posterior_func=posterior,
        partitions=partitions_sample,
        true_partitions=true_partitions,
    )

    # add extra columns to results
    traj_results["level"] = "trajectory"
    action_results["level"] = "action"

    # combine results and plot
    results = traj_results.append(action_results, ignore_index=True)
    ax = sns.lineplot(
        data=results,
        x="N",
        y="err_mean",
        hue="beta",
        style="level",
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
