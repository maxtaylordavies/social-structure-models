import numpy as np
from scipy.stats import mode
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.gridworld import Gridworld
from src.agent import Agent, Population
from src.modelling import evaluate_model
from src.utils import create_reward_functions, normalise
from scripts.discrete_choice import posterior


def generate_observations(config):
    O = np.zeros((config["M"], config["N"]))

    for n in range(config["N"]):
        R = config["R"][
            : config["K"], np.random.choice(config["Ltot"], size=config["L"], replace=False)
        ]
        population = Population(
            assignments=config["z"],
            group_rewards=R,
        )
        traj = population.generate_trajectories(
            world=config["world"],
            beta=config["beta"],
            start_pos=np.array([5, 5]),
            T=5,
        )  # shape (M, 5)
        O[:, n] = mode(traj, axis=1)[0].squeeze()

    return O


def visualise_q_values(config):
    reward_funcs = [np.array([10, 10, 10])] + [
        np.roll(np.array([100, 10, 1]), i) for i in range(3)
    ]
    Q_tables = []
    for r in reward_funcs:
        agent = Agent(r=r)
        _ = agent.get_action(world=config["world"], s=np.array([5, 5]), beta=1)
        Q_tables.append(agent.Q)
        agent.reset()

    # fig, axs = plt.subplots(2, 2)
    # actions = ["Left", "Up", "Right", "Down"]
    # for i in range(4):
    #     ax = axs[i // 2, i % 2]
    #     vals = agent.Q[:, i].reshape(6, 11)
    #     sns.heatmap(vals, cbar=False, xticklabels=False, yticklabels=False, ax=ax)
    #     ax.set_title(actions[i])
    # plt.show()

    fig, axs = plt.subplots(2, len(Q_tables))
    for i, Q in enumerate(Q_tables):
        V = normalise(Q.max(axis=1).reshape(6, 11))
        sns.heatmap(V, cbar=True, xticklabels=False, yticklabels=False, ax=axs[0, i])

        a_stars = Q.argmax(axis=1).reshape(6, 11)
        sns.heatmap(
            a_stars,
            cbar=True,
            # cmap=sns.color_palette("Set2", 4),
            cmap=sns.color_palette("ch:s=0.4,r=-1", 4),
            vmin=0,
            vmax=4,
            xticklabels=False,
            yticklabels=False,
            ax=axs[1, i],
        )

    colorbar = axs[1, -1].collections[0].colorbar
    colorbar.set_ticks([0.5, 1.5, 2.5, 3.5])
    colorbar.set_ticklabels(["Left", "Up", "Right", "Down"])

    fig.set_tight_layout(True)
    plt.show()


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
    config["R"] = create_reward_functions(config["Ltot"], config["ratio"])

    config["world"] = Gridworld(
        height=6,
        width=11,
        goals=[(5, 0), (0, 5), (5, 10)],
        move_cost=-1,
        gamma=0.9,
    )

    results = evaluate_model(
        config=config,
        gen_obs_func=generate_observations,
        posterior_func=posterior,
    )

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
