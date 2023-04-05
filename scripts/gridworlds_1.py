import numpy as np
from scipy.stats import mode
import seaborn as sns
import matplotlib.pyplot as plt

from src.distributions import boltzmann2d
from src.modelling import evaluate_model
from src.utils import create_reward_functions
from scripts.discrete_choice import posterior

sns.set_theme()

ALL_OBJECTS = ["A", "B", "C", "D"]
OPTIMAL_PATHS = list(
    {
        "left": np.array([0, 0, 0, 0, 0]),
        "up": np.array([1, 1, 1, 1, 1]),
        "right": np.array([2, 2, 2, 2, 2]),
    }.values()
)


def create_object_reward_functions(config):
    R = create_reward_functions(config["Ltot"], config["ratio"])
    return [
        {obj: R[i, j] for j, obj in enumerate(ALL_OBJECTS)} for i in range(config["Ltot"])
    ]


def generate_gridworld(config):
    # sample a random mapping from locations to objects
    sample = np.random.choice(ALL_OBJECTS, size=config["L"], replace=False)
    return {loc: sample[i] for i, loc in enumerate(["left", "up", "right"])}


def generate_trajectories_for_gridworld(gridworld, config):
    R = np.array([[r[obj] for obj in gridworld.values()] for r in config["R"]])
    R /= np.sum(R, axis=1, keepdims=True)
    P = boltzmann2d(R, config["beta"])
    return np.stack(
        [
            OPTIMAL_PATHS[np.random.choice(config["L"], p=P[config["z"][m]])]
            for m in range(config["M"])
        ]
    )


def generate_observations(config):
    trajectories = np.stack(
        [
            generate_trajectories_for_gridworld(generate_gridworld(config), config)
            for _ in range(config["N"])
        ]
    ).transpose([1, 0, 2])
    return mode(trajectories, axis=2)[0].squeeze()


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
    config["R"] = create_object_reward_functions(config)

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
