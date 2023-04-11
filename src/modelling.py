import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import (
    random_partition,
    generate_all_partitions,
    posterior_mean,
    map_estimate,
    error,
    log,
)


def evaluate_model(
    config,
    gen_obs_func,
    posterior_func,
    partitions=None,
    true_partitions=None,
    plot_func=None,
):
    results = {"beta": [], "N": [], "mean": [], "map": [], "err_mean": [], "err_map": []}

    if partitions is None:
        if config["sample"]:
            partitions = [random_partition(config["K"], config["M"]) for _ in range(1000)]
            partitions = np.array(partitions)
        else:
            partitions = generate_all_partitions(config["M"])

    if true_partitions is None:
        true_partitions = [random_partition(config["K"], config["M"]) for _ in range(config["repeats"])]

    beta_vals = config["betas"]

    for rep in tqdm(range(config["repeats"])):
        config["z"] = true_partitions[rep]
        if config["sample"]:
            partitions[-1] = config["z"]

        for beta in beta_vals:
            config["beta"] = beta

            O = gen_obs_func(config)

            priors, n_vals = None, np.arange(0, config["N"] + 1, config["update_batch_size"])
            for n in n_vals:
                post = posterior_func(
                    O[:, max(n - config["update_batch_size"], 0) : n],
                    partitions,
                    config,
                    priors=priors,
                )
                priors = post

                mean, _map = posterior_mean(partitions, post), map_estimate(partitions, post)

                results["beta"].append(beta)
                results["N"].append(n)
                results["mean"].append(mean)
                results["map"].append(_map)
                results["err_mean"].append(error(mean, config["z"]))
                results["err_map"].append(error(_map, config["z"]))

        if plot_func:
            plot_func(pd.DataFrame(results), config)

    return pd.DataFrame(results)
