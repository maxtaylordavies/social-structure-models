from datetime import datetime

from itertools import permutations

import numpy as np
from tqdm import tqdm


def log(msg, use_tqdm=True):
    # log current time to millisecond precision
    msg = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}"
    if use_tqdm:
        tqdm.write(msg)
    else:
        print(msg)


#  helper function to generate all possible partitions for the case of <= 3 clusters
# returns a list of length 3^M
def generate_all_partitions(M):
    partitions = set()
    for i in range(0, M + 1):
        k = M - i
        for j in range(0, k + 1):
            base = ([0] * i) + ([1] * j) + ([2] * (k - j))
            partitions = partitions.union(set(permutations(base)))

    partitions = np.array(list(partitions)).astype(np.float64)
    for i in range(len(partitions)):
        k = len(set(partitions[i])) - 1
        row = partitions[i] - np.min(partitions[i])
        if np.max(row) > 0:
            row *= k / np.max(row)
        if row[0] > row[-1]:
            row = row[::-1]
        partitions[i] = row

    return partitions


def random_partition(K, M):
    return order_partition(np.random.choice(K, size=M))


def order_partition(z):
    # return a new partition with ordered clusters
    # e.g. [2, 1, 1, 0, 0, 0] -> [0, 1, 1, 2, 2, 2]
    firsts = {}
    for i, k in enumerate(z):
        if k not in firsts:
            firsts[k] = i

    unique, new = sorted(firsts, key=firsts.get), np.zeros_like(z)
    for i, k in enumerate(unique):
        new[z == k] = i

    return new


def create_reward_functions(L, ratio):
    base = np.array([1 * (ratio**i) for i in range(L)])
    # base /= np.sum(base)

    r = np.zeros((L, L))
    for i in range(L):
        r[i] = np.roll(base, i)

    return r


# normalise a numpy ndarray to the range [0, 1]
def normalise(x):
    return np.nan_to_num((x - np.min(x)) / (np.max(x) - np.min(x)))


def posterior_mean(partitions, probs):
    return np.sum(np.multiply(partitions, probs.reshape((-1, 1))), axis=0)


def map_estimate(partitions, probs):
    return order_partition(partitions[np.argmax(probs)])


def error(z, z_true):
    z, z_true = normalise(z), normalise(z_true)
    return np.square(z - z_true).mean()
