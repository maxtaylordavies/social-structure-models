from itertools import permutations

import numpy as np

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
    return np.random.choice(K, size=M)

def boltzmann(r, beta):
    p = np.exp(r / beta)
    return p / np.sum(p)

# normalise a numpy ndarray to the range [0, 1]
def normalise(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def posterior_mean(partitions, probs):
    return np.sum(np.multiply(partitions, probs.reshape((-1, 1))), axis=0)


def map_estimate(partitions, probs):
    return partitions[np.argmax(probs)]


def error(z, z_true):
    z, true = normalise(z), normalise(z_true)
    return np.square(z - true).mean()