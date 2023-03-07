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


def error(partitions, probs, z_true):
    mean = np.sum(np.multiply(partitions, probs.reshape((-1, 1))), axis=0)
    mean = (mean - np.min(mean)) / (np.max(mean) - np.min(mean))
    true = (z_true - np.min(z_true)) / (np.max(z_true) - np.min(z_true))
    return mean, np.sum(np.square(mean - true))