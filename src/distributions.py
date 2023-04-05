from collections import Counter

import numpy as np
from scipy.special import gamma


def boltzmann1d(r, beta):
    p = np.exp(r / beta)
    return p / np.sum(p)


def boltzmann2d(r, beta):
    p = np.exp(r / beta)
    return p / np.sum(p, axis=1).reshape((-1, 1))


def crp(z, c):
    counts = Counter(z)
    prod = np.prod([gamma(counts[k]) for k in counts])
    coeff = ((c ** len(counts)) * gamma(c)) / gamma(c + len(z))
    return coeff * prod


def dirichlet_multinomial(counts, T, L, g, log=True):
    coeff = (gamma(L * g) * gamma(T + 1)) / gamma(T + (L * g))
    prods = np.prod(gamma(counts + g) / (gamma(g) * gamma(counts + 1)), axis=1)

    tmp = coeff * prods
    return np.sum(np.log(tmp)) if log else np.prod(tmp)
