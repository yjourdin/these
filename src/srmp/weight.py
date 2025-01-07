import numpy as np
from more_itertools import powerset
from numpy.random import Generator
from scipy.stats import rankdata


def random_weights(nb_crit: int, rng: Generator) -> np.ndarray:
    return rng.dirichlet(np.ones(nb_crit))


def normalize_weights(weights: np.ndarray):
    weights[-1] = 1 - weights[:-1].sum()
    return weights


def frozen_importance_relation_from_weights(w: np.ndarray):
    power_sets = powerset(range(len(w)))
    result = []

    for set in power_sets:
        result.append(w[list(set)].sum())

    return tuple(rankdata(result, "dense").tolist())
