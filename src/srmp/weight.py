import numpy as np
from more_itertools import powerset
from scipy.stats import rankdata

from src.random import RNGParam, rng_
from src.utils import tolist


def random_weights(nb_crit: int, rng: RNGParam = None):
    return np.diff(
        np.pad(np.sort(rng_(rng).random(nb_crit - 1)), 1, constant_values=(0, 1))
    )


def normalize_weights(weights: np.ndarray[tuple[int], np.dtype[np.float64]]):
    weights[-1] = 1 - weights[:-1].sum()
    return weights


def frozen_importance_relation_from_weights(
    w: np.ndarray[tuple[int], np.dtype[np.float64]],
):
    power_sets = powerset(range(len(w)))
    result: list[float] = []

    for set in power_sets:
        result.append(w[list(set)].sum())

    return tuple(tolist(np.array(rankdata(result)).astype(np.float64)))  # pyright: ignore[reportUnknownArgumentType]
