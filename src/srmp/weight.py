import numpy as np
import numpy.typing as npt
from more_itertools import powerset
from scipy.stats import rankdata

from ..random import RNGParam, rng_
from ..utils import tolist


def random_weights(nb_crit: int, rng: RNGParam = None):
    return np.diff(
        np.pad(np.sort(rng_(rng).random(nb_crit - 1)), 1, constant_values=(0, 1))
    )


def normalize_weights(weights: npt.NDArray[np.float64]):
    weights[-1] = 1 - weights[:-1].sum()
    return weights


def frozen_importance_relation_from_weights(w: npt.NDArray[np.float64]):  # type: ignore
    power_sets = powerset(range(len(w)))
    result = []

    for set in power_sets:
        result.append(w[list(set)].sum())

    return tuple(tolist(rankdata(result, "dense")))  # type: ignore
