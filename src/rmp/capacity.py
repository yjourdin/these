from typing import Any

import numpy as np
from more_itertools import powerset

from src.julia.function import generate_linext
from src.random import RNGParam, int_, rng_
from src.utils import tolist

type Capacity[T: float] = dict[frozenset[Any], T]


def random_capacity(nb_crit: int, rng: RNGParam = None) -> Capacity[float]:
    linext = generate_linext(nb_crit, int_(rng))

    crits = np.arange(nb_crit)
    return dict(
        zip(
            [frozenset(tolist(crits[subset])) for subset in linext],
            np.sort(rng_(rng).random(int(2**nb_crit))),
        )
    )


def balanced_capacity(nb_crit: int) -> Capacity[float]:
    return {frozenset(x): len(x) / nb_crit for x in powerset(range(nb_crit))}


def random_capacity_int(nb_crit: int, rng: RNGParam = None) -> Capacity[int]:
    linext = generate_linext(nb_crit, int_(rng))

    crits = np.arange(nb_crit)
    return dict(
        zip(
            [frozenset(crits[subset]) for subset in linext],
            np.sort(rng_(rng).integers(int(2**nb_crit), size=int(2**nb_crit))),
        )
    )


def balanced_capacity_int(nb_crit: int) -> Capacity[int]:
    return {frozenset(x): len(x) for x in powerset(range(nb_crit))}
