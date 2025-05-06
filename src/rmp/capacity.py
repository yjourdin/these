from typing import Any

import numpy as np
from more_itertools import powerset
from numpy.random import Generator

from ..julia.function import generate_linext
from ..random import seed
from ..utils import tolist

type Capacity[T: float] = dict[frozenset[Any], T]


def random_capacity(nb_crit: int, rng: Generator) -> Capacity[float]:
    linext = generate_linext(nb_crit, seed(rng))

    crits = np.arange(nb_crit)
    return dict(
        zip(
            [frozenset(tolist(crits[subset])) for subset in linext],
            np.sort(rng.random(2**nb_crit)),  # type: ignore
        )
    )


def balanced_capacity(nb_crit: int) -> Capacity[float]:
    return {frozenset(x): len(x) / nb_crit for x in powerset(range(nb_crit))}


def random_capacity_int(nb_crit: int, rng: Generator) -> Capacity[int]:
    linext = generate_linext(nb_crit, seed(rng))

    crits = np.arange(nb_crit)
    return dict(
        zip(
            [frozenset(crits[subset]) for subset in linext],
            np.sort(rng.integers(2**nb_crit, size=2**nb_crit)),  # type: ignore
        )
    )


def balanced_capacity_int(nb_crit: int) -> Capacity[int]:
    return {frozenset(x): len(x) for x in powerset(range(nb_crit))}
