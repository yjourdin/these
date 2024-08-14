import ast
from subprocess import run
from typing import Any

import numpy as np
from more_itertools import powerset
from numpy.random import Generator

Capacity = dict[frozenset[Any], float]


def random_capacity(nb_crit: int, rng: Generator) -> dict[frozenset[Any], float]:
    linext = ast.literal_eval(
        run(
            [
                "julia",
                "src/rmp/random_capacity.jl",
                f"{nb_crit}",
                f"{rng.integers(2**16)}",
            ],
            capture_output=True,
            text=True,
        ).stdout
    )
    crits = np.arange(nb_crit)
    return dict(
        zip(
            [
                frozenset(crits[np.array([bool(int(x)) for x in node])])
                for node in linext
            ],
            np.sort(rng.random(2**nb_crit)),
        )
    )


def balanced_capacity(nb_crit: int):
    return {frozenset(x): len(x) / nb_crit for x in powerset(range(nb_crit))}
