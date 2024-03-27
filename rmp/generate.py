from itertools import chain, combinations
from subprocess import run
from typing import Any

import numpy as np
from mcda.core.matrices import PerformanceTable
from mcda.core.scales import NormalScale
from numpy.random import Generator
from pandas import DataFrame

from performance_table.normal_performance_table import NormalPerformanceTable

from .model import RMPModel


def random_capacities(nb_crit: int, rng: Generator) -> dict[frozenset[Any], float]:
    linext = eval(
        run(
            [
                "julia",
                "rmp/random_capacities.jl",
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


def random_rmp(
    nb_profiles: int,
    nb_crit: int,
    rng: Generator,
    profiles_values: PerformanceTable[NormalScale] | None = None,
) -> RMPModel:
    if profiles_values:
        idx = np.sort(rng.choice(len(profiles_values.data), (nb_profiles, nb_crit)))
        profiles = NormalPerformanceTable(
            DataFrame(
                {
                    i: profiles_values.data.iloc[idx[:, i], i].to_list()
                    for i in range(nb_crit)
                }
            )
        )
    else:
        profiles = NormalPerformanceTable(
            np.sort(rng.random((nb_profiles, nb_crit)), 0)
        )
    capacities = random_capacities(nb_crit, rng)
    lex_order = rng.permutation(nb_profiles)
    return RMPModel(profiles, capacities, lex_order.tolist())


def balanced_rmp(
    nb_profiles: int,
    nb_crit: int,
    rng: Generator,
    profiles_values: PerformanceTable[NormalScale] | None = None,
) -> RMPModel:
    if profiles_values:
        profiles = NormalPerformanceTable(
            profiles_values.data.iloc[
                [
                    int(i / (nb_profiles + 1) * len(profiles_values.alternatives))
                    for i in range(1, nb_profiles + 1)
                ],
                :,
            ],
        )
    else:
        profiles = NormalPerformanceTable(
            [[x / (nb_profiles + 1)] * nb_crit for x in range(1, nb_profiles + 1)]
        )

    crits = np.arange(nb_crit)
    power_set = chain.from_iterable(
        combinations(crits, r) for r in range(len(crits) + 1)
    )
    capacities = {frozenset(x): len(x) / nb_crit for x in power_set}
    lex_order = rng.permutation(nb_profiles)
    return RMPModel(profiles, capacities, lex_order.tolist())
