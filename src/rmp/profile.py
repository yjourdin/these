import numpy as np
from mcda.internal.core.scales import NormalScale
from mcda.matrices import PerformanceTable
from numpy.random import Generator
from pandas import DataFrame

from ..performance_table.normal_performance_table import NormalPerformanceTable


def random_profiles(
    nb: int,
    nb_crit: int,
    rng: Generator,
    profiles_values: PerformanceTable[NormalScale] | None = None,
):
    if profiles_values:
        idx = np.sort(rng.choice(len(profiles_values.data), (nb, nb_crit)))
        return NormalPerformanceTable(
            DataFrame(
                {
                    i: profiles_values.data.iloc[idx[:, i], i].to_list()
                    for i in range(nb_crit)
                }
            )
        )
    else:
        return NormalPerformanceTable(np.sort(rng.random((nb, nb_crit)), 0))


def balanced_profiles(
    nb_profiles: int,
    nb_crit: int,
    profiles_values: PerformanceTable[NormalScale] | None = None,
):
    if profiles_values:
        return NormalPerformanceTable(
            profiles_values.data.iloc[
                [
                    int(i / (nb_profiles + 1) * len(profiles_values.alternatives))
                    for i in range(1, nb_profiles + 1)
                ],
                :,
            ],
        )
    else:
        return NormalPerformanceTable(
            [[x / (nb_profiles + 1)] * nb_crit for x in range(1, nb_profiles + 1)]
        )
