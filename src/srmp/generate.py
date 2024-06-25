import numpy as np
from mcda.internal.core.scales import NormalScale
from mcda.matrices import PerformanceTable
from numpy.random import Generator

from ..performance_table.normal_performance_table import NormalPerformanceTable
from .model import SRMPModel


def random_weights(nb_crit: int, rng: Generator) -> list[float]:
    return np.diff(
        np.sort(np.concatenate([[0], rng.random(nb_crit - 1), [1]]))
    ).tolist()


def random_srmp(
    nb_profiles: int,
    nb_crit: int,
    rng: Generator,
    profiles_values: PerformanceTable[NormalScale] | None = None,
):
    if profiles_values:
        idx = np.sort(rng.choice(len(profiles_values.data), (nb_profiles, nb_crit)), 0)
        profiles = NormalPerformanceTable(
            np.array(
                [
                    profiles_values.data.iloc[idx[:, i], i].to_numpy()
                    for i in range(nb_crit)
                ]
            ).transpose()
        )
    else:
        profiles = NormalPerformanceTable(
            np.sort(rng.random((nb_profiles, nb_crit)), 0)
        )

    weights = random_weights(nb_crit, rng)

    lexicographic_order = rng.permutation(nb_profiles).tolist()

    return SRMPModel(profiles, weights, lexicographic_order)


def balanced_srmp(
    nb_profiles: int,
    nb_crit: int,
    rng: Generator,
    profiles_values: PerformanceTable[NormalScale] | None = None,
):
    if profiles_values:
        profiles = NormalPerformanceTable(
            profiles_values.data.iloc[
                [
                    int(i / (nb_profiles + 1) * len(profiles_values.alternatives))
                    for i in range(1, nb_profiles + 1)
                ],
                :,
            ]
        )
    else:
        profiles = NormalPerformanceTable(
            np.sort(rng.random((nb_profiles, nb_crit)), 0)
        )

    weights = [1 / nb_crit] * nb_crit

    lexicographic_order = rng.permutation(nb_profiles).tolist()

    return SRMPModel(profiles, weights, lexicographic_order)
