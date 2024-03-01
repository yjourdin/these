from mcda.core.matrices import PerformanceTable
from mcda.core.scales import NormalScale
from numpy import array, concatenate, diff, sort
from numpy.random import Generator

from performance_table.core import NormalPerformanceTable

from .model import SRMPModel


def random_weights(nb_crit: int, rng: Generator) -> list[float]:
    return diff(sort(concatenate([[0], rng.random(nb_crit - 1), [1]]))).tolist()


def random_srmp(
    nb_profiles: int,
    nb_crit: int,
    rng: Generator,
    profiles_values: PerformanceTable[NormalScale] | None = None,
):
    if profiles_values:
        idx = sort(rng.choice(len(profiles_values.data), (nb_profiles, nb_crit)), 0)
        profiles = NormalPerformanceTable(
            array(
                [
                    profiles_values.data.iloc[idx[:, i], i].to_numpy()
                    for i in range(nb_crit)
                ]
            ).transpose()
        )
    else:
        profiles = NormalPerformanceTable(sort(rng.random((nb_profiles, nb_crit)), 0))

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
        profiles = NormalPerformanceTable(sort(rng.random((nb_profiles, nb_crit)), 0))

    weights = [1 / nb_crit] * nb_crit

    lexicographic_order = rng.permutation(nb_profiles).tolist()

    return SRMPModel(profiles, weights, lexicographic_order)
