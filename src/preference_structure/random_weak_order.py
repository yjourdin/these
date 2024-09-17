from functools import cache

from mcda import PerformanceTable
from mcda.internal.core.values import Ranking
from mcda.relations import PreferenceStructure
from mcda.scales import DiscreteQuantitativeScale, PreferenceDirection
from mcda.values import CommensurableValues
from numpy.random import Generator
from pandas import Series

from ..performance_table.dominance_relation import dominance_relation


@cache
def w(m: int, k: int):
    if k > m:
        return 0
    if k == 1:
        return 1
    return k * w(m - 1, k) + w(m - 1, k - 1)


def W(m: int):
    return sum(w(m, k) for k in range(1, m + 1))


def generate_partial_sum(m: int, Wm: int, delta: float = 0.01):
    k = 0
    S: list[float] = [0]

    while Wm - S[-1] >= delta:
        k += 1
        S.append(S[-1] + (k**m) / (2 ** (k + 1)))

    return S


def random_nb_blocks(Wm: int, S: list[float], rng: Generator):

    Y = rng.uniform(0, Wm)

    for K, Sk in enumerate(S):
        if Sk >= Y:
            break

    return K


def random_ranking(alternatives: list, k: int | None, rng: Generator):
    m = len(alternatives)
    k = k or m

    ranks = rng.integers(k, size=m, endpoint=True)

    return CommensurableValues(
        Series(ranks, alternatives),
        scale=DiscreteQuantitativeScale(
            ranks.tolist(),
            PreferenceDirection.MIN,
        ),
    )


def random_ranking_with_tie(
    alternatives: list, k: int | None, rng: Generator, delta: float = 0.01
):
    m = len(alternatives)
    Wm = W(m)
    S = generate_partial_sum(m, Wm, delta)
    K = random_nb_blocks(Wm, S, rng)

    return random_ranking(alternatives, K, rng)


def dominance_compliant(ranking: Ranking, dominance_structure: PreferenceStructure):
    for r in dominance_structure:
        if ranking[r.a] >= ranking[r.b]:
            return False
    return True


def random_preference_relation(
    performance_table: PerformanceTable, rng: Generator, delta: float = 0.01
):
    m = len(performance_table.alternatives)
    Wm = W(m)
    S = generate_partial_sum(m, Wm, delta)

    dominance_structure = dominance_relation(performance_table)

    cond = True
    while cond:
        cond = False

        K = random_nb_blocks(Wm, S, rng)

        ranking = random_ranking(performance_table.alternatives, K, rng)

        cond = not dominance_compliant(ranking, dominance_structure)

    return ranking
