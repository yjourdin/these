from itertools import product
from typing import cast

from numpy import sort
from numpy.random import Generator, SeedSequence, default_rng
from pandas import DataFrame

from mcda_local.core.performance_table import NormalPerformanceTable, PerformanceTable
from mcda_local.core.power_set import PowerSet
from mcda_local.core.ranker import Ranker
from mcda_local.core.relations import IndifferenceRelation, PreferenceRelation, Relation
from mcda_local.core.values import Ranking
from mcda_local.ranker.rmp import RMP
from mcda_local.ranker.srmp import SRMP


def random_generator(seed: int | None = None):
    ss = SeedSequence(seed)
    seed = cast(int, ss.entropy)
    return default_rng(ss), seed


def random_alternatives(nb_alt: int, nb_crit: int, rng: Generator) -> PerformanceTable:
    return NormalPerformanceTable(rng.random((nb_alt, nb_crit)))


def random_srmp(
    nb_profiles: int,
    nb_crit: int,
    rng: Generator,
    profiles_values: PerformanceTable | None = None,
):
    if profiles_values:
        idx = sort(rng.choice(len(profiles_values.data), (nb_profiles, nb_crit)), 0)
        profiles = PerformanceTable(
            DataFrame(
                [
                    profiles_values.data.iloc[idx[:, i], i].to_numpy()
                    for i in range(nb_crit)
                ]
            ).transpose(),
            profiles_values.scales,
        )
    else:
        profiles = NormalPerformanceTable(sort(rng.random((nb_profiles, nb_crit)), 0))
    weights = dict(enumerate(rng.random(nb_crit)))
    lex_order = rng.permutation(nb_profiles)
    return SRMP(weights, profiles, lex_order.tolist())


def balanced_srmp(
    nb_profiles: int,
    nb_crit: int,
    rng: Generator,
    profiles_values: PerformanceTable | None = None,
):
    if profiles_values:
        profiles = PerformanceTable(
            profiles_values.data.iloc[
                [
                    int(i / (nb_profiles + 1) * len(profiles_values.alternatives))
                    for i in range(1, nb_profiles + 1)
                ],
                :,
            ],
            profiles_values.scales,
        )
    else:
        profiles = NormalPerformanceTable(sort(rng.random((nb_profiles, nb_crit)), 0))
    weights = dict(enumerate(rng.random(nb_crit)))
    lex_order = rng.permutation(nb_profiles)
    return SRMP(weights, profiles, lex_order.tolist())


def random_rmp(
    nb_profiles: int,
    nb_crit: int,
    rng: Generator,
    profiles_values: PerformanceTable | None = None,
) -> RMP:
    if profiles_values:
        idx = sort(rng.choice(len(profiles_values.data), (nb_profiles, nb_crit)))
        profiles = PerformanceTable(
            DataFrame(
                {
                    i: profiles_values.data.iloc[idx[:, i], i].to_list()
                    for i in range(nb_crit)
                }
            ),
            profiles_values.scales,
        )
    else:
        profiles = NormalPerformanceTable(sort(rng.random((nb_profiles, nb_crit)), 0))
    capacities = PowerSet(list(range(nb_crit)))
    for ss in capacities.keys():
        capacities[ss] = rng.integers(
            capacities.min_capacity(ss), capacities.max_capacity(ss), endpoint=True
        )
    lex_order = rng.permutation(nb_profiles)
    return RMP(capacities, profiles, lex_order.tolist())


def balanced_rmp(
    nb_profiles: int,
    nb_crit: int,
    rng: Generator,
    profiles_values: PerformanceTable | None = None,
) -> RMP:
    if profiles_values:
        profiles = PerformanceTable(
            profiles_values.data.iloc[
                [
                    int(i / (nb_profiles + 1) * len(profiles_values.alternatives))
                    for i in range(1, nb_profiles + 1)
                ],
                :,
            ],
            profiles_values.scales,
        )
    else:
        profiles = NormalPerformanceTable(
            [[x / (nb_profiles + 1)] * nb_crit for x in range(1, nb_profiles + 1)]
        )
    capacities = PowerSet(list(range(nb_crit)))
    for ss in capacities.keys():
        capacities[ss] = rng.integers(
            capacities.min_capacity(ss), capacities.max_capacity(ss), endpoint=True
        )
    lex_order = rng.permutation(nb_profiles)
    return RMP(capacities, profiles, lex_order.tolist())


def random_comparisons(
    nb: int,
    alternatives: PerformanceTable,
    model: Ranker,
    rng: Generator,
) -> list[Relation]:
    ranking = cast(Ranking, model.rank(alternatives))
    all_pairs = list(product(ranking.labels, repeat=2))
    pairs = rng.choice(all_pairs, nb, replace=False)
    preference_stucture: list[Relation] = []
    for a, b in pairs:
        rank_a = ranking.data[a]
        rank_b = ranking.data[b]
        if rank_a < rank_b:
            preference_stucture.append(PreferenceRelation(a, b))
        elif rank_a == rank_b:
            preference_stucture.append(IndifferenceRelation(a, b))
        else:
            preference_stucture.append(PreferenceRelation(b, a))
    return preference_stucture


def all_comparisons(alternatives: PerformanceTable, model: Ranker) -> list[Relation]:
    ranking = cast(Ranking, model.rank(alternatives))
    all_pairs = list(product(ranking.labels, repeat=2))
    result: list[Relation] = []
    for a, b in all_pairs:
        # print(f"{a}, {b}")
        if ranking.data[a] < ranking.data[b]:
            result.append(PreferenceRelation(a, b))
        elif ranking.data[a] == ranking.data[b]:
            result.append(IndifferenceRelation(a, b))
        else:
            result.append(PreferenceRelation(b, a))
    return result
