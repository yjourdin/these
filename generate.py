from itertools import chain, combinations, permutations
from subprocess import check_output
from typing import Any, cast

from numpy import arange, array, concatenate, diff, sort, triu_indices
from numpy.random import Generator
from pandas import DataFrame

from mcda_local.core.performance_table import NormalPerformanceTable, PerformanceTable
from mcda_local.core.ranker import Ranker
from mcda_local.core.relations import (
    IndifferenceRelation,
    PreferenceRelation,
    PreferenceStructure,
    Relation,
)
from mcda_local.core.values import Ranking
from mcda_local.ranker.rmp import RMP
from mcda_local.ranker.srmp import SRMP


def random_alternatives(nb_alt: int, nb_crit: int, rng: Generator) -> PerformanceTable:
    return NormalPerformanceTable(rng.random((nb_alt, nb_crit)))


def random_weights(nb_crit: int, rng: Generator) -> dict[Any, float]:
    return dict(
        zip(
            range(nb_crit), diff(sort(concatenate([[0], rng.random(nb_crit - 1), [1]])))
        )
    )


def random_capacities(nb_crit: int, rng: Generator) -> dict[frozenset[Any], float]:
    linext = eval(
        check_output(
            f"julia ./random_capacities.jl {nb_crit} {rng.integers(2**16)}", shell=True
        )
    )
    rng.random(nb_crit)
    crits = arange(nb_crit)
    return dict(
        zip(
            [frozenset(crits[array([bool(int(x)) for x in node])]) for node in linext],
            sort(rng.random(2**nb_crit)),
        )
    )


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
    weights = random_weights(nb_crit, rng)
    s = sum(weights.values())
    for c, w in weights.items():
        weights[c] = w / s
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
    weights = {k: 1 / nb_crit for k in range(nb_crit)}
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
    capacities = random_capacities(nb_crit, rng)
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

    crits = arange(nb_crit)
    power_set = chain.from_iterable(
        combinations(crits, r) for r in range(len(crits) + 1)
    )
    capacities = {frozenset(x): len(x) / nb_crit for x in power_set}
    lex_order = rng.permutation(nb_profiles)
    return RMP(capacities, profiles, lex_order.tolist())


def random_comparisons(
    nb: int,
    alternatives: PerformanceTable,
    model: Ranker,
    rng: Generator,
) -> PreferenceStructure:
    ranking = cast(Ranking, model.rank(alternatives))
    ranking_dict = ranking.data.to_dict()
    all_pairs = list(permutations(ranking.labels, 2))
    pairs = rng.choice(all_pairs, nb, replace=False)
    result: list[Relation] = []
    for a, b in pairs:
        if ranking_dict[a] < ranking_dict[b]:
            result.append(PreferenceRelation(a, b))
        elif ranking_dict[a] == ranking_dict[b]:
            result.append(IndifferenceRelation(a, b))
        else:
            result.append(PreferenceRelation(b, a))
    return PreferenceStructure(result)


def all_comparisons(
    alternatives: PerformanceTable, model: Ranker
) -> PreferenceStructure:
    ranking = cast(Ranking, model.rank(alternatives))
    ranking_dict = ranking.data.to_dict()
    # all_pairs = list(permutations(ranking.labels, 2))
    all_pairs = array(triu_indices(len(alternatives.data), 1)).transpose()
    result: list[Relation] = []
    for a, b in all_pairs:
        if ranking_dict[a] < ranking_dict[b]:
            result.append(PreferenceRelation(a, b))
        elif ranking_dict[a] == ranking_dict[b]:
            result.append(IndifferenceRelation(a, b))
        else:
            result.append(PreferenceRelation(b, a))
    return PreferenceStructure(result)


def noisy_comparisons(
    comparisons: PreferenceStructure, error_rate: float, rng: Generator
) -> PreferenceStructure:
    relations = comparisons.relations
    selected_relations = rng.choice(
        array(relations), int(error_rate * len(relations)), replace=False
    )
    relations = list(set(relations) - set(selected_relations))
    selected_indifferences = [
        r for r in selected_relations if isinstance(r, IndifferenceRelation)
    ]
    nb_indifferences = len(selected_indifferences)
    selected_preferences = list(set(selected_relations) - set(selected_indifferences))
    nb_preferences = len(selected_preferences)
    changed_indifferences = [
        PreferenceRelation(*rng.permutation(r.elements)) for r in selected_indifferences
    ]
    changed_preferences = [
        IndifferenceRelation(*(r.elements))
        if rng.random() < (nb_indifferences / nb_preferences)
        else PreferenceRelation(r.b, r.a)
        for r in selected_preferences
    ]
    return PreferenceStructure(relations + changed_indifferences + changed_preferences)
