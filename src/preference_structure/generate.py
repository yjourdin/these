from itertools import combinations
from typing import Any, Iterator, cast

import numpy as np
from mcda import PerformanceTable
from mcda.internal.core.values import Ranking
from mcda.relations import I, P, PreferenceStructure
from numpy.random import Generator

from ..generate_weak_order import (
    generate_partial_sum,
    random_ranking_with_tie_from_partial_sum,
)
from ..model import Model
from ..performance_table.dominance_relation import (
    dominance_relation,
    dominance_structure,
)
from ..relation import WeakOrder


def random_preference_relation(
    performance_table: PerformanceTable, rng: Generator, delta: float = 0.01
):
    m = len(performance_table.alternatives)
    S = generate_partial_sum(m, delta)

    dom_rel = dominance_relation(performance_table)

    cond = True
    while cond:
        cond = False

        ranking = random_ranking_with_tie_from_partial_sum(
            performance_table.alternatives, S, rng
        )
        preference_relation = WeakOrder.random_from_ranking(
            ranking, performance_table.alternatives
        )

        cond = not (dom_rel < preference_relation)

    return preference_relation.structure


def from_ranking(ranking: Ranking, nb: int | None = None, rng: Generator | None = None):
    result = PreferenceStructure()
    n = 0
    pairs: Iterator[tuple[Any, Any]] = combinations(ranking.labels, 2)
    ranks = ranking.data.to_dict()

    if nb:
        if rng:
            pairs = cast(
                Iterator[tuple[Any, Any]],
                rng.permutation(np.array(list(pairs))),
            )

    for a, b in pairs:
        if ranks[a] < ranks[b]:
            result._relations.append(P(a, b))
            n += 1
        elif ranks[a] == ranks[b]:
            result._relations.append(I(a, b))
            n += 1
        else:
            result._relations.append(P(b, a))
            n += 1

        if n == nb:
            break

    return result


def random_comparisons(
    alternatives: PerformanceTable,
    model: Model | None,
    nb: int | None = None,
    rng: Generator | None = None,
):
    if model:
        ranking = model.rank(alternatives)
        preference_structure = from_ranking(ranking, nb, rng)
    else:
        assert rng
        preference_structure = random_preference_relation(alternatives, rng)

    return preference_structure - dominance_structure(alternatives)


def noisy_comparisons(
    comparisons: PreferenceStructure, error_rate: float, rng: Generator
):
    result = PreferenceStructure()
    relations = comparisons.relations

    selected_relations = rng.choice(
        np.array(relations), int(error_rate * len(relations)), replace=False
    )
    relations = list(set(relations) - set(selected_relations))

    selected_indifferences = [r for r in selected_relations if isinstance(r, I)]
    nb_indifferences = len(selected_indifferences)

    selected_preferences = list(set(selected_relations) - set(selected_indifferences))
    nb_preferences = len(selected_preferences)

    changed_indifferences = [
        P(*rng.permutation(r.elements)) for r in selected_indifferences
    ]
    changed_preferences = [
        (
            I(*(r.elements))
            if rng.random() < (nb_indifferences / nb_preferences)
            else P(r.b, r.a)
        )
        for r in selected_preferences
    ]

    result._relations = relations + changed_indifferences + changed_preferences
    return result
