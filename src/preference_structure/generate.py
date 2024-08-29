from itertools import combinations
from typing import Any, Iterator, cast

import numpy as np
from mcda import PerformanceTable
from mcda.internal.core.values import Ranking
from mcda.relations import I, P, PreferenceStructure
from numpy.random import Generator

from ..model import Model
from ..performance_table.dominance_relation import DominanceRelation, dominance_relation


def from_ranking(
    ranking: Ranking,
    nb: int | None = None,
    rng: Generator | None = None,
    dominance_relation: DominanceRelation | None = None,
    check_dominance=False,
    remove_dominance=False,
):
    result = PreferenceStructure()
    n = 0
    pairs: Iterator[tuple[Any, Any]] = combinations(ranking, 2)
    ranks = ranking.data.to_dict()
    if nb:
        if rng:
            pairs = cast(
                Iterator[tuple[Any, Any]],
                rng.permutation(np.array(pairs)),
            )
    for a, b in pairs:
        if remove_dominance:
            assert dominance_relation
            if (a, b) in dominance_relation:
                continue
        if ranks[a] < ranks[b]:
            if check_dominance:
                assert dominance_relation
                if (b, a) in dominance_relation:
                    raise ValueError("ranking does not respect dominance")
            result._relations.append(P(a, b))
            n += 1
        elif ranks[a] == ranks[b]:
            if check_dominance:
                assert dominance_relation
                if ((a, b) in dominance_relation) or ((b, a) in dominance_relation):
                    raise ValueError("ranking does not respect dominance")
            result._relations.append(I(a, b))
            n += 1
        else:
            if check_dominance:
                assert dominance_relation
                if (a, b) in dominance_relation:
                    raise ValueError("ranking does not respect dominance")
            result._relations.append(P(b, a))
            n += 1
        if n == nb:
            break
    return result


def random_comparisons(
    alternatives: PerformanceTable,
    model: Model,
    nb: int | None = None,
    rng: Generator | None = None,
):
    return from_ranking(
        model.rank(alternatives),
        nb=nb,
        rng=rng,
        dominance_relation=dominance_relation(alternatives),
        remove_dominance=True,
    )


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
